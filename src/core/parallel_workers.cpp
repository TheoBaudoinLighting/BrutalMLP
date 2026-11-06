// Worker-count resolution and thread joining for coarse parallelism.

#include "core/parallel_workers.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>


namespace brutal_mlp::detail {

[[nodiscard]] std::size_t resolve_parallel_worker_count(const ParallelOptions& options,
                                                        std::size_t sample_count) noexcept {
    if (sample_count == 0 || options.execution != ParallelExecution::worker_threads) {
        return sample_count == 0 ? 0 : 1;
    }
    if (options.minimum_parallel_samples > 0 && sample_count < options.minimum_parallel_samples) {
        return 1;
    }

    std::size_t worker_count = options.thread_count;
    if (worker_count == 0) {
        worker_count = static_cast<std::size_t>(std::thread::hardware_concurrency());
    }
    if (worker_count == 0) {
        worker_count = 1;
    }
    return std::min(worker_count, sample_count);
}

[[nodiscard]] std::size_t required_batch_scratch_size(std::size_t single_sample_scratch,
                                                      std::size_t sample_count,
                                                      const ParallelOptions& options) noexcept {
    const std::size_t worker_count = resolve_parallel_worker_count(options, sample_count);
    if (single_sample_scratch != 0 &&
        worker_count > std::numeric_limits<std::size_t>::max() / single_sample_scratch) {
        return std::numeric_limits<std::size_t>::max();
    }
    return single_sample_scratch * worker_count;
}

void join_workers_noexcept(std::vector<std::thread>& workers) noexcept {
    for (std::thread& worker : workers) {
        if (!worker.joinable()) {
            continue;
        }
        try {
            worker.join();
        } catch (...) {
            if (worker.joinable()) {
                try {
                    worker.detach();
                } catch (...) {
                }
            }
        }
    }
}


struct WorkerPool::State {
    std::mutex mutex;
    std::condition_variable wake;      // Signals a new generation (or shutdown) to helpers.
    std::condition_variable finished;  // Signals the caller that every helper left the generation.
    std::uint64_t generation{0};
    bool stopping{false};
    std::size_t helpers_in_generation{0};

    // Description of the current run, written under `mutex` before `generation` is incremented.
    ErasedTask task{nullptr};
    void* context{nullptr};
    std::size_t task_count{0};
    std::atomic<std::size_t> next_task{0};

    std::mutex error_mutex;
    std::exception_ptr error;
    std::size_t error_task{0};  // Task index of `error`: the lowest index that threw.
};

WorkerPool::WorkerPool(std::size_t worker_count)
    : state_(std::make_unique<State>()) {
    const std::size_t helper_count = worker_count > 1 ? worker_count - 1 : 0;
    try {
        threads_.reserve(helper_count);
        for (std::size_t helper = 0; helper < helper_count; ++helper) {
            threads_.emplace_back([this, helper] { worker_loop(helper + 1); });
        }
    } catch (...) {
        // Fewer threads than requested only changes speed: tasks run on whoever is available.
    }
}

WorkerPool::~WorkerPool() {
    {
        const std::lock_guard<std::mutex> lock(state_->mutex);
        state_->stopping = true;
    }
    state_->wake.notify_all();
    join_workers_noexcept(threads_);
}

std::size_t WorkerPool::worker_count() const noexcept {
    return threads_.size() + 1;
}

void WorkerPool::run_erased(std::size_t task_count, ErasedTask task, void* context) {
    if (task_count == 0) {
        return;
    }
    if (threads_.empty() || task_count == 1) {
        for (std::size_t task_index = 0; task_index < task_count; ++task_index) {
            task(context, task_index, 0);
        }
        return;
    }

    {
        const std::lock_guard<std::mutex> lock(state_->mutex);
        state_->task = task;
        state_->context = context;
        state_->task_count = task_count;
        state_->next_task.store(0, std::memory_order_relaxed);
        state_->error = nullptr;
        state_->helpers_in_generation = threads_.size();
        ++state_->generation;
    }
    state_->wake.notify_all();

    execute_tasks(0);

    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->finished.wait(lock, [this] { return state_->helpers_in_generation == 0; });
        const std::lock_guard<std::mutex> error_lock(state_->error_mutex);
        error = state_->error;
        state_->error = nullptr;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

void WorkerPool::worker_loop(std::size_t worker_index) {
    std::uint64_t seen_generation = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(state_->mutex);
            state_->wake.wait(lock, [&] { return state_->stopping || state_->generation != seen_generation; });
            if (state_->stopping) {
                return;
            }
            seen_generation = state_->generation;
        }

        execute_tasks(worker_index);

        {
            const std::lock_guard<std::mutex> lock(state_->mutex);
            --state_->helpers_in_generation;
            if (state_->helpers_in_generation == 0) {
                state_->finished.notify_one();
            }
        }
    }
}

void WorkerPool::execute_tasks(std::size_t worker_index) noexcept {
    for (;;) {
        const std::size_t task_index = state_->next_task.fetch_add(1, std::memory_order_relaxed);
        if (task_index >= state_->task_count) {
            return;
        }
        try {
            state_->task(state_->context, task_index, worker_index);
        } catch (...) {
            // Keep the failure of the lowest task index, so the reported error does not depend
            // on which worker failed first.
            const std::lock_guard<std::mutex> lock(state_->error_mutex);
            if (!state_->error || task_index < state_->error_task) {
                state_->error = std::current_exception();
                state_->error_task = task_index;
            }
        }
    }
}

} // namespace brutal_mlp::detail
