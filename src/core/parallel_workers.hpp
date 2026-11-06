#pragma once

// Internal header: Worker-count resolution and thread joining for coarse parallelism.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

namespace brutal_mlp::detail {

[[nodiscard]] std::size_t resolve_parallel_worker_count(const ParallelOptions& options,
                                                        std::size_t sample_count) noexcept;

[[nodiscard]] std::size_t required_batch_scratch_size(std::size_t single_sample_scratch,
                                                      std::size_t sample_count,
                                                      const ParallelOptions& options) noexcept;

void join_workers_noexcept(std::vector<std::thread>& workers) noexcept;

// Fixed set of threads reused for many parallel loops (one fit call, for instance), so that
// no thread is created per loop. run() executes task(task_index, worker_index) for every task
// index in [0, task_count); the calling thread takes part as worker 0. Tasks are claimed in any
// order by any worker: callers that need deterministic results must make each task's output
// independent of which worker runs it (for instance one output buffer per task).
class WorkerPool {
public:
    // `worker_count` counts the calling thread; 0 and 1 both mean "run everything inline".
    explicit WorkerPool(std::size_t worker_count);
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    [[nodiscard]] std::size_t worker_count() const noexcept;

    // Blocks until every task finished. If tasks throw, rethrows the first captured exception
    // after all workers stopped touching the task data.
    template <typename Task>
    void run(std::size_t task_count, Task& task) {
        run_erased(task_count,
                   [](void* context, std::size_t task_index, std::size_t worker_index) {
                       (*static_cast<Task*>(context))(task_index, worker_index);
                   },
                   &task);
    }

private:
    using ErasedTask = void (*)(void* context, std::size_t task_index, std::size_t worker_index);

    void run_erased(std::size_t task_count, ErasedTask task, void* context);
    void worker_loop(std::size_t worker_index);
    void execute_tasks(std::size_t worker_index) noexcept;

    struct State;
    std::unique_ptr<State> state_;
    std::vector<std::thread> threads_;
};

} // namespace brutal_mlp::detail
