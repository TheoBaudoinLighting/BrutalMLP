// Regression and classification metrics.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/checks.hpp"
#include "data/data_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace brutal_mlp::detail {

namespace {

[[nodiscard]] bool is_regression_metric(Metric metric) noexcept {
    return metric == Metric::mean_squared_error ||
           metric == Metric::mean_absolute_error ||
           metric == Metric::root_mean_squared_error ||
           metric == Metric::r2_score;
}

[[nodiscard]] bool is_classification_metric(Metric metric) noexcept {
    return metric == Metric::accuracy ||
           metric == Metric::precision ||
           metric == Metric::recall ||
           metric == Metric::f1_score ||
           metric == Metric::confusion_matrix;
}

[[nodiscard]] bool contains_metric(const std::vector<Metric>& metrics, Metric metric) {
    return std::find(metrics.begin(), metrics.end(), metric) != metrics.end();
}

void validate_evaluation_options(const EvaluationOptions& options) {
    require_finite(options.classification_threshold, "classification_threshold");
    require(options.classification_threshold >= Scalar{0} && options.classification_threshold <= Scalar{1},
            "classification_threshold must be in [0, 1]");

    for (std::size_t i = 0; i < options.metrics.size(); ++i) {
        require(options.metrics[i] != Metric::custom, "custom metrics must be registered with add_custom_metric");
        for (std::size_t j = i + 1; j < options.metrics.size(); ++j) {
            require(options.metrics[i] != options.metrics[j], "duplicate metric requested");
        }
    }

    for (std::size_t i = 0; i < options.custom_metrics.size(); ++i) {
        const CustomMetric& metric = options.custom_metrics[i];
        require(!metric.name.empty(), "custom metric name must not be empty");
        require(metric.evaluate != nullptr, "custom metric callback is required");
        for (std::size_t j = i + 1; j < options.custom_metrics.size(); ++j) {
            require(metric.name != options.custom_metrics[j].name, "duplicate custom metric name");
        }
    }
}

[[nodiscard]] std::vector<Metric> selected_metrics(const EvaluationOptions& options) {
    if (!options.metrics.empty() || !options.custom_metrics.empty()) {
        return options.metrics;
    }
    return EvaluationOptions::regression().metrics;
}

void validate_evaluation_data(const Matrix& predictions,
                              const Matrix& targets) {
    validate_matrix(predictions, predictions.empty() ? 0 : predictions.front().size(), "predictions");
    require(!predictions.front().empty(), "predictions must have at least one column");
    validate_matrix(targets, predictions.front().size(), "targets");
    require(predictions.size() == targets.size(), "predictions and targets must have the same number of rows");
}

void validate_evaluation_data(const DenseMatrix& predictions,
                              const DenseMatrix& targets) {
    validate_dense_matrix(predictions, predictions.columns(), "predictions");
    require(predictions.columns() > 0, "predictions must have at least one column");
    validate_dense_matrix(targets, predictions.columns(), "targets");
    require(predictions.rows() == targets.rows(), "predictions and targets must have the same number of rows");
}

[[nodiscard]] std::size_t argmax_index(const Vector& values) {
    return static_cast<std::size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

[[nodiscard]] std::size_t argmax_index(const Scalar* values, std::size_t size) {
    require(values != nullptr, "values buffer must not be null");
    return static_cast<std::size_t>(std::distance(values, std::max_element(values, values + size)));
}

[[nodiscard]] std::size_t class_count_for_output_size(std::size_t output_size) noexcept {
    return output_size == 1 ? std::size_t{2} : output_size;
}

[[nodiscard]] std::size_t class_label_from_row(const Vector& row, Scalar threshold) {
    if (row.size() == 1) {
        return row[0] >= threshold ? std::size_t{1} : std::size_t{0};
    }
    return argmax_index(row);
}

[[nodiscard]] std::size_t class_label_from_row(const Scalar* row, std::size_t size, Scalar threshold) {
    require(row != nullptr, "classification row buffer must not be null");
    if (size == 1) {
        return row[0] >= threshold ? std::size_t{1} : std::size_t{0};
    }
    return argmax_index(row, size);
}

[[nodiscard]] ConfusionMatrix make_confusion_matrix(const Matrix& predictions,
                                                    const Matrix& targets,
                                                    Scalar threshold) {
    const std::size_t class_count = class_count_for_output_size(predictions.front().size());
    ConfusionMatrix matrix;
    matrix.counts.assign(class_count, std::vector<std::size_t>(class_count, 0));

    for (std::size_t row = 0; row < predictions.size(); ++row) {
        const std::size_t actual = class_label_from_row(targets[row], threshold);
        const std::size_t predicted = class_label_from_row(predictions[row], threshold);
        ++matrix.counts[actual][predicted];
    }
    return matrix;
}

[[nodiscard]] ConfusionMatrix make_confusion_matrix(const DenseMatrix& predictions,
                                                    const DenseMatrix& targets,
                                                    Scalar threshold) {
    const std::size_t output_size = predictions.columns();
    const std::size_t class_count = class_count_for_output_size(output_size);
    ConfusionMatrix matrix;
    matrix.counts.assign(class_count, std::vector<std::size_t>(class_count, 0));

    for (std::size_t row = 0; row < predictions.rows(); ++row) {
        const std::size_t actual = class_label_from_row(targets.row_data(row), output_size, threshold);
        const std::size_t predicted = class_label_from_row(predictions.row_data(row), output_size, threshold);
        ++matrix.counts[actual][predicted];
    }
    return matrix;
}

[[nodiscard]] Scalar ratio_or_zero(std::size_t numerator, std::size_t denominator) noexcept {
    return denominator == 0 ? Scalar{0} : static_cast<Scalar>(numerator) / static_cast<Scalar>(denominator);
}

struct ClassificationSummary {
    Scalar accuracy{Scalar{0}};
    Scalar precision{Scalar{0}};
    Scalar recall{Scalar{0}};
    Scalar f1_score{Scalar{0}};
};

struct ClassCounts {
    std::size_t true_positive{0};
    std::size_t false_positive{0};
    std::size_t false_negative{0};
};

[[nodiscard]] ClassCounts counts_for_class(const ConfusionMatrix& matrix, std::size_t class_index) {
    ClassCounts counts;
    counts.true_positive = matrix.counts[class_index][class_index];

    for (std::size_t i = 0; i < matrix.counts.size(); ++i) {
        if (i != class_index) {
            counts.false_positive += matrix.counts[i][class_index];
            counts.false_negative += matrix.counts[class_index][i];
        }
    }
    return counts;
}

[[nodiscard]] Scalar precision_from_counts(const ClassCounts& counts) noexcept {
    return ratio_or_zero(counts.true_positive, counts.true_positive + counts.false_positive);
}

[[nodiscard]] Scalar recall_from_counts(const ClassCounts& counts) noexcept {
    return ratio_or_zero(counts.true_positive, counts.true_positive + counts.false_negative);
}

[[nodiscard]] Scalar f1_from_precision_recall(Scalar precision, Scalar recall) noexcept {
    return precision + recall == Scalar{0} ? Scalar{0} : Scalar{2} * precision * recall / (precision + recall);
}

[[nodiscard]] ClassificationSummary summarize_classification(const ConfusionMatrix& matrix,
                                                             const EvaluationOptions& options) {
    const std::size_t class_count = matrix.counts.size();
    require(class_count > 0, "confusion matrix must not be empty");
    require(options.positive_class < class_count, "positive_class is out of range");

    std::size_t correct = 0;
    std::size_t total = 0;
    for (std::size_t actual = 0; actual < class_count; ++actual) {
        for (std::size_t predicted = 0; predicted < class_count; ++predicted) {
            const std::size_t count = matrix.counts[actual][predicted];
            total += count;
            if (actual == predicted) {
                correct += count;
            }
        }
    }

    ClassificationSummary summary;
    summary.accuracy = ratio_or_zero(correct, total);

    if (options.averaging == Averaging::binary) {
        const ClassCounts counts = counts_for_class(matrix, options.positive_class);
        summary.precision = precision_from_counts(counts);
        summary.recall = recall_from_counts(counts);
        summary.f1_score = f1_from_precision_recall(summary.precision, summary.recall);
        return summary;
    }

    if (options.averaging == Averaging::micro) {
        ClassCounts counts;
        for (std::size_t class_index = 0; class_index < class_count; ++class_index) {
            const ClassCounts current = counts_for_class(matrix, class_index);
            counts.true_positive += current.true_positive;
            counts.false_positive += current.false_positive;
            counts.false_negative += current.false_negative;
        }
        summary.precision = precision_from_counts(counts);
        summary.recall = recall_from_counts(counts);
        summary.f1_score = f1_from_precision_recall(summary.precision, summary.recall);
        return summary;
    }

    Scalar precision_total = Scalar{0};
    Scalar recall_total = Scalar{0};
    Scalar f1_total = Scalar{0};
    for (std::size_t class_index = 0; class_index < class_count; ++class_index) {
        const ClassCounts counts = counts_for_class(matrix, class_index);
        const Scalar precision = precision_from_counts(counts);
        const Scalar recall = recall_from_counts(counts);
        precision_total += precision;
        recall_total += recall;
        f1_total += f1_from_precision_recall(precision, recall);
    }

    const Scalar scale = Scalar{1} / static_cast<Scalar>(class_count);
    summary.precision = precision_total * scale;
    summary.recall = recall_total * scale;
    summary.f1_score = f1_total * scale;
    return summary;
}

struct RegressionSummary {
    Scalar mean_squared_error{Scalar{0}};
    Scalar mean_absolute_error{Scalar{0}};
    Scalar root_mean_squared_error{Scalar{0}};
    Scalar r2_score{Scalar{0}};
};

[[nodiscard]] RegressionSummary summarize_regression(const Matrix& predictions, const Matrix& targets) {
    Scalar squared_error = Scalar{0};
    Scalar absolute_error = Scalar{0};
    Scalar target_sum = Scalar{0};
    std::size_t count = 0;

    for (std::size_t row = 0; row < predictions.size(); ++row) {
        for (std::size_t column = 0; column < predictions[row].size(); ++column) {
            const Scalar delta = predictions[row][column] - targets[row][column];
            squared_error += delta * delta;
            absolute_error += static_cast<Scalar>(std::abs(delta));
            target_sum += targets[row][column];
            ++count;
        }
    }

    const Scalar count_scalar = static_cast<Scalar>(count);
    const Scalar mean_target = target_sum / count_scalar;
    Scalar target_variance_sum = Scalar{0};
    for (const Vector& target : targets) {
        for (Scalar value : target) {
            const Scalar centered = value - mean_target;
            target_variance_sum += centered * centered;
        }
    }

    RegressionSummary summary;
    summary.mean_squared_error = squared_error / count_scalar;
    summary.mean_absolute_error = absolute_error / count_scalar;
    summary.root_mean_squared_error = static_cast<Scalar>(std::sqrt(summary.mean_squared_error));
    if (target_variance_sum == Scalar{0}) {
        summary.r2_score = squared_error == Scalar{0} ? Scalar{1} : Scalar{0};
    } else {
        summary.r2_score = Scalar{1} - squared_error / target_variance_sum;
    }
    return summary;
}

[[nodiscard]] RegressionSummary summarize_regression(const DenseMatrix& predictions, const DenseMatrix& targets) {
    Scalar squared_error = Scalar{0};
    Scalar absolute_error = Scalar{0};
    Scalar target_sum = Scalar{0};
    const std::size_t count = predictions.scalar_count();

    for (std::size_t i = 0; i < count; ++i) {
        const Scalar delta = predictions.values()[i] - targets.values()[i];
        squared_error += delta * delta;
        absolute_error += static_cast<Scalar>(std::abs(delta));
        target_sum += targets.values()[i];
    }

    const Scalar count_scalar = static_cast<Scalar>(count);
    const Scalar mean_target = target_sum / count_scalar;
    Scalar target_variance_sum = Scalar{0};
    for (Scalar value : targets.values()) {
        const Scalar centered = value - mean_target;
        target_variance_sum += centered * centered;
    }

    RegressionSummary summary;
    summary.mean_squared_error = squared_error / count_scalar;
    summary.mean_absolute_error = absolute_error / count_scalar;
    summary.root_mean_squared_error = static_cast<Scalar>(std::sqrt(summary.mean_squared_error));
    if (target_variance_sum == Scalar{0}) {
        summary.r2_score = squared_error == Scalar{0} ? Scalar{1} : Scalar{0};
    } else {
        summary.r2_score = Scalar{1} - squared_error / target_variance_sum;
    }
    return summary;
}

} // namespace

} // namespace brutal_mlp::detail

namespace brutal_mlp {

using namespace detail;

std::size_t ConfusionMatrix::class_count() const noexcept {
    return counts.size();
}

std::size_t ConfusionMatrix::total() const noexcept {
    std::size_t result = 0;
    for (const auto& row : counts) {
        result += std::accumulate(row.begin(), row.end(), std::size_t{0});
    }
    return result;
}

EvaluationOptions EvaluationOptions::regression() {
    EvaluationOptions options;
    options.metrics = {
        Metric::mean_squared_error,
        Metric::mean_absolute_error,
        Metric::root_mean_squared_error,
        Metric::r2_score,
    };
    return options;
}

EvaluationOptions EvaluationOptions::binary_classification(Scalar threshold, std::size_t positive_class_value) {
    EvaluationOptions options;
    options.metrics = {
        Metric::accuracy,
        Metric::precision,
        Metric::recall,
        Metric::f1_score,
        Metric::confusion_matrix,
    };
    options.classification_threshold = threshold;
    options.positive_class = positive_class_value;
    options.averaging = Averaging::binary;
    validate_evaluation_options(options);
    return options;
}

EvaluationOptions EvaluationOptions::multiclass_classification(Averaging averaging_value) {
    EvaluationOptions options;
    options.metrics = {
        Metric::accuracy,
        Metric::precision,
        Metric::recall,
        Metric::f1_score,
        Metric::confusion_matrix,
    };
    options.averaging = averaging_value;
    validate_evaluation_options(options);
    return options;
}

EvaluationOptions EvaluationOptions::all() {
    EvaluationOptions options;
    options.metrics = {
        Metric::mean_squared_error,
        Metric::mean_absolute_error,
        Metric::root_mean_squared_error,
        Metric::r2_score,
        Metric::accuracy,
        Metric::precision,
        Metric::recall,
        Metric::f1_score,
        Metric::confusion_matrix,
    };
    return options;
}

EvaluationOptions& EvaluationOptions::include(Metric metric) {
    require(metric != Metric::custom, "custom metrics must be registered with add_custom_metric");
    require(!contains_metric(metrics, metric), "duplicate metric requested");
    metrics.push_back(metric);
    return *this;
}

EvaluationOptions& EvaluationOptions::add_custom_metric(std::string metric_name,
                                                        CustomMetricFunction evaluate,
                                                        void* context) {
    require(!metric_name.empty(), "custom metric name must not be empty");
    require(evaluate != nullptr, "custom metric callback is required");
    for (const CustomMetric& metric : custom_metrics) {
        require(metric.name != metric_name, "duplicate custom metric name");
    }
    custom_metrics.push_back(CustomMetric{std::move(metric_name), evaluate, context});
    return *this;
}

bool EvaluationResult::has_metric(Metric metric_value) const noexcept {
    if (metric_value == Metric::confusion_matrix) {
        return has_confusion_matrix;
    }
    return std::any_of(values.begin(), values.end(), [metric_value](const MetricValue& value) {
        return value.metric == metric_value;
    });
}

bool EvaluationResult::has_custom_metric(std::string_view metric_name) const noexcept {
    return std::any_of(values.begin(), values.end(), [metric_name](const MetricValue& value) {
        return value.metric == Metric::custom && value.name == metric_name;
    });
}

Scalar EvaluationResult::metric(Metric metric_value) const {
    require(metric_value != Metric::custom, "use custom_metric to read custom metric values");
    require(metric_value != Metric::confusion_matrix, "confusion_matrix is not a scalar metric");
    for (const MetricValue& value : values) {
        if (value.metric == metric_value) {
            return value.value;
        }
    }
    throw std::out_of_range("metric is not present: " + to_string(metric_value));
}

Scalar EvaluationResult::custom_metric(std::string_view metric_name) const {
    for (const MetricValue& value : values) {
        if (value.metric == Metric::custom && value.name == metric_name) {
            return value.value;
        }
    }
    throw std::out_of_range("custom metric is not present: " + std::string(metric_name));
}

EvaluationResult evaluate_predictions(const Matrix& predictions,
                                      const Matrix& targets,
                                      const EvaluationOptions& options) {
    validate_evaluation_data(predictions, targets);
    validate_evaluation_options(options);

    const std::vector<Metric> metrics = selected_metrics(options);
    const bool needs_regression = std::any_of(metrics.begin(), metrics.end(), is_regression_metric);
    const bool needs_classification = std::any_of(metrics.begin(), metrics.end(), is_classification_metric);

    EvaluationResult result;
    RegressionSummary regression;
    if (needs_regression) {
        regression = summarize_regression(predictions, targets);
    }

    ClassificationSummary classification;
    if (needs_classification) {
        result.confusion_matrix = make_confusion_matrix(predictions, targets, options.classification_threshold);
        result.has_confusion_matrix = true;
        classification = summarize_classification(result.confusion_matrix, options);
    }

    for (Metric metric : metrics) {
        switch (metric) {
        case Metric::mean_squared_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.mean_squared_error});
            break;
        case Metric::mean_absolute_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.mean_absolute_error});
            break;
        case Metric::root_mean_squared_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.root_mean_squared_error});
            break;
        case Metric::r2_score:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.r2_score});
            break;
        case Metric::accuracy:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.accuracy});
            break;
        case Metric::precision:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.precision});
            break;
        case Metric::recall:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.recall});
            break;
        case Metric::f1_score:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.f1_score});
            break;
        case Metric::confusion_matrix:
            break;
        case Metric::custom:
            throw std::invalid_argument("custom metrics must be registered with add_custom_metric");
        }
    }

    for (const CustomMetric& metric : options.custom_metrics) {
        const Scalar value = metric.evaluate(predictions, targets, metric.context);
        require_finite(value, "custom metric");
        result.values.push_back(MetricValue{Metric::custom, metric.name, value});
    }

    return result;
}

EvaluationResult evaluate_predictions(std::initializer_list<Vector> predictions,
                                      std::initializer_list<Vector> targets,
                                      const EvaluationOptions& options) {
    return evaluate_predictions(Matrix(predictions), Matrix(targets), options);
}

EvaluationResult evaluate_predictions(const DenseMatrix& predictions,
                                      const DenseMatrix& targets,
                                      const EvaluationOptions& options) {
    validate_evaluation_data(predictions, targets);
    validate_evaluation_options(options);

    const std::vector<Metric> metrics = selected_metrics(options);
    const bool needs_regression = std::any_of(metrics.begin(), metrics.end(), is_regression_metric);
    const bool needs_classification = std::any_of(metrics.begin(), metrics.end(), is_classification_metric);

    EvaluationResult result;
    RegressionSummary regression;
    if (needs_regression) {
        regression = summarize_regression(predictions, targets);
    }

    ClassificationSummary classification;
    if (needs_classification) {
        result.confusion_matrix = make_confusion_matrix(predictions, targets, options.classification_threshold);
        result.has_confusion_matrix = true;
        classification = summarize_classification(result.confusion_matrix, options);
    }

    for (Metric metric : metrics) {
        switch (metric) {
        case Metric::mean_squared_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.mean_squared_error});
            break;
        case Metric::mean_absolute_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.mean_absolute_error});
            break;
        case Metric::root_mean_squared_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.root_mean_squared_error});
            break;
        case Metric::r2_score:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.r2_score});
            break;
        case Metric::accuracy:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.accuracy});
            break;
        case Metric::precision:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.precision});
            break;
        case Metric::recall:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.recall});
            break;
        case Metric::f1_score:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.f1_score});
            break;
        case Metric::confusion_matrix:
            break;
        case Metric::custom:
            throw std::invalid_argument("custom metrics must be registered with add_custom_metric");
        }
    }

    if (!options.custom_metrics.empty()) {
        const Matrix prediction_matrix = predictions.to_matrix();
        const Matrix target_matrix = targets.to_matrix();
        for (const CustomMetric& metric : options.custom_metrics) {
            const Scalar value = metric.evaluate(prediction_matrix, target_matrix, metric.context);
            require_finite(value, "custom metric");
            result.values.push_back(MetricValue{Metric::custom, metric.name, value});
        }
    }

    return result;
}

} // namespace brutal_mlp
