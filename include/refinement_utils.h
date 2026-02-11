#ifndef REFINEMENT_UTILS_H
#define REFINEMENT_UTILS_H

#include "types.h"
#include <cmath>
#include <stdexcept>
#include <functional>
#include <numeric>

class LossFunction {
public:
    // Mean Squared Error
    static double mse_loss(const Groundtruth& y, const Groundtruth& y_hat) {
        validate_inputs(y, y_hat);
        double sum = 0.0;
        for (size_t i = 0; i < y.size(); ++i)
            sum += std::pow(y[i] - y_hat[i], 2);
        return sum / y.size();
    }

    static double euclidean_loss(const Groundtruth& y, const Groundtruth& y_hat) {
        return std::sqrt(mse_loss(y, y_hat));
    }

    // Mean Absolute Error
    static double mae_loss(const Groundtruth& y, const Groundtruth& y_hat) {
        validate_inputs(y, y_hat);
        double sum = 0.0;
        for (size_t i = 0; i < y.size(); ++i)
            sum += std::fabs(y[i] - y_hat[i]);
        return sum / y.size();
    }

    // Huber Loss
    static double huber_loss(const Groundtruth& y, const Groundtruth& y_hat) {
        double delta = 2.0;
        validate_inputs(y, y_hat);
        double sum = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            double diff = y[i] - y_hat[i];
            if (std::fabs(diff) <= delta)
                sum += 0.5 * diff * diff;
            else
                sum += delta * (std::fabs(diff) - 0.5 * delta);
        }
        return sum / y.size();
    }

    // Cosine Error (1 - cosine similarity)
    static double cosine_error(const Groundtruth& y, const Groundtruth& y_hat) {
        validate_inputs(y, y_hat);
        double dot = 0.0, norm_y = 0.0, norm_yhat = 0.0;

        for (size_t i = 0; i < y.size(); ++i) {
            dot += y[i] * y_hat[i];
            norm_y += y[i] * y[i];
            norm_yhat += y_hat[i] * y_hat[i];
        }

        norm_y = std::sqrt(norm_y);
        norm_yhat = std::sqrt(norm_yhat);

        if (norm_y == 0.0 || norm_yhat == 0.0)
            throw std::runtime_error("Cosine error: zero-magnitude vector.");

        double cosine_similarity = dot / (norm_y * norm_yhat);
        return 1.0 - cosine_similarity;
    }

    // Log-Cosh Error
    static double log_cosh_error(const Groundtruth& y, const Groundtruth& y_hat) {
        validate_inputs(y, y_hat);
        double sum = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            double diff = y_hat[i] - y[i];
            sum += std::log(std::cosh(diff));
        }
        return sum / y.size();
    }

private:
    static void validate_inputs(const Groundtruth& y, const Groundtruth& y_hat) {
        if (y.size() != y_hat.size() || y.empty())
            throw std::invalid_argument("LossFunction: input vectors must be non-empty and of equal size.");
    }
};


class Optimizer {
public:
    Feature target_feature; // target feature for refinement

    Optimizer(
        std::function<Feature(const Groundtruth&)> forward_function,
        std::function<double(const Groundtruth&, const Feature&)> loss_function
    ) : forward_function(forward_function), loss_function(loss_function) {}

    virtual ~Optimizer() = default;
    virtual State optimize(const State& current_state) = 0;
    
protected:
    static constexpr double infinitesimal = 1e-6; // infinitesimal step size
    std::function<Feature(const Groundtruth&)> forward_function; // can capture variables
    std::function<double(const Groundtruth&, const Feature&)> loss_function;
    double compute_loss(const Groundtruth& gt) {
        if(forward_function) {
            Feature output_feature = forward_function(gt);
            double loss = loss_function(output_feature, target_feature);
            return loss;
        }
        throw std::runtime_error("Loss function not defined.");
    }

    Groundtruth gradient_approximation(const Datapoint& dp) {
        Groundtruth derivative(dp.groundtruth.size(), 0.0);
        Datapoint temp = dp;  // copy for perturbation

        for (size_t i = 0; i < dp.groundtruth.size(); ++i) {
            double original = temp.groundtruth[i];

            temp.groundtruth[i] = original + infinitesimal;
            double loss_plus = compute_loss(temp.groundtruth);

            temp.groundtruth[i] = original - infinitesimal;
            double loss_minus = compute_loss(temp.groundtruth);

            derivative[i] = (loss_plus - loss_minus) / (2.0 * infinitesimal);
            // debug print
            temp.groundtruth[i] = original;
        }
        return derivative;
    }
};

class RefinementEngine {
public:
    ~RefinementEngine() {
        clear_logs();
        seed_vector = Datapoint{};
    }
    RefinementEngine(Optimizer* optimizer) : optimizer(optimizer) {}

    void set_seed(const Datapoint& seed_vector) {
        this->seed_vector = seed_vector;
    }

    void set_target(const Feature& target_feature) {
        optimizer->target_feature = target_feature;
    }

    void set_logging(bool log_flag) {
        this->log_flag = log_flag;
    }

    Groundtruth refine(int iterations) {
        clear_logs();
        if(seed_vector.features.empty()) {
            throw std::runtime_error("Seed vector not set.");
        }
        State current_state = {seed_vector, 0.0};
        for(int i=0; i<iterations; ++i){
            current_state = optimizer->optimize(current_state);
            if(log_flag) {
                data_history.push_back(current_state.first);
                loss_history.push_back(current_state.second);
            }
        }
        return current_state.first.groundtruth;
    }
    vector<double> get_loss_history() const {
        return loss_history;
    }
    vector<Groundtruth> get_data_history() const {
        vector<Groundtruth> data_history;
        for(const auto& dp : this->data_history) {
            data_history.push_back(dp.groundtruth);
        }
        return data_history;
    }

private:
    Datapoint seed_vector;
    Optimizer* optimizer;
    vector<Datapoint> data_history;
    vector<double> loss_history;
    bool log_flag = false;

    void clear_logs() {
        data_history.clear();
        loss_history.clear();
    }
};

#endif // REFINEMENT_UTILS_H