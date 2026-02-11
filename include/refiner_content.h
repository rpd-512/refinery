#ifndef REFINER_CONTENT_H
#define REFINER_CONTENT_H

#include "types.h"
#include "refinement_utils.h"

//SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, NAdam

class GradientDescentOptimizer : public Optimizer {
public:
    GradientDescentOptimizer(
        std::function<Feature(const Groundtruth&)> forward_function,
        std::function<double(const Groundtruth&, const Feature&)> loss_function,
        double learning_rate = 0.01
    ): Optimizer(forward_function, loss_function), learning_rate(learning_rate) {}

    State optimize(const State& current_state) override {
        if (!forward_function || !loss_function)
            throw std::runtime_error("GradientDescentOptimizer: missing function pointers.");
        State new_state = current_state;
        Groundtruth grad = gradient_approximation(current_state.first);

        // Gradient Descent Update: x = x - η * ∇L
        for (size_t i = 0; i < grad.size(); ++i) {
            new_state.first.groundtruth[i] -= learning_rate * grad[i];
        }
        new_state.second = compute_loss(new_state.first.groundtruth);
        return new_state;
    }
private:
    double learning_rate;
};

class GradientMomentumOptimizer : public Optimizer {
public:
    GradientMomentumOptimizer(
        std::function<Feature(const Groundtruth&)> forward_function,
        std::function<double(const Groundtruth&, const Feature&)> loss_function,
        double learning_rate = 0.01,
        double momentum = 0.9
    ) : Optimizer(forward_function, loss_function),
        learning_rate(learning_rate), momentum(momentum) {}

    State optimize(const State& current_state) override {
        if (!forward_function || !loss_function)
            throw std::runtime_error("GDWithMomentumOptimizer: missing function pointers.");

        State new_state = current_state;
        Groundtruth grad = gradient_approximation(current_state.first);
        // Initialize velocity once, matching parameter size
        const size_t dim = grad.size();

        if (velocity.empty())
            velocity = std::vector<double>(dim, 0.0);

        // Gradient Descent with momentum:
        for (size_t i = 0; i < new_state.first.groundtruth.size(); ++i) {
            velocity[i] = momentum * velocity[i] - learning_rate * grad[i];
            new_state.first.groundtruth[i] += velocity[i];
        }
        new_state.second = compute_loss(new_state.first.groundtruth);
        return new_state;
    }

private:
    double learning_rate;
    double momentum;
    vector<double> velocity;

};

class GradientNesterovOptimizer : public Optimizer {
public:
    GradientNesterovOptimizer(
        std::function<Feature(const Groundtruth&)> forward_function,
        std::function<double(const Groundtruth&, const Feature&)> loss_function,
        double learning_rate = 0.01,
        double momentum = 0.9
    ) : Optimizer(forward_function, loss_function),
        learning_rate(learning_rate), momentum(momentum) {}

    State optimize(const State& current_state) override {
        if (!forward_function || !loss_function)
            throw std::runtime_error("GradientNesterovOptimizer: missing function pointers.");

        State new_state = current_state;
        State lookahead_state = current_state;

        if (velocity.empty()) {
            velocity.resize(current_state.first.groundtruth.size(), 0.0);
        }

        // Lookahead step
        for (size_t i = 0; i < lookahead_state.first.groundtruth.size(); ++i) {
            lookahead_state.first.groundtruth[i] += momentum * velocity[i];
        }
        Groundtruth grad = gradient_approximation(lookahead_state.first);
        size_t dim = grad.size();
        
        // --- Nesterov update ---
        for (size_t i = 0; i < dim; ++i) {
            velocity[i] = momentum * velocity[i] - learning_rate * grad[i];
            new_state.first.groundtruth[i] += velocity[i];
        }

        new_state.second = compute_loss(new_state.first.groundtruth);
        return new_state;
    }
private:
    double learning_rate;
    double momentum;
    vector<double> velocity;
};

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(
        std::function<Feature(const Groundtruth&)> forward_function,
        std::function<double(const Groundtruth&, const Feature&)> loss_function,
        double learning_rate = 0.001,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-5
    ) : Optimizer(forward_function, loss_function),
        learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon),
        t(0) {}

    State optimize(const State& current_state) override {
        if (!forward_function || !loss_function)
            throw std::runtime_error("AdamOptimizer: missing function pointers.");

        State new_state = current_state;
        Groundtruth grad = gradient_approximation(current_state.first);
        size_t n = grad.size();

        if (m.empty()) {
            m.resize(n, 0.0);
            v.resize(n, 0.0);
        }

        t++;

        // Update biased moment estimates
        for (size_t i = 0; i < n; ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * (grad[i] * grad[i]);

            double m_hat = m[i] / (1.0 - std::pow(beta1, t));
            double v_hat = v[i] / (1.0 - std::pow(beta2, t));

            // Parameter update
            new_state.first.groundtruth[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        new_state.second = compute_loss(new_state.first.groundtruth);
        return new_state;
    }

private:
    double learning_rate;
    double beta1, beta2;
    double epsilon;
    int t;
    std::vector<double> m, v; // Moment estimates
};

class AdagradOptimizer : public Optimizer {
public:
    AdagradOptimizer(
        std::function<Feature(const Groundtruth&)> forward_function,
        std::function<double(const Groundtruth&, const Feature&)> loss_function,
        double learning_rate = 0.01,
        double epsilon = infinitesimal
    ) : Optimizer(forward_function, loss_function),
        learning_rate(learning_rate), epsilon(epsilon) {}
    State optimize(const State& current_state) override {
        if (!forward_function || !loss_function)
            throw std::runtime_error("AdagradOptimizer: missing function pointers.");

        State new_state = current_state;
        Groundtruth grad = gradient_approximation(current_state.first);
        const size_t dim = grad.size();

        if (accumulated.empty())
            accumulated.assign(dim, 0.0);

        for (size_t i = 0; i < dim; ++i) {
            accumulated[i] += grad[i] * grad[i];  // accumulate squared gradients
            new_state.first.groundtruth[i] -= learning_rate * grad[i] / (std::sqrt(accumulated[i]) + epsilon);
        }

        new_state.second = compute_loss(new_state.first.groundtruth);
        return new_state;
    }
private:
    double learning_rate;
    double epsilon;
    vector<double> accumulated;
};

class RMSpropOptimizer : public Optimizer {
public:
    RMSpropOptimizer(
        std::function<Feature(const Groundtruth&)> forward_function,
        std::function<double(const Groundtruth&, const Feature&)> loss_function,
        double learning_rate = 0.001,
        double decay_rate = 0.99,
        double epsilon = infinitesimal
    ) : Optimizer(forward_function, loss_function),
        learning_rate(learning_rate), decay_rate(decay_rate), epsilon(epsilon) {}

    State optimize(const State& current_state) override {
        if (!forward_function || !loss_function)
            throw std::runtime_error("RMSpropOptimizer: missing function pointers.");

        State new_state = current_state;
        Groundtruth grad = gradient_approximation(current_state.first);
        const size_t dim = grad.size();

        if (avg_sq_grad.empty())
            avg_sq_grad.assign(dim, 0.0);

        for (size_t i = 0; i < dim; ++i) {
            avg_sq_grad[i] = decay_rate * avg_sq_grad[i] + (1.0 - decay_rate) * grad[i] * grad[i];
            new_state.first.groundtruth[i] -= learning_rate * grad[i] / (std::sqrt(avg_sq_grad[i]) + epsilon);
        }

        new_state.second = compute_loss(new_state.first.groundtruth);
        return new_state;
    }

private:
    double learning_rate;
    double decay_rate;
    double epsilon;
    vector<double> avg_sq_grad;
};

#endif // REFINER_CONTENT_H