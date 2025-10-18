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
        for (size_t i = 0; i < new_state.first.groundtruth.size(); ++i) {
            new_state.first.groundtruth[i] -= learning_rate * grad[i];
        }
        new_state.second = compute_loss(new_state.first.groundtruth);
        return new_state;
    }
private:
    double learning_rate;
};

#endif // REFINER_CONTENT_H