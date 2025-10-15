#ifndef REFINEMENT_UTILS_H
#define REFINEMENT_UTILS_H

#include "types.h"

//SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, NAdam


class Optimizer {
public:
    Optimizer(double (*loss_fn)(const Datapoint&)) {
        forward_loss = loss_fn;
    }
    virtual ~Optimizer() = default;
    virtual State optimize(const State& current_state) = 0;
    
private:
    double eta = 1e-5; // infinitesimal step size
    double (*forward_loss)(const Datapoint&); // store the function pointer
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

    void clear_logs() {
        data_history.clear();
        loss_history.clear();
    }

    void set_logging(bool log_flag) {
        this->log_flag = log_flag;
    }

    Datapoint refine(int iterations) {
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
        return current_state.first;
    }

private:
    Datapoint seed_vector;
    Optimizer* optimizer;
    vector<Datapoint> data_history;
    vector<double> loss_history;
    bool log_flag = false;
};

#endif // REFINEMENT_UTILS_H