#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>

using namespace std;
using namespace std::chrono;

// ---------- Utility function ----------
vector<double> random_gradients(int n) {
    vector<double> grad(n);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-0.1, 0.1);
    for(int i=0; i<n; i++) grad[i] = dis(gen);
    return grad;
}

// ---------- SGD Update ----------
void sgd_update(vector<double>& params, const vector<double>& grad, double lr, double momentum, vector<double>& velocity) {
    for(size_t i=0; i<params.size(); i++) {
        velocity[i] = momentum * velocity[i] - lr * grad[i];
        params[i] += velocity[i];
    }
}

// ---------- Adam Update ----------
void adam_update(vector<double>& params, const vector<double>& grad, vector<double>& m, vector<double>& v,
                 double lr, double beta1, double beta2, double eps, int t) 
{
    for(size_t i=0; i<params.size(); i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];

        double m_hat = m[i] / (1 - pow(beta1, t));
        double v_hat = v[i] / (1 - pow(beta2, t));

        params[i] -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}

int main() {
    int n = 100000;           // number of parameters
    int steps = 1000;         // number of update steps

    vector<double> params_sgd(n, 0.0);
    vector<double> velocity(n, 0.0);

    vector<double> params_adam(n, 0.0);
    vector<double> m(n, 0.0);
    vector<double> v(n, 0.0);

    double lr = 0.01;
    double momentum = 0.9;
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

    // ----- Measure SGD -----
    auto start_sgd = high_resolution_clock::now();
    for(int t=1; t<=steps; t++) {
        vector<double> grad = random_gradients(n);
        sgd_update(params_sgd, grad, lr, momentum, velocity);
    }
    auto end_sgd = high_resolution_clock::now();
    cout << "SGD time: " 
         << duration_cast<milliseconds>(end_sgd - start_sgd).count() << " ms\n";

    // ----- Measure Adam -----
    auto start_adam = high_resolution_clock::now();
    for(int t=1; t<=steps; t++) {
        vector<double> grad = random_gradients(n);
        adam_update(params_adam, grad, m, v, lr, beta1, beta2, eps, t);
    }
    auto end_adam = high_resolution_clock::now();
    cout << "Adam time: " 
         << duration_cast<milliseconds>(end_adam - start_adam).count() << " ms\n";

    return 0;
}
