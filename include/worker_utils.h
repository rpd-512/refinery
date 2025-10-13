#ifndef WORKER_UTILS_H
#define WORKER_UTILS_H

#include <random>
#include <vector>
#include <cmath>

inline thread_local std::mt19937 gen(std::random_device{}());

int randint(int low, int high) {
    std::uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

double uniform(double low, double high) {
    std::uniform_real_distribution<> dist(low, high);
    return dist(gen);
}

double distance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors must be of same length");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
    //return std::sqrt(sum);
}

#endif