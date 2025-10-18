#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <vector>

using std::cout;
using std::vector;
using std::endl;

using Feature = std::vector<double>;
using Groundtruth = std::vector<double>;

struct Datapoint {
    int id;
    Feature features;    // input features (position)
    Groundtruth groundtruth; // value for that feature to exist (angle)Groundtruth
    // Static factory method
    static Datapoint from_vector(const vector<double>& vec, int id = 0) {
        return Datapoint{id, vec, {}};
    }
};

struct RefinementSample{
    Datapoint data;
    Groundtruth target; // Target values
};
struct KDNode{
    Datapoint data;
    KDNode* right;
    KDNode* left;
};

using State = std::pair<Datapoint, double>;

#endif