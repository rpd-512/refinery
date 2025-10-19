#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <vector>

using std::cout;
using std::vector;
using std::endl;

using Feature = std::vector<double>;
using Groundtruth = std::vector<double>;

class Datapoint {
public:
    int id;
    vector<double> features;     // input features (position)
    vector<double> groundtruth;  // value for that feature to exist (angle)

    // Constructors
    Datapoint() : id(0), features({}), groundtruth({}) {}
    Datapoint(int id_, const vector<double>& features_, const vector<double>& groundtruth_ = {})
        : id(id_), features(features_), groundtruth(groundtruth_) {}

    // Static factory method
    static Datapoint from_vector(const vector<double>& ftr, const vector<double>& gth={}, int id = 0) {
        return Datapoint{id, ftr, gth};
    }

    static vector<double> to_vector(const Datapoint& dp) {
        vector<double> vec = dp.features;
        vec.insert(vec.end(), dp.groundtruth.begin(), dp.groundtruth.end());
        return vec;
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