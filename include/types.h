#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <vector>

using std::cout;
using std::vector;
using std::endl;

struct Datapoint{
    int id;
    vector<double> features;
    vector<double> groundtruth;
};

struct KDNode{
    Datapoint data;
    KDNode* right;
    KDNode* left;
};



#endif