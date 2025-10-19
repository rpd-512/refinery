#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <fstream>
#include "types.h"
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>


struct chromoInfo;
struct position3D;
struct RobotInfo;
struct dh_param;

void print_vector(vector<double> vec) {
    for (double v : vec) {
        cout << v << ",\t";
    }
    cout << endl;
}

void print_datapoint(const Datapoint& dp) {
    cout << "Datapoint ID: " << dp.id << endl;
    cout << "Features    : ";
    print_vector(dp.features);
    cout << "Groundtruth : ";
    print_vector(dp.groundtruth);
}

void print2DVector(const vector<vector<double>>& vec) {
    for (const auto& row : vec) {
        for (const auto& element : row) {
            cout << element << "\t";
        }
        cout << endl;
    }
}

//-----robot-testing-inverse-kinematics-----//
using namespace Eigen;
using namespace std;
using namespace YAML;

typedef struct dh_param{
    double a;
    double d;
    double alpha;
}dh_param;

typedef struct position3D{
    double x;
    double y;
    double z;
}position3D;

#endif
