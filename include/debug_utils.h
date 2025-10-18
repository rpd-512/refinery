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


// Read CSV into Datapoint vector
std::vector<Datapoint> read_csv(const std::string& filename, int dof = 5) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    std::vector<Datapoint> dataset;
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        Datapoint dp;

        // initial angles (5)
        for (int i = 0; i < dof; ++i) {
            std::getline(ss, cell, ',');
        }

        // target positions (3)
        for (int i = 0; i < 3; ++i) {
            std::getline(ss, cell, ',');
            dp.features.push_back(std::stod(cell));
        }

        // final angles (5)
        for (int i = 0; i < dof; ++i) {
            std::getline(ss, cell, ',');
            dp.groundtruth.push_back(std::stod(cell));
        }

        // algorithm (string)
        std::getline(ss, cell, ',');

        dataset.push_back(dp);
    }

    return dataset;
}

void dh_transform(const dh_param& param, double theta, Matrix4d& A) {
    double alpha = param.alpha * M_PI / 180.0;
    double a     = param.a;
    double d     = param.d;
    double ct    = cos(theta), st = sin(theta);
    double ca    = cos(alpha), sa = sin(alpha);

    A << ct, -st * ca,  st * sa, a * ct,
         st,  ct * ca, -ct * sa, a * st,
         0,       sa,      ca,     d,
         0,        0,       0,     1;
}


vector<dh_param> loadDHFromYAML(const string& filename) {
    vector<dh_param> robot;
    try {
        YAML::Node config = YAML::LoadFile(filename);
        const auto& params = config["dh_parameters"];

        if (!params || !params.IsSequence()) {
            cerr << "Invalid or missing 'dh_parameters' in YAML file." << endl;
            return {};
        }

        robot.clear();
        for (const auto& node : params) {
            dh_param param;
            param.a = node["a"].as<double>();
            param.d = node["d"].as<double>();
            param.alpha = node["alpha"].as<double>();
            robot.push_back(param);
        }

        return robot;
    } catch (const YAML::Exception& e) {
        cerr << "YAML error: " << e.what() << endl;
        return {};
    } catch (const exception& e) {
        cerr << "Error loading DH parameters: " << e.what() << endl;
        return {};
    }
}

Groundtruth forward_kinematics(const vector<double>& theta, const vector<dh_param>& dh) {
    int dof = 5;
    Matrix4d T = Matrix4d::Identity();
    Matrix4d A;
    Vector4d origin(0, 0, 0, 1);
    Vector4d pos;
    vector<position3D> joint_positions;
    joint_positions.reserve(dof + 1);

    pos.noalias() = T * origin;
    joint_positions.push_back({static_cast<double>(pos(0)), static_cast<double>(pos(1)), static_cast<double>(pos(2))});
    for (int i = 0; i < dof; ++i) {
        dh_transform(dh[i], theta[i], A);
        T = T * A;
        pos.noalias() = T * origin;
        joint_positions.push_back({static_cast<double>(pos(0)), static_cast<double>(pos(1)), static_cast<double>(pos(2))});
    }
    position3D ee = joint_positions.back(); // end-effector position
    return {ee.x, ee.y, ee.z}; // convert to Groundtruth (vector<double>)
}

#endif
