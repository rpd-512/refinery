#ifndef NEAREST_NEIGHBOR_UTIL_H
#define NEAREST_NEIGHBOR_UTIL_H

#include "types.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include "worker_utils.h"

class NearestNeighbourEngine {
public:
    ~NearestNeighbourEngine() {
        clear(root,0);
        nodes.clear();
    }

    NearestNeighbourEngine(int dimensions, vector<Datapoint> datapoints) {
        k = dimensions;
        for (const auto& dp : datapoints) {
            KDNode* new_node = new KDNode{dp, nullptr, nullptr};
            nodes.push_back(new_node);
        }
        root = build(nodes);
    }

    float get_balance_score() {
        if (root == nullptr) return 1.0f; // Empty tree is perfectly balanced

        int max_h = get_max_height(root);
        int min_h = get_min_height(root);

        if (min_h == 0 && max_h == 0) return 1.0f;

        float diff = abs(min_h - max_h);
        float max_height = std::max(min_h, max_h);

        return 1.0f - (diff / max_height); // 1 = perfect balance

    }

    void insert(const Datapoint& dp) {
        KDNode* new_node = new KDNode{dp, nullptr, nullptr};
        nodes.push_back(new_node);
        insert_into_kd_tree(root, *new_node, 0);
    }

    Datapoint query(const Datapoint& target) {
        Datapoint nearest;
        nn_search(root, target, 0, nearest);
        return nearest;
    }
    
    void rebuild() {
        clear(root,1);
        root = build(nodes, 0);
    }

    Datapoint linear_search(const Datapoint& target) {
        Datapoint nearest;
        double min_dist = std::numeric_limits<double>::max();

        for (const auto& node : nodes) {
            double dist = squared_distance(target.features, node->data.features);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = node->data;
            }
        }
        return nearest;
    }

private:
    KDNode* root = nullptr;
    int k = 0;
    vector<KDNode*> nodes;

    KDNode* build(vector<KDNode*> point_nodes, int depth = 0) {
        if (point_nodes.empty()) return nullptr;

        int axis = depth % k;
        size_t median = point_nodes.size() / 2;

        // Sort points by the current axis
        sort(point_nodes.begin(), point_nodes.end(),
                [axis](const KDNode* a, const KDNode* b) {
                    return a->data.features[axis] < b->data.features[axis];
                });

        KDNode* node = point_nodes[median];
        node->left  = build(vector<KDNode*>(point_nodes.begin(), point_nodes.begin() + median), depth + 1);
        node->right = build(vector<KDNode*>(point_nodes.begin() + median + 1, point_nodes.end()), depth + 1);

        return node; // return the node for recursion

    }

    void nn_search(KDNode* node, const Datapoint& target, int depth, Datapoint& nearest) {
        if (node == nullptr) return;

        int axis = depth % k;
        bool go_left = target.features[axis] < node->data.features[axis];

        KDNode* next_branch = go_left ? node->left : node->right;
        KDNode* opposite_branch = go_left ? node->right : node->left;

        // Explore the next branch
        nn_search(next_branch, target, depth + 1, nearest);

        // Update nearest if this node is closer
        if (nearest.features.empty() || squared_distance(target.features, node->data.features) < squared_distance(target.features, nearest.features)) {
            nearest = node->data;
        }

        // Check if we need to explore the opposite branch
        if (opposite_branch != nullptr) {
            double diff = target.features[axis] - node->data.features[axis];
            if (diff * diff < squared_distance(target.features, nearest.features)) {
                nn_search(opposite_branch, target, depth + 1, nearest);
            }
        }
    }

    void clear(KDNode* node, int delete_flag) {
        if (!node) return;
        clear(node->left, delete_flag);
        clear(node->right, delete_flag);

        if (!delete_flag) { // flag is 0
            delete node;          // actually free memory
        } else { // flag is 1
            node->left = node->right = nullptr;  // just detach children, keep node alive
        }
    }

    void insert_into_kd_tree(KDNode*& node, const KDNode& kdnode, int depth) {
        if (node == nullptr) {
            node = new KDNode(kdnode);
            return;
        }

        int axis = depth % k;
        if (kdnode.data.features[axis] < node->data.features[axis]) {
            insert_into_kd_tree(node->left, kdnode, depth + 1);
        } else {
            insert_into_kd_tree(node->right, kdnode, depth + 1);
        }
    }

    int get_max_height(KDNode* node) {
        if (node == nullptr) return 0;
        return 1 + std::max(get_max_height(node->left), get_max_height(node->right));
    }

    int get_min_height(KDNode* node) {
        if (node == nullptr) return 0;
        return 1 + std::min(get_min_height(node->left), get_min_height(node->right));
    }
};

#endif
