#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <stdexcept>
#include "/home/chiranjeet/c++_maths/cmathematics/layer.hpp"
#include "/home/chiranjeet/c++_maths/cmathematics/matrix_iu.hpp"
#include "/home/chiranjeet/c++_maths/cmathematics/neuron.hpp"

using namespace std;

class Neural_Networks {
private:
    vector<int> topology;
    vector<Layer*> layers;
    vector<Matrix*> weight_matrix;

public:
    Neural_Networks(vector<int> topology) : topology(topology) {
        for (int i = 0; i < topology.size(); i++) {
            Layer* l = new Layer(topology.at(i), true);
            this->layers.push_back(l);
        }
        for (int i = 0; i < topology.size() - 1; i++) {
            Matrix* m = new Matrix(topology.at(i), topology.at(i + 1), true);
            this->weight_matrix.push_back(m);
        }
    }

    ~Neural_Networks() {
        for (auto layer : layers) {
            delete layer;
        }
        for (auto matrix : weight_matrix) {
            delete matrix;
        }
    }

    vector<double> feedForward(const vector<double>& input) {
        vector<double> currentValues = input;

        for (size_t i = 0; i < weight_matrix.size(); i++) {
            auto result = process(currentValues, weight_matrix[i]);
            currentValues = std::get<0>(result);
        }

        return currentValues;
    }

    tuple<vector<double>, vector<double>> process(const vector<double>& input, Matrix* weight_matrix) {
        vector<double> activated_layer;
        vector<double> derived_layer;

        for (int j = 0; j < weight_matrix->getCols(); j++) {
            double sum = 0.0;
            for (int i = 0; i < input.size(); i++) {
                sum += input[i] * weight_matrix->getValue(i, j);
            }
            Neuron n(sum);
            n.activate();
            n.derive();
            activated_layer.push_back(n.getActivateVal());
            derived_layer.push_back(n.getDerivedVal());
        }

        return make_tuple(activated_layer, derived_layer);
    }

    vector<int> get_topology() const {
        return topology;
    }

    vector<Layer*> get_layers() const {
        return layers;
    }

    vector<Matrix*> get_weight_matrix() const {
        return weight_matrix;
    }
};

#endif
