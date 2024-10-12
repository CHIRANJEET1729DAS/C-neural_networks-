#ifndef BACKPROPAGATION_HPP
#define BACKPROPAGATION_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include "/home/chiranjeet/c++_maths/cmathematics/neural_network.hpp"

using namespace std;

class Backpropagation {
private:
    double learningRate;

public:
    Backpropagation(double learningRate = 0.1) {
        this->learningRate = learningRate;
    }

    void backpropagate(Neural_Networks &neuralNetwork, vector<double> targetOutput) {
        vector<vector<double>> deltaValues(neuralNetwork.get_topology().size());
        vector<vector<double>> activatedValues(neuralNetwork.get_topology().size());

        for (int i = 0; i < neuralNetwork.get_layers().size(); ++i) {
            vector<Neuron*> current_neurons = neuralNetwork.get_layers()[i]->Get_layer_vector();
            vector<double> neuronValues;

            for (Neuron* neuron : current_neurons) {
                neuronValues.push_back(neuron->getActivateVal());
            }

            activatedValues[i] = neuronValues;
        }

        for (int i = neuralNetwork.get_layers().size() - 1; i >= 0; --i) {
            if (i == neuralNetwork.get_layers().size() - 1) {
                for (int j = 0; j < neuralNetwork.get_layers()[i]->Get_layer_vector().size(); ++j) {
                    double output = activatedValues[i][j];
                    double error = targetOutput[j] - output;
                    double delta = error * output * (1.0 - output);
                    deltaValues[i].push_back(delta);
                }
            } else {
                for (int j = 0; j < neuralNetwork.get_layers()[i]->Get_layer_vector().size(); ++j) {
                    double sumError = 0.0;
                    for (int k = 0; k < neuralNetwork.get_weight_matrix()[i]->getCols(); ++k) {
                        sumError += deltaValues[i + 1][k] * neuralNetwork.get_weight_matrix()[i]->getValue(j, k);
                    }
                    double delta = sumError * activatedValues[i][j] * (1.0 - activatedValues[i][j]);
                    deltaValues[i].push_back(delta);
                }
            }
        }

        for (int i = 0; i < neuralNetwork.get_weight_matrix().size(); ++i) {
            for (int j = 0; j < neuralNetwork.get_weight_matrix()[i]->getRows(); ++j) {
                for (int k = 0; k < neuralNetwork.get_weight_matrix()[i]->getCols(); ++k) {
                    double updatedWeight = neuralNetwork.get_weight_matrix()[i]->getValue(j, k) +
                                           learningRate * deltaValues[i + 1][k] * activatedValues[i][j];
                    neuralNetwork.get_weight_matrix()[i]->setValue(j, k, updatedWeight);
                }
            }
        }

        calculateError(neuralNetwork, targetOutput, activatedValues.back());
    }

    void calculateError(Neural_Networks &neuralNetwork, vector<double> targetOutput, vector<double> output) {
        double error = 0.0;
        for (size_t i = 0; i < targetOutput.size(); ++i) {
            error += pow(targetOutput[i] - output[i], 2);
        }
        error /= targetOutput.size();
        cout << "Error: " << error << endl;
    }
};

#endif
