#ifndef TRAINING_HPP
#define TRAINING_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "/home/chiranjeet/c++_maths/cmathematics/neural_network.hpp"
#include "/home/chiranjeet/c++_maths/cmathematics/back_propagation.hpp"

using namespace std;

class Training {
private:
    Neural_Networks& neuralNetwork; 
    vector<vector<double>> trainingInputs; 
    vector<vector<double>> trainingTargets; 
    double learningRate; 
    int epochs; 
    vector<double> epochErrors; 

public:
    Training(Neural_Networks& nn, const vector<vector<double>>& inputs, const vector<vector<double>>& targets, double lr, int ep)
        : neuralNetwork(nn), trainingInputs(inputs), trainingTargets(targets), learningRate(lr), epochs(ep) {}

    double calculateError(const vector<double>& target, const vector<double>& output) {
        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            error += pow(target[i] - output[i], 2);
        }
        return error / target.size();
    }

    void train() {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalError = 0.0;
            Backpropagation bp(learningRate);

            for (size_t i = 0; i < trainingInputs.size(); i++) {
                auto output = neuralNetwork.feedForward(trainingInputs[i]);
                double error = calculateError(trainingTargets[i], output);
                totalError += error;
                bp.backpropagate(neuralNetwork, trainingTargets[i]);
            }

            totalError /= trainingInputs.size();
            epochErrors.push_back(totalError);
            cout << "Epoch " << epoch + 1 << " Error: " << totalError << endl;
        }

        writeErrorDataToFile("/home/chiranjeet/c++_maths/errortext/error.txt");
        displayFinalOutputs();
    }

    void displayFinalOutputs() {
        cout << "Final output matrix after training:" << endl;
        for (const auto& input : trainingInputs) {
            auto finalOutput = neuralNetwork.feedForward(input);
            for (const auto& val : finalOutput) {
                cout << val << " ";
            }
            cout << endl;
        }
    }

    void writeErrorDataToFile(const string& filename) {
        ofstream outFile(filename);
        if (!outFile) {
            cerr << "Error opening file for writing: " << filename << endl;
            return;
        }
        for (const auto& err : epochErrors) {
            outFile << err << endl;
        }
        outFile.close();
    }
};

#endif
