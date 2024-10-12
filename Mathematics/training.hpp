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
    Neural_Networks& neuralNetwork; // Reference to the neural network being trained
    vector<vector<double>> trainingInputs; // Input data for training
    vector<vector<double>> trainingTargets; // Target outputs for training
    double learningRate; // Learning rate for training
    int epochs; // Number of training epochs
    vector<double> epochErrors; // Store the average error for each epoch

public:
    // Constructor to initialize the training parameters
    Training(Neural_Networks& nn, const vector<vector<double>>& inputs, const vector<vector<double>>& targets, double lr, int ep)
        : neuralNetwork(nn), trainingInputs(inputs), trainingTargets(targets), learningRate(lr), epochs(ep) {}

    // Calculate mean squared error between target and output
    double calculateError(const vector<double>& target, const vector<double>& output) {
        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            error += pow(target[i] - output[i], 2);
        }
        return error / target.size(); // Return the average error
    }

    // Train the neural network for a specified number of epochs
    void train() {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalError = 0.0;
            Backpropagation bp(learningRate); // Create Backpropagation instance

            for (size_t i = 0; i < trainingInputs.size(); i++) {
                // Feed forward the input to get the output
                auto output = neuralNetwork.feedForward(trainingInputs[i]);
                // Calculate the error for the current input
                double error = calculateError(trainingTargets[i], output);
                totalError += error; // Accumulate total error

                // Backpropagate the error through the network
                bp.backpropagate(neuralNetwork, trainingTargets[i]); // Corrected to pass two arguments

                // Uncomment if you have clipGradients method
                // bp.clipGradients(1.0); // Call to clip gradients if implemented
            }

            // Calculate average error for the epoch
            totalError /= trainingInputs.size();
            epochErrors.push_back(totalError); // Store the average error for this epoch
            
            // Log the average error for this epoch
            cout << "Epoch " << epoch + 1 << " Error: " << totalError << endl;

            // Optionally adjust learning rate here if needed
            // adjustLearningRate(epoch);
        }

        // Write the epoch errors to a file for plotting
        writeErrorDataToFile("/home/chiranjeet/c++_maths/errortext/error.txt");
        
        // Print final outputs after training
        displayFinalOutputs();
    }

    // Function to display final outputs after training
    void displayFinalOutputs() {
        cout << "Final output matrix after training:" << endl;
        for (const auto& input : trainingInputs) {
            auto finalOutput = neuralNetwork.feedForward(input); // Get final output for each input
            for (const auto& val : finalOutput) {
                cout << val << " "; // Print each output value
            }
            cout << endl; // Newline for clarity
        }
    }

    // Function to write error data to a file
    void writeErrorDataToFile(const string& filename) {
        ofstream outFile(filename);
        if (!outFile) {
            cerr << "Error opening file for writing: " << filename << endl; // Error handling for file opening
            return;
        }
        for (const auto& err : epochErrors) {
            outFile << err << endl; // Write each error to file
        }
        outFile.close(); // Close the file stream
    }
};

#endif
