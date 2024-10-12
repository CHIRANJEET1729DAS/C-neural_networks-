#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>
#include <random>
#include "/home/chiranjeet/c++_maths/cmathematics/neuron.hpp"


using namespace std;

class Layer {
private:
    int numNeurons;
    vector<Neuron*> neurons;

public:
    Layer(int numNeurons, bool isRandom = false) {
        this->numNeurons = numNeurons; 
        for (int i = 0; i < this->numNeurons; i++) { 
            double neuronValue = isRandom ? getRandomNumber() : 0.0; 
            Neuron* neuron = new Neuron(neuronValue); 
            this->neurons.push_back(neuron); 
        }
    }

    ~Layer() {
        for (Neuron* neuron : neurons) {
            delete neuron; 
        }
    }

    vector<Neuron*> Get_layer_vector() {
        return this->neurons;
    }

private:
    double getRandomNumber() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        return dis(gen);
    }
};

#endif
