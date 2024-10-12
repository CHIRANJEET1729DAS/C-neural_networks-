#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>
using namespace std;

class Neuron {
private:
    double val;
    double activatedVal;
    double derivedVal;

public:

    Neuron(double val) : val(val) {
        activate();
        derive();
    }

  
    ~Neuron() {}

    // Activation function
    void activate() {

        this->activatedVal = this->val / (1 + abs(this->val)); 
    }

    // Derivation
    void derive() {
        this->derivedVal = this->activatedVal * (1 - this->activatedVal);
    }


    double getVal() const {
        return this->val;
    }

    double getActivateVal() const {
        return this->activatedVal;
    }

    double getDerivedVal() const {
        return this->derivedVal;
    }
};

#endif
