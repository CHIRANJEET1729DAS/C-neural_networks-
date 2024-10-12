#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>

using namespace std;

class Matrix {
private:
    static std::mt19937 gen; 
    int numRows;
    int numCols;
    vector<vector<double>> value;

public:
    Matrix(int numRows, int numCols, bool isRandom = false) {
        this->numRows = numRows;
        this->numCols = numCols;
        value.resize(numRows, vector<double>(numCols, 0.0)); // Initialize with zeros

        if (isRandom) {
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    value[i][j] = getRandomNumber();
                }
            }
        }
    }

    ~Matrix() {}

    static double getRandomNumber() {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        return dis(gen);
    }

    void print() const { 
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                cout << this->value[i][j] << "\t";
            }
            cout << endl;
        }
    }

    Matrix transpose() const { 
        Matrix transposed(numCols, numRows);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                transposed.value[j][i] = this->value[i][j];
            }
        }
        return transposed;
    }

    int getRows() const {
        return numRows;
    }

    int getCols() const {
        return numCols;
    }

    double getValue(int row, int col) const {
        if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return value[row][col];
    }

    void setValue(int row, int col, double newValue) {
        if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        value[row][col] = newValue;
    }
};

std::mt19937 Matrix::gen(std::random_device{}());

#endif
