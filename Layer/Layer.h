
#pragma once

#include "../Matrix/Matrix.h"
#include <functional>

class Layer {
public:
  // member variables
  Matrix weights;
  Matrix biases;
  Matrix lastInput; // store the input from last forward pass
  Matrix lastZ;

  // constructor
  Layer(unsigned int inputSize, unsigned int outputSize,
        std::function<Matrix(const Matrix &)> activation,
        std::function<Matrix(const Matrix &)> activationDerivative);
  Layer(unsigned int inputSize, unsigned int outputSize,
        std::function<Matrix(const Matrix &)> activation);

  // activation function and its derivative
  std::function<Matrix(const Matrix &)> activation;
  std::function<Matrix(const Matrix &)> activationDerivative;

  // forward pass operation.
  Matrix forward(const Matrix &input);

  // backward operation
  Matrix backward(const Matrix &outputGradient, double learningRate);
};
