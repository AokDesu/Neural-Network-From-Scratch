#pragma once // prevent the file from being included multiple times

#include <functional>
#include <iostream>
#include <vector>

class Matrix {
private:
  unsigned int rows;
  unsigned int cols;
  std::vector<std::vector<double>> data;

public:
  // Constructor
  Matrix(unsigned int numRows, unsigned int numCols);

  Matrix(const std::vector<std::vector<double>> &initial_data);

  static Matrix softmax(const Matrix &m);

  // we define as static because we can call it
  // Matrix C = Matrix::multiply(a,b);
  static Matrix multiply(const Matrix &a, const Matrix &b);

  Matrix multiply(const Matrix &other) const; // Element-wise
  Matrix add(const Matrix &other) const;
  Matrix multiply(double scalar) const;
  Matrix subtract(const Matrix &other) const;
  // transpose matrix function
  Matrix transpose() const;

  // appies a function to every element of matrix.
  // this is modern C++ way of saying "an argument named func that is a function
  // take double and return a double"
  Matrix apply(std::function<double(double)> func) const;

  Matrix divide(double scalar) const;

  // Fills the matrix with random values between -1.0 and 1.0
  void randomize();

  // Utility function to print the matrix
  void print() const;
  std::vector<std::vector<double>> getData() const;
  double norm() const;
  unsigned int getRows() const;
  unsigned int getCols() const;
};
