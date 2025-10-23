
/**
 * @file Matrix.cpp
 * @brief Implements the Matrix class, the core mathematical engine for the
 * network.
 *
 * This class handles all matrix operations, from creation and arithmetic
 * to specialized neural network functions like softmax and He initialization.
 */

#include "Matrix.h"
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits> // For std::numeric_limits
#include <random>
#include <stdexcept>

// --- Constructors ---

/**
 * @brief Constructs a new Matrix, initializing all elements to 0.0.
 * @param numRows The number of rows for the new matrix.
 * @param numCols The number of columns for the new matrix.
 */
Matrix::Matrix(unsigned int numRows, unsigned int numCols)
    : rows(numRows), cols(numCols) {
  // Resize the outer vector to hold the correct number of rows.
  data.resize(rows);
  // For each row, resize the inner vector to hold the correct number of
  // columns.
  for (unsigned int i = 0; i < rows; ++i) {
    data[i].resize(cols, 0.0);
  }
}

/**
 * @brief Constructs a new Matrix from an existing 2D vector of data.
 * @param initial_data A 2D vector to copy data from.
 */
Matrix::Matrix(const std::vector<std::vector<double>> &initial_data) {
  if (initial_data.empty() || initial_data[0].empty()) {
    this->rows = 0;
    this->cols = 0;
  } else {
    this->rows = initial_data.size();
    this->cols = initial_data[0].size();
    this->data = initial_data;
  }
}

// --- Public Member Functions (Operations) ---

/**
 * @brief Performs element-wise (Hadamard) multiplication.
 * @param other The matrix to multiply with.
 * @return A new Matrix where C(i,j) = A(i,j) * B(i,j).
 */
Matrix Matrix::multiply(const Matrix &other) const {
  if (this->rows != other.rows || this->cols != other.cols) {
    throw std::invalid_argument(
        "Matrix dimensions must match for element-wise multiplication.");
  }
  Matrix result(this->rows, this->cols);
  for (unsigned int i = 0; i < this->rows; ++i) {
    for (unsigned int j = 0; j < this->cols; ++j) {
      result.data[i][j] = this->data[i][j] * other.data[i][j];
    }
  }
  return result;
}

/**
 * @brief Performs scalar multiplication.
 * @param scalar The scalar value to multiply with.
 * @return A new Matrix where C(i,j) = A(i,j) * scalar.
 */
Matrix Matrix::multiply(double scalar) const {
  Matrix result(this->rows, this->cols);
  for (unsigned int i = 0; i < this->rows; ++i) {
    for (unsigned int j = 0; j < this->cols; ++j) {
      result.data[i][j] = this->data[i][j] * scalar;
    }
  }
  return result;
}

/**
 * @brief Performs element-wise addition.
 * @param other The matrix to add.
 * @return A new Matrix where C(i,j) = A(i,j) + B(i,j).
 */
Matrix Matrix::add(const Matrix &other) const {
  if (this->rows != other.rows || this->cols != other.cols) {
    throw std::invalid_argument("Matrix dimensions must match for addition.");
  }
  Matrix c(this->rows, this->cols);
  for (unsigned int i = 0; i < this->rows; ++i) {
    for (unsigned int j = 0; j < this->cols; ++j) {
      c.data[i][j] = this->data[i][j] + other.data[i][j];
    }
  }
  return c;
}

/**
 * @brief Performs element-wise subtraction.
 * @param other The matrix to subtract.
 * @return A new Matrix where C(i,j) = A(i,j) - B(i,j).
 */
Matrix Matrix::subtract(const Matrix &other) const {
  if (this->rows != other.rows || this->cols != other.cols) {
    throw std::invalid_argument(
        "Matrix dimensions must match for subtraction.");
  }
  Matrix result(this->rows, this->cols);
  for (unsigned int i = 0; i < this->rows; ++i) {
    for (unsigned int j = 0; j < this->cols; ++j) {
      result.data[i][j] = this->data[i][j] - other.data[i][j];
    }
  }
  return result;
}

/**
 * @brief Performs scalar division.
 * @param scalar The scalar value to divide by.
 * @return A new Matrix where C(i,j) = A(i,j) / scalar.
 */
Matrix Matrix::divide(double scalar) const {
  if (scalar == 0) {
    throw std::invalid_argument("Division by zero.");
  }
  Matrix result(rows, cols);
  for (unsigned int i = 0; i < rows; ++i) {
    for (unsigned int j = 0; j < cols; ++j) {
      result.data[i][j] = data[i][j] / scalar;
    }
  }
  return result;
}

/**
 * @brief Returns the transpose of the matrix.
 * @return A new Matrix where rows and columns are flipped.
 */
Matrix Matrix::transpose() const {
  Matrix result(this->cols, this->rows);
  for (unsigned int i = 0; i < this->rows; ++i) {
    for (unsigned int j = 0; j < this->cols; ++j) {
      result.data[j][i] = this->data[i][j];
    }
  }
  return result;
}

/**
 * @brief Applies a given function to every element of the matrix.
 * @param func A lambda or function that takes a double and returns a double.
 * @return A new Matrix with the function applied to each element.
 */
Matrix Matrix::apply(std::function<double(double)> func) const {
  Matrix result(this->rows, this->cols);
  for (unsigned int i = 0; i < this->rows; ++i) {
    for (unsigned int j = 0; j < this->cols; ++j) {
      result.data[i][j] = func(this->data[i][j]);
    }
  }
  return result;
}

/**
 * @brief Fills the matrix with random values using He Initialization.
 *
 * This is the modern standard for initializing weights in ReLU-based networks.
 * It scales the variance based on the number of inputs to the layer,
 * preventing vanishing or exploding gradients during the initial passes.
 */
void Matrix::randomize() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0.0, 1.0); // Mean 0, Stddev 1

  for (unsigned int i = 0; i < rows; ++i) {
    for (unsigned int j = 0; j < cols; ++j) {
      // He initialization: scale by sqrt(2.0 / num_inputs)
      // For a weights matrix, num_inputs is the number of columns.
      data[i][j] = dis(gen) * std::sqrt(2.0 / cols);
    }
  }
}

/**
 * @brief Calculates the Euclidean norm (magnitude) of the matrix.
 * Used as a diagnostic tool for gradient clipping.
 * @return The scalar magnitude of the matrix.
 */
double Matrix::norm() const {
  double sum_sq = 0.0;
  for (const auto &row : data) {
    for (double val : row) {
      sum_sq += val * val;
    }
  }
  return std::sqrt(sum_sq);
}

/**
 * @brief Prints the matrix contents to the console.
 */
void Matrix::print() const {
  for (unsigned int i = 0; i < rows; ++i) {
    for (unsigned int j = 0; j < cols; ++j) {
      std::cout << std::setw(8) << data[i][j] << " ";
    }
    std::cout << "\n";
  }
}

// --- Public Static Functions ---

/**
 * @brief Performs dot-product multiplication on two matrices.
 * @param a The left-hand matrix (m x n).
 * @param b The right-hand matrix (n x p).
 * @return A new Matrix (m x p) resulting from the dot product.
 */
Matrix Matrix::multiply(const Matrix &a, const Matrix &b) {
  if (a.cols != b.rows) {
    throw std::invalid_argument(
        "Matrix dimensions are not compatible for multiplication.");
  }

  Matrix c(a.rows, b.cols);
  for (unsigned int i = 0; i < a.rows; ++i) {
    for (unsigned int j = 0; j < b.cols; ++j) {
      double result = 0.0;
      // Dot product of row from 'a' and col from 'b'
      for (unsigned int k = 0; k < a.cols; ++k) {
        result += a.data[i][k] * b.data[k][j];
      }
      c.data[i][j] = result;
    }
  }
  return c;
}

/**
 * @brief Calculates the Softmax function for an output vector.
 *
 * This function is numerically stable, preventing overflow by
 * subtracting the max value from the logits before exponentiating.
 *
 * @param m A column vector (N x 1 matrix) of raw logits.
 * @return A new (N x 1) matrix of probabilities that sum to 1.
 */
Matrix Matrix::softmax(const Matrix &m) {
  // 1. Find the max value for numerical stability
  double max_val = std::numeric_limits<double>::lowest();
  for (unsigned int i = 0; i < m.rows; ++i) {
    max_val = std::max(max_val, m.data[i][0]);
  }

  // 2. Exponentiate and sum
  Matrix result(m.rows, m.cols);
  double sum = 0.0;
  for (unsigned int i = 0; i < m.rows; ++i) {
    // Apply stability trick: subtract max value
    result.data[i][0] = std::exp(m.data[i][0] - max_val);
    sum += result.data[i][0];
  }

  // 3. Normalize to get probabilities
  for (unsigned int i = 0; i < m.rows; ++i) {
    result.data[i][0] /= sum;
  }

  return result;
}

// --- Getters ---

std::vector<std::vector<double>> Matrix::getData() const { return this->data; }

unsigned int Matrix::getRows() const { return this->rows; }
unsigned int Matrix::getCols() const { return this->cols; }
