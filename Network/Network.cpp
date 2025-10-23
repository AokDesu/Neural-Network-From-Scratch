
/**
 * @file Network.cpp
 * @brief Implements the Network class, which manages the layers and training.
 *
 * This class orchestrates the entire process, including the forward pass
 * (for predictions) and the full training loop (forward, backpropagation,
 * and parameter updates).
 */

#include "Network.h"
#include <cmath>
#include <iostream> // For std::cout in the train function

/**
 * @brief Adds a new, pre-initialized layer to the network.
 * @param layer The Layer object to add to the end of the network.
 */
void Network::addLayer(const Layer &layer) { this->layers.push_back(layer); }

/**
 * @brief Performs a full forward pass through all layers to get a prediction.
 *
 * @param input The input matrix (e.g., a 784x1 image vector).
 * @return The final output matrix from the last layer (e.g., a 10x1 probability
 * vector).
 */
Matrix Network::predict(const Matrix &input) {
  Matrix output = input;

  // Pass the input through each layer sequentially.
  // The output of one layer becomes the input for the next.
  for (Layer &layer : this->layers) {
    output = layer.forward(output);
  }

  return output;
}

/**
 * @brief The main training function that runs the learning process.
 *
 * This function iterates through the data for a number of epochs,
 * performing the full forward and backward pass for each training example.
 *
 * @param training_data A vector of input matrices.
 * @param training_labels A vector of corresponding label matrices.
 * @param epochs The total number of times to iterate through the entire
 * dataset.
 * @param learningRate The step size for gradient descent.
 */
void Network::train(const std::vector<Matrix> &training_data,
                    const std::vector<Matrix> &training_labels, int epochs,
                    double learningRate) {
  // Outer loop: Iterates over the entire dataset multiple times.
  for (int i = 0; i < epochs; ++i) {
    double total_error = 0;

    // Inner loop: Iterates through each example in the dataset.
    for (size_t j = 0; j < training_data.size(); ++j) {
      const Matrix &input = training_data[j];
      const Matrix &label = training_labels[j];

      // --- Step 1: Forward Pass ---
      // Get the network's prediction for the current input.
      Matrix prediction = this->predict(input);

      // --- Step 2: Calculate Error ---
      // Use Cross-Entropy Loss, the standard for Softmax.
      total_error += Network::cross_entropy_loss(prediction, label);

      // --- Step 3: Backward Pass (Backpropagation) ---
      // Get the initial error gradient from the loss function.
      Matrix gradient =
          Network::cross_entropy_loss_derivative(prediction, label);

      // Pass the gradient backward through all layers, from last to first.
      // Each layer updates its own weights and returns the gradient
      // for the previous layer.
      for (int k = this->layers.size() - 1; k >= 0; --k) {
        gradient = this->layers[k].backward(gradient, learningRate);
      }
    }

    // Print the average error for this epoch.
    std::cout << "Epoch " << i + 1 << "/" << epochs
              << ", Error: " << total_error / training_data.size() << std::endl;
  }
}

// --- Loss Functions and Derivatives ---

/**
 * @brief Calculates the Cross-Entropy Loss.
 *
 * This is the standard loss function for multi-class classification
 * when the output layer uses Softmax.
 *
 * @param prediction The (N x 1) probability vector from the Softmax layer.
 * @param label The (N x 1) one-hot encoded correct label.
 * @return The scalar loss value for this example.
 */
double Network::cross_entropy_loss(const Matrix &prediction,
                                   const Matrix &label) {
  double loss = 0.0;
  // A small constant to prevent log(0), which would be -infinity.
  const double epsilon = 1e-9;

  for (unsigned int i = 0; i < prediction.getRows(); ++i) {
    // The formula is -y * log(p)
    // Since y is one-hot encoded, we only care about the case
    // where label.data[i][0] is 1.0.
    loss -=
        label.getData()[i][0] * std::log(prediction.getData()[i][0] + epsilon);
  }
  return loss;
}

/**
 * @brief Calculates the derivative of (Cross-Entropy Loss + Softmax).
 *
 * The derivative of Softmax(Z) with respect to Z and the derivative
 * of Cross-Entropy Loss with respect to Softmax(Z) simplify
 * mathematically to this one elegant formula: (prediction - label).
 *
 * @param prediction The (N x 1) probability vector from Softmax.
 * @param label The (N x 1) one-hot encoded correct label.
 * @return The initial gradient (p - y) to be passed to the last layer.
 */
Matrix Network::cross_entropy_loss_derivative(const Matrix &prediction,
                                              const Matrix &label) {
  // The derivative is simply: prediction - label
  return prediction.subtract(label);
}

/**
 * @brief [DEPRECATED] Calculates Mean Squared Error.
 * Good for regression problems or as a simpler alternative.
 */
double Network::mse(const Matrix &prediction, const Matrix &label) {
  Matrix diff = prediction.subtract(label);
  double sum = 0.0;
  for (unsigned int i = 0; i < diff.getRows(); ++i) {
    for (unsigned int j = 0; j < diff.getCols(); ++j) {
      sum += std::pow(diff.getData()[i][j], 2);
    }
  }
  return sum / (diff.getRows() * diff.getCols());
}

/**
 * @brief [DEPRECATED] Calculates the derivative of Mean Squared Error.
 */
Matrix Network::mse_derivative(const Matrix &prediction, const Matrix &label) {
  return prediction.subtract(label).multiply(
      2.0 / (label.getRows() * label.getCols()));
}
