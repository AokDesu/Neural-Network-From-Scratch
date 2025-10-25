#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "./Mnist_Loader/mnist_loader.h"
#include "Layer/Layer.h"
#include "Network/Network.h"

// A helper function to find the predicted digit from the output matrix
int get_predicted_digit(const Matrix &prediction) {
  const auto &data = prediction.getData();
  // Find the iterator to the element with the maximum value
  auto max_it = std::max_element(
      data.begin(), data.end(), [](const std::vector<double> &a, const std::vector<double> &b) { return a[0] < b[0]; });
  // Return the index of that element
  return std::distance(data.begin(), max_it);
}

int main() {
  try {
    // --- 1. Load the MNIST Data ---
    std::cout << "Loading MNIST data..." << std::endl;
    std::vector<Matrix> train_images = load_mnist_images("../train_data/train-images.idx3-ubyte");
    std::vector<Matrix> train_labels = load_mnist_labels("../train_data/train-labels.idx1-ubyte");
    std::vector<Matrix> test_images = load_mnist_images("../test_data/t10k-images.idx3-ubyte");
    std::vector<Matrix> test_labels = load_mnist_labels("../test_data/t10k-labels.idx1-ubyte");
    std::cout << "Data loaded successfully." << std::endl;

    // --- 2. Define Activation Functions ---
    // ReLU for the hidden layer
    auto relu = [](const Matrix &m) {
      auto fn = [](double x) { return std::max(0.0, x); };
      return m.apply(fn);
    };
    auto relu_derivative = [](const Matrix &m) {
      auto fn = [](double x) { return x > 0 ? 1.0 : 0.0; };
      return m.apply(fn);
    };

    // Leaky ReLU for the hidden layers
    const double ALPHA = 0.01;
    auto leaky_relu = [ALPHA](const Matrix &m) { return m.apply([ALPHA](double x) { return x > 0 ? x : ALPHA * x; }); };
    auto leaky_relu_derivative = [ALPHA](const Matrix &m) {
      return m.apply([ALPHA](double x) { return x > 0 ? 1.0 : ALPHA; });
    };

    // Sigmoid for the output layer
    auto sigmoid = [](const Matrix &m) {
      auto fn = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
      return m.apply(fn);
    };
    auto sigmoid_derivative = [](const Matrix &m) {
      auto fn = [](double x) {
        double sig = 1.0 / (1.0 + std::exp(-x));
        return sig * (1.0 - sig);
      };
      return m.apply(fn);
    };
    // Softmax for the output layer
    auto softmax_activation = [](const Matrix &m) { return Matrix::softmax(m); };

    // --- 3. Build the Network Architecture ---
    Network nn;
    nn.addLayer(Layer(784, 256, leaky_relu, leaky_relu_derivative));
    // Input layer (784 pixels) -> 1st Hidden layer (256 neurons)

    nn.addLayer(Layer(256, 128, leaky_relu, leaky_relu_derivative));
    // 1st Hidden layer -> 2nd Hidden layer (128 neurons)

    nn.addLayer(Layer(128, 10, softmax_activation));
    // 2nd Hidden layer -> Output layer (10 neurons, for digits 0-9)

    // --- 4. Train the Network ---
    std::cout << "Starting training... (This may take around 45 minutes)\n";
    nn.train(train_images, train_labels, 10, 0.01); // Train for 10 epochs
    std::cout << "Training complete." << std::endl;

    // --- 5. Test the Network ---
    int correct_predictions = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
      Matrix prediction = nn.predict(test_images[i]);
      int predicted_digit = get_predicted_digit(prediction);
      int actual_digit = get_predicted_digit(test_labels[i]);

      if (predicted_digit == actual_digit) {
        correct_predictions++;
      }
    }

    double accuracy = static_cast<double>(correct_predictions) / test_images.size() * 100.0;
    std::cout << "\n--- Network Performance ---" << std::endl;
    std::cout << "Accuracy on test set: " << accuracy << "%" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
