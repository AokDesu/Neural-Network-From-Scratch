#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "./Mnist_Loader/mnist_loader.h"
#include "Layer/Layer.h"
#include "Network/Network.h"

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
    nn.train(train_images, train_labels, 10, 0.001, test_images, test_labels); // Train for 10 epochs
    std::cout << "Training complete." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
