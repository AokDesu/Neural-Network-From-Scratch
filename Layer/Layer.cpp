
/**
 * @file Layer.cpp
 * @brief Implements the Layer class, a core component of the neural network.
 *
 * A Layer object holds the parameters (weights and biases) for a single
 * layer and implements the forward and backward propagation logic for that
 * layer.
 */

#include "Layer.h"

// --- Constructors ---

/**
 * @brief Main constructor for a neural network layer.
 *
 * @param inputSize The number of neurons in the previous layer.
 * @param outputSize The number of neurons in this layer.
 * @param activation The activation function to be used (e.g., Leaky ReLU).
 * @param activationDerivative The derivative of the activation function.
 */
Layer::Layer(unsigned int inputSize, unsigned int outputSize, std::function<Matrix(const Matrix &)> activation,
             std::function<Matrix(const Matrix &)> activationDerivative)
    // Use a member initializer list for efficiency.
    // Initialize matrices with correct dimensions.
    : weights(outputSize, inputSize), biases(outputSize, 1), lastInput(0, 0), // Initialized as empty
      lastZ(0, 0)                                                             // Initialized as empty
{
  // Initialize weights and biases with small random values.
  weights.randomize();
  biases.randomize();

  // Store the provided activation and derivative functions.
  this->activation = activation;
  this->activationDerivative = activationDerivative;
}

/**
 * @brief Helper lambda for the output layer's "dummy" derivative.
 *
 * The math for Softmax + Cross-Entropy Loss simplifies the derivative
 * for the final layer. This lambda acts as a "dummy" derivative that
 * returns a matrix of ones, effectively bypassing the first step in the
 * backward pass for that layer.
 */
auto getMatrixOfOnes = [](const Matrix &m) {
  std::vector<std::vector<double>> ones(m.getRows(), std::vector<double>(m.getCols(), 1.0));
  return Matrix(ones);
};

/**
 * @brief Overloaded constructor, typically for the output layer.
 *
 * This constructor is a convenience for layers (like our Softmax layer)
 * where the activation derivative is handled by the loss function.
 * It automatically assigns the correct "dummy" derivative.
 *
 * @param inputSize The number of neurons in the previous layer.
 * @param outputSize The number of neurons in this layer.
 * @param activation The activation function to be used (e.g., Softmax).
 */
Layer::Layer(unsigned int inputSize, unsigned int outputSize, std::function<Matrix(const Matrix &)> activation)
    : weights(outputSize, inputSize), biases(outputSize, 1), lastInput(0, 0), lastZ(0, 0) {
  weights.randomize();
  biases.randomize();
  this->activation = activation;
  this->activationDerivative = getMatrixOfOnes; // Assign the dummy function
}

// --- Public Member Functions ---

/**
 * @brief Performs the forward pass calculation for this layer.
 *
 * Calculates: Output = Activation( (Weights * Input) + Biases )
 *
 * @param input The output matrix from the previous layer.
 * @return The activated output matrix for this layer.
 */
Matrix Layer::forward(const Matrix &input) {
  // Store the input and pre-activation sum (Z) for use in backpropagation.
  this->lastInput = input;

  // Step 1: Z = (Weights * Input)
  Matrix product = Matrix::multiply(this->weights, input);

  // Step 2: Z = Z + Biases
  Matrix sum = product.add(this->biases);
  this->lastZ = sum; // Store the pre-activation value

  // Step 3: A = Activation(Z)
  Matrix activated = this->activation(sum);

  return activated;
}

/**
 * @brief Performs the backward pass (backpropagation) for this layer.
 *
 * This function calculates the gradients for this layer's weights and biases,
 * updates them, and returns the error signal (gradient) for the previous layer.
 *
 * @param outputGradient The error signal (gradient) from the next layer.
 * @param learningRate The step size for gradient descent.
 * @return The error signal (gradient) to be passed to the previous layer.
 */
Matrix Layer::backward(const Matrix &outputGradient, double learningRate) {
  // This threshold is a "safety valve" to prevent exploding gradients.
  const double CLIP_THRESHOLD = 5.0;

  // --- Step 1: Calculate the core error signal (z_gradient) ---
  // z_gradient = output_gradient * activation_derivative(lastZ)
  Matrix activation_grad = this->activationDerivative(this->lastZ);
  Matrix z_gradient = outputGradient.multiply(activation_grad);

  // --- Step 2: Gradient Clipping at the Source ---
  // This is the most crucial part for stability. We cap the core
  // error signal itself, which prevents the explosion from
  // corrupting this layer's updates or the previous layers.
  double norm = z_gradient.norm();
  if (norm > CLIP_THRESHOLD) {
    z_gradient = z_gradient.divide(norm / CLIP_THRESHOLD);
  }

  // --- Step 3: Calculate Gradients for Parameters (Weights & Biases) ---
  // All gradients are now derived from the "safe," clipped z_gradient.
  Matrix lastInput_T = this->lastInput.transpose();
  Matrix weights_gradient = Matrix::multiply(z_gradient, lastInput_T);
  Matrix biases_gradient = z_gradient;

  // --- Step 4: Calculate Gradient to Pass to Previous Layer ---
  // This is the error signal for the layer before this one.
  Matrix weights_T = this->weights.transpose();
  Matrix input_gradient = Matrix::multiply(weights_T, z_gradient);

  // --- Step 5: Update Parameters (Gradient Descent) ---
  // Apply the updates using the gradients and learning rate.
  this->weights = this->weights.subtract(weights_gradient.multiply(learningRate));
  this->biases = this->biases.subtract(biases_gradient.multiply(learningRate));

  // Return the error signal for the next loop in the backward pass.
  return input_gradient;
}
