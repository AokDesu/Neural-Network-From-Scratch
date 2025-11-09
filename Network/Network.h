
#pragma once

#include "../Layer/Layer.h"
#include <vector>

class Network {
public:
  std::vector<Layer> layers;

  void addLayer(const Layer &layer);
  Matrix predict(const Matrix &input);
  void train(const std::vector<Matrix> &training_data, const std::vector<Matrix> &training_labels, int epochs,
             double learningRate, std::vector<Matrix> &test_images, std::vector<Matrix> &test_labels);

  int get_predicted_digit(const Matrix &prediction);
  // Loss function and its derivative
  static double mse(const Matrix &prediction, const Matrix &label);
  static Matrix mse_derivative(const Matrix &prediction, const Matrix &label);
  static double cross_entropy_loss(const Matrix &prediction, const Matrix &label);
  static Matrix cross_entropy_loss_derivative(const Matrix &prediction, const Matrix &label);
};
