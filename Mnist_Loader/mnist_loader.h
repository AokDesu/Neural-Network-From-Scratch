// mnist_loader.h

#pragma once

#include "../Matrix/Matrix.h"
#include <string>
#include <vector>

// Loads the MNIST image data from the specified file
std::vector<Matrix> load_mnist_images(const std::string &path);

// Loads the MNIST label data, one-hot encoding them
std::vector<Matrix> load_mnist_labels(const std::string &path);
