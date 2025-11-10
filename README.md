# This project is a part from Linear Algebra course
I made this project because I want to understand how Neural Network actually works under the hood and how it related to linear algebra.

## Requirements
1. CMAKE
2. C/C++ Compiler
3. Build tool

## How to set up project
### 1. clone the repository <br>
```
git clone <repo-url>
```
### 2. build the source code
**on linux**
```
cd Neural-Network-From-Scratch
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
**on windows**
```
cd Neural-Network-From-Scratch
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
mingw32-make
```

### 3. execute the output file
```
cd ../output
./Neural_Network
```
***Note:*** Usually, it takes round 45 minutes to complete.



## References
[MNIST datasets](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) <br>
[Gradient Decent blog](https://medium.com/@dishantkharkar9/gradient-descent-the-backbone-of-machine-learning-optimization-4acd8eed3b4f) <br>
[Learn to build a Neural Network from scratch blog](https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc) <br>
[Neural Network explanation playlists](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2) <br>
[Softmax blog explanation](https://medium.com/super-ai-engineer/softmax-function-%E0%B8%84%E0%B8%B7%E0%B8%AD%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3-eae1f1bbef63) <br>
[LeakyRelu blog explanation](https://www.geeksforgeeks.org/machine-learning/Leaky-Relu-Activation-Function-in-Deep-Learning) <br>
[Cross entropy loss blog explanation](https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5) <br>
**Note:** In this project, I used Gemini to help me learn about other activation functions suitable for MNIST datasets, and I also used Gemini to help me learn how to write proper C++.
<br>
> ***I also want to thank you for all the high-quality blogs and videos created for everyone to learn from.***
> ***Finally if there are any mistakes please let me know and I will fix them thank you for your attention.***
