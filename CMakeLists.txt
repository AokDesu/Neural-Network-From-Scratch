cmake_minimum_required(VERSION 3.10)

project(Neural_Network)

# Gather all source files
file(GLOB SOURCES
    main.cpp
    Layer/*.cpp
    Matrix/*.cpp
    Mnist_Loader/*.cpp
    Network/*.cpp
)

# Create the executable
add_executable(Neural_Network ${SOURCES})

# Specify the include directories (where the headers are located)
target_include_directories(Neural_Network
    PRIVATE
        ${PROJECT_SOURCE_DIR}/Layer
        ${PROJECT_SOURCE_DIR}/Matrix
        ${PROJECT_SOURCE_DIR}/Mnist_Loader
        ${PROJECT_SOURCE_DIR}/Network
)

# Set output directory
set_target_properties(Neural_Network PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output
)

# Set release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")

# Ensure these flags are used only in release mode
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
