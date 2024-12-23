cmake_minimum_required(VERSION 3.10)

project(AudioDeepLearning)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add executable
# add_executable(AudioDeepLearning src/main.cpp)
add_executable(AudioDeepLearning src/main.cpp include/kissfft/kiss_fft.c)

# Include directories
include_directories(
    "C:/My Documents/DSPProjects/onnxruntime/include"
    "C:/My Documents/DSPProjects/onnxruntime/include/onnxruntime/core"
    "C:/My Documents/DSPProjects/onnxruntime/include/onnxruntime/core/session"
    "C:/My Documents/DSPProjects/onnxruntime/include/onnxruntime/core/providers"
    "C:/My Documents/DSPProjects/onnxruntime/include/onnxruntime/core/providers/cpu"
    "C:/My Documents/DSPProjects/onnxruntime/include/onnxruntime/core/providers/dml",
)

# Link ONNX Runtime library
target_link_libraries(AudioDeepLearning PRIVATE
    "C:/My Documents/DSPProjects/onnxruntime/build/Windows/RelWithDebInfo/Debug/onnxruntime.lib"
)

# Link the ONNX Runtime dynamic library (DLL)
# target_link_libraries(AudioDeepLearning
#     "C:/My Documents/DSPProjects/onnxruntime/build/Windows/RelWithDebInfo/Debug/onnxruntime.dll"
# )
