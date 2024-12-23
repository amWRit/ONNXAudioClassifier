#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
// #include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include "../include/AudioFile.h"
#include "../include/Utils.h"
#include "../include/kissfft/kiss_fft.h" // Correctly include KISS FFT header

// // Function to read WAV file and return audio samples (assuming mono-channel)
// static std::vector<double> read_wav_file(const std::string& filename) {
//     AudioFile<double> audio_file;
//     if (!audio_file.load(filename)) {
//         std::cerr << "Error loading file: " << filename << std::endl;
//         std::cin.get();
//         exit(1);
//     }
//     return audio_file.samples[0];  // Return the samples from the first channel
// }

// // Function to compute a simple spectrogram
// static std::vector<float> compute_spectrogram(const std::vector<double>& audio_samples, int width, int height) {
//     std::vector<float> spectrogram(width * height, 0.0f);
//     for (int i = 0; i < width * height; ++i) {
//         spectrogram[i] = static_cast<float>(audio_samples[i % audio_samples.size()]);
//     }
//     return spectrogram;
// }

// Function to get class name from index
static std::string get_class_name(int index, const std::unordered_map<int, std::string>& reverse_class_mapping) {
    auto it = reverse_class_mapping.find(index);
    if (it != reverse_class_mapping.end()) {
        return it->second;
    }
    return "Unknown";
}

int main() {
    // Class mapping
    std::unordered_map<std::string, int> class_mapping = {
        {"flu", 0}, // Flute
        {"tru", 1}, // Trumpet
        {"vio", 2}, // Violin
        {"gac", 3}, // Guitar
        {"pia", 4}, // Piano
        {"cel", 5}, // Cello
        {"cla", 6}, // Clarinet
        {"gel", 7}, // Glockenspiel
        {"org", 8}, // Organ
        {"sax", 9}  // Saxophone
    };

    // Reverse mapping for class names
    std::unordered_map<int, std::string> reverse_class_mapping;
    for (const auto& pair : class_mapping) {
        reverse_class_mapping[pair.second] = pair.first;
    }

    std::cout << "Setting up ONNX Runtime..." << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    std::string model_path = "C:/My Documents/DSPProjects/AudioDeepLearning/model.onnx";
    std::wstring model_path_wide(model_path.begin(), model_path.end());

    std::unique_ptr<Ort::Session> session;
    try {
        session = std::make_unique<Ort::Session>(env, model_path_wide.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "Error creating session: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully!" << std::endl;

    // Load the audio file
    std::string audio_filename = "C:/My Documents/DSPProjects/AudioDeepLearning/audio/test1.wav";
    std::vector<double> audio_samples = Utils::read_wav_file(audio_filename);

    // Compute the spectrogram
    int spectrogram_width = 64;  
    int spectrogram_height = 64; 
    // std::vector<float> spectrogram_data = Utils::compute_spectrogram(audio_samples, spectrogram_width, spectrogram_height);

    int window_size = 1024; // Size of each analysis window
    int overlap = 512;       // Number of overlapping samples
    auto spectrogram_2d = Utils::compute_spectrogram(audio_samples, window_size, overlap);

    // Debugging output to confirm dimensions
    std::cout << "Number of windows: " << spectrogram_2d.size() << std::endl;
    if (!spectrogram_2d.empty()) {
        std::cout << "Width of first window: " << spectrogram_2d[0].size() << std::endl;
    } else {
        std::cout << "Spectrogram is empty!" << std::endl;
    }

    // Flatten and normalize the spectrogram data
    // std::vector<float> spectrogram_data;
    // float max_value = 0.0f;

    // Assuming spectrogram_2d has been computed correctly
    std::vector<float> resized_spectrogram_data(64 * 64); // Initialize for 64x64

    // Resize logic: map values from spectrogram_2d to resized_spectrogram_data
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            // Calculate source indices based on original size
            int source_i = static_cast<int>(i * (static_cast<float>(spectrogram_2d.size()) / 64));
            int source_j = static_cast<int>(j * (static_cast<float>(spectrogram_2d[0].size()) / 64));
            resized_spectrogram_data[i * 64 + j] = spectrogram_2d[source_i][source_j]; // Map values correctly
        }
    }

    // Find maximum value for normalization
    float max_value = *std::max_element(resized_spectrogram_data.begin(), resized_spectrogram_data.end());

    // Flattening and normalizing
    for (auto& value : resized_spectrogram_data) {
        value /= max_value; // Normalize to range [0, 1]
    }

    // Check if spectrogram is empty
    if (resized_spectrogram_data.empty() || resized_spectrogram_data.empty()) {
        std::cerr << "Error: Spectrogram data is empty." << std::endl;
        return -1; // or handle the error appropriately
    }

    // Print dimensions for debugging
    std::cout << "Resized Spectrogram Dimensions: " << 64 << " x " << 64 << std::endl; // Since you defined it as 64x64
    std::cout << "Total number of elements: " << resized_spectrogram_data.size() << std::endl; // This will be 4096

    // Reshape the spectrogram data to (1, 1, 64, 64)
    std::vector<int64_t> input_tensor_shape = {1, 1, spectrogram_height, spectrogram_width};

    // Define input tensor shape as before
    // std::vector<int64_t> input_tensor_shape = {1, 1, 
                                                // static_cast<int64_t>(spectrogram_2d.size()), 
                                                // static_cast<int64_t>(spectrogram_2d[0].size())};

    // Create memory info for CPU
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create the input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        resized_spectrogram_data.data(),
        resized_spectrogram_data.size(),
        input_tensor_shape.data(),
        input_tensor_shape.size()
    );

    // Run inference
    const char* input_names[] = {"input"};   // Adjust according to your model's input name
    const char* output_names[] = {"output"}; // Adjust according to your model's output name

    try {
        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                           input_names,
                                           &input_tensor,
                                           1,
                                           output_names,
                                           1);

        float* output_data = output_tensors.front().GetTensorMutableData<float>();

        // Calculate probabilities using softmax manually (or use a library)
        std::vector<float> probabilities(output_data, output_data + class_mapping.size());
        float sum_exp = 0.0f;
        
        for (float& prob : probabilities) {
            prob = exp(prob); // Calculate e^x for each element
            sum_exp += prob;
        }
        
        for (float& prob : probabilities) {
            prob /= sum_exp; // Normalize to get probabilities
        }

        // Print class names and probabilities
        for (int i = 0; i < probabilities.size(); ++i) {
            // Ensure that the index 'i' is within the bounds of reverse_class_mapping
            auto it = reverse_class_mapping.find(i);
            if (it != reverse_class_mapping.end()) {
                std::cout << "Class: " << it->second 
                        << ", Probability: " << probabilities[i] << std::endl;
            } else {
                std::cout << "Class index " << i << " not found in mapping." << std::endl;
            }
        }

        // Get predicted class index using size_t
        std::ptrdiff_t predicted_class_index = std::distance(probabilities.begin(), 
                                                    std::max_element(probabilities.begin(), probabilities.end()));

        // Output the predicted class index
        std::cout << "Predicted class index: " << predicted_class_index << std::endl;

        // Retrieve and print the predicted class name using the index
        auto predicted_class_name_it = reverse_class_mapping.find(predicted_class_index);
        if (predicted_class_name_it != reverse_class_mapping.end()) {
            std::cout << "Predicted class name: " << predicted_class_name_it->second << std::endl;
        } else {
            std::cout << "Predicted class name not found for index: " << predicted_class_index << std::endl;
}

    } catch (const Ort::Exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }
    std::cin.get();
    return 0;
}


