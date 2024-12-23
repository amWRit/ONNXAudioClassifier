#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "kissfft/kiss_fft.h" // Include KISS FFT header
// #include "kissfft/kiss_fft.c"
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "../include/AudioFile.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
class Utils
{
    public:
        // Function to compute a spectrogram using STFT with KISS FFT
        static std::vector<std::vector<float>> compute_spectrogram(const std::vector<double>& audio_samples, int window_size, int overlap) {
            int hop_size = window_size - overlap; // Calculate hop size
            int num_windows = (audio_samples.size() - overlap) / hop_size; // Number of windows
            std::vector<std::vector<float>> spectrogram(num_windows, std::vector<float>(window_size / 2 + 1, 0.0f));

            // Allocate memory for KISS FFT
            kiss_fft_cfg fft_cfg = kiss_fft_alloc(window_size, 0, nullptr, nullptr);
            std::vector<kiss_fft_cpx> fft_input(window_size);
            std::vector<kiss_fft_cpx> fft_output(window_size);

            // Perform STFT
            for (int w = 0; w < num_windows; ++w) {
                // Apply windowing function (Hanning Window)
                for (int n = 0; n < window_size; ++n) {
                    if ((w * hop_size + n) < audio_samples.size()) {
                        double window_value = 0.5 * (1 - cos(2 * M_PI * n / (window_size - 1))); // Hanning window
                        fft_input[n].r = static_cast<float>(audio_samples[w * hop_size + n] * window_value); // Real part
                        fft_input[n].i = 0; // Imaginary part
                    } else {
                        fft_input[n].r = 0.0f; // Zero padding
                        fft_input[n].i = 0.0f;
                    }
                }

                // Execute FFT
                kiss_fft(fft_cfg, fft_input.data(), fft_output.data());

                // Compute magnitude spectrum
                for (int k = 0; k < window_size / 2 + 1; ++k) {
                    spectrogram[w][k] = std::sqrt(fft_output[k].r * fft_output[k].r + fft_output[k].i * fft_output[k].i); // Store magnitude
                }
            }

            kiss_fft_free(fft_cfg); // Free the FFT configuration memory
            return spectrogram;
        }


        // Function to read WAV file and return audio samples (assuming mono-channel)
        static std::vector<double> read_wav_file(const std::string& filename) {
            AudioFile<double> audio_file;
            if (!audio_file.load(filename)) {
                std::cerr << "Error loading file: " << filename << std::endl;
                std::cin.get();
                exit(1);
            }
            return audio_file.samples[0];  // Return the samples from the first channel
        }

        // Function to compute a simple spectrogram
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
};
