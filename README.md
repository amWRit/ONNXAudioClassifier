# Audio Classification Using ONNX Runtime and Spectrogram Analysis

## Overview
This project implements an audio classification system that leverages the ONNX Runtime for efficient inference of a pre-trained model. The system processes audio files to extract relevant features using Short-Time Fourier Transform (STFT) to compute spectrograms, which are then fed into the model for classification.

## Key Features
- **Audio File Handling**: Utilizes the AudioFile library to read audio samples from WAV files, ensuring compatibility with various audio formats.
- **Spectrogram Computation**: Computes spectrograms from audio samples using KISS FFT, applying a Hanning window function to enhance frequency resolution.
- **ONNX Model Integration**: Integrates with ONNX Runtime to load a pre-trained model for audio classification, supporting efficient inference with multi-threading capabilities.
- **Class Mapping and Prediction**: Includes a mapping of musical instrument classes (e.g., flute, trumpet, violin) to their respective indices, allowing for easy interpretation of model predictions.

## Technical Details
- **Programming Language**: C++
- **Libraries Used**:
  - ONNX Runtime for model inference (`onnxruntime/core/session/onnxruntime_c_api.h`)
  - KISS FFT for Fast Fourier Transform computations
  - AudioFile library for reading audio files
- **Input Data**: Accepts audio files in WAV format, specifically designed to process mono-channel recordings.
- **Output Data**: Outputs predicted class names and their associated probabilities, providing insights into the model's confidence in its predictions.

## Implementation Steps
1. **Setup ONNX Runtime**: Initialize the environment and load the ONNX model.
2. **Load Audio Samples**: Use the `read_wav_file` function to retrieve audio samples from a specified file path.
3. **Compute Spectrogram**: Implement the `compute_spectrogram` function to transform audio samples into a frequency representation.
4. **Resize and Normalize Data**: Prepare the spectrogram data by resizing it to match the input shape expected by the model (1, 1, 64, 64) and normalizing the values.
5. **Run Inference**: Execute the model inference using the prepared input tensor and retrieve classification probabilities.
6. **Display Results**: Output the predicted class index and name along with their probabilities.

## Getting Started
1. Clone this repository:
```git clone https://github.com/yourusername/AudioClassifier.git```
2. Install necessary dependencies (if applicable).
3. Compile the project using your preferred build system (e.g., CMake).
    - ```cd build```
    -  ```cmake ..```
    - ```cmake --build .```
4. Run the application with your desired audio file.
    - Run .exe file (build > Debug)

## License
This project is licensed under the MIT License.

## Acknowledgments
- [KISS FFT](https://github.com/mborgerding/kissfft) for Fast Fourier Transform computations.
- [ONNX Runtime](https://onnxruntime.ai/) for efficient model inference.
- [IRMAS Project](https://www.upf.edu/web/mtg/irmas/) for dataset and building ONNX model file.
    - [Github](https://github.com/claudia-hm/IRMAS_Deep_Learning?tab=readme-ov-file/)