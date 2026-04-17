# Computer-Vision-Projects
Accuracy Killer 1: The "Centered Reflection" Padding
By default, librosa.feature.melspectrogram (which calls librosa.stft under the hood) does not just chop your audio into frames. It pads the beginning and end of your entire audio array with a "reflection" of the audio equal to N_FFT / 2.

If your C++ code just starts reading frames from index 0 without this reflection padding, every single frame is shifted, and the edges of your audio are lost. The model sees this as severe distortion.

The C++ Fix:

C++
#include <vector>

// Librosa pads the audio by N_FFT / 2 on both sides using "reflect" mode.
std::vector<float> pad_audio_reflect(const std::vector<float>& audio, int n_fft) {
    int pad_len = n_fft / 2;
    std::vector<float> padded_audio(audio.size() + 2 * pad_len);

    // 1. Copy original audio into the center
    for (size_t i = 0; i < audio.size(); ++i) {
        padded_audio[pad_len + i] = audio[i];
    }

    // 2. Reflect pad the beginning
    for (int i = 0; i < pad_len; ++i) {
        // Reflecting: index 1 goes to pad_len-1, index 2 goes to pad_len-2, etc.
        padded_audio[pad_len - 1 - i] = audio[i + 1]; 
    }

    // 3. Reflect pad the end
    for (int i = 0; i < pad_len; ++i) {
        padded_audio[pad_len + audio.size() + i] = audio[audio.size() - 2 - i];
    }

    return padded_audio;
}
Accuracy Killer 2: The Missing Hann Window
Before taking the FFT of a frame, Librosa multiplies the frame by a Hann Window. If you skip this in C++ and just pass raw chunks of audio to your FFT function, you introduce "spectral leakage" (high-frequency noise that destroys accuracy).

The C++ Fix:
You must generate the exact same Hann window Librosa uses and multiply it against every single frame before the FFT.

C++
#include <cmath>
#include <vector>

const double PI = 3.14159265358979323846;

// Generate Librosa's exact Hann Window once during initialization
std::vector<float> get_hann_window(int n_fft) {
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        // Librosa uses scipy.signal.get_window('hann', n_fft, fftbins=True)
        // The formula for fftbins=True is: 0.5 - 0.5 * cos(2.0 * PI * i / n_fft)
        window[i] = 0.5f - 0.5f * std::cos(2.0f * PI * i / n_fft);
    }
    return window;
}

// Inside your framing loop:
// for (int i = 0; i < n_fft; ++i) {
//     frame[i] = padded_audio[frame_start + i] * hann_window[i];
// }
Accuracy Killer 3: The Mel Filterbank Math
Librosa uses a very specific algorithm to generate the frequencies for its Mel filterbank (often defaulting to the Slaney formula). If your C++ library uses the HTK formula, or calculates the triangle bins even slightly differently, the final 128 values will be drastically altered.

The C++ Fix (The Cheat Code):
Do not try to write the Mel filterbank math in C++. Because your SAMPLE_RATE (22050), N_FFT (2048), and N_MELS (128) are fixed constants, the Librosa filterbank is a static matrix of size 128×1025.

Write a tiny Python script to export Librosa's matrix:

Python
import librosa
import numpy as np
mel_basis = librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128)
np.savetxt("mel_filters.h", mel_basis.flatten(), delimiter=",")
Hardcode that exported array into your C++ code. This guarantees 100% mathematical identicality.

C++
// 1. Include your exported 1D array (size 128 * 1025)
const float MEL_BASIS[131200] = { /* Paste the values from Python here */ };

// 2. Multiply your STFT power spectrum by the exact Librosa weights
std::vector<float> apply_mel_filterbank(const std::vector<float>& stft_power_frame, int n_mels, int n_fft_bins) {
    std::vector<float> mel_frame(n_mels, 0.0f);
    
    for (int m = 0; m < n_mels; ++m) {
        float sum = 0.0f;
        for (int k = 0; k < n_fft_bins; ++k) {
            // stft_power_frame is the (magnitude * magnitude) of your FFT output
            sum += MEL_BASIS[m * n_fft_bins + k] * stft_power_frame[k];
        }
        mel_frame[m] = sum;
    }
    return mel_frame;
}
The Final Validation Step
To debug this systematically, do not look at the final TFLite accuracy. You need to compare the intermediate steps. In Python, print the exact values of np.abs(librosa.stft(audio))[:5, 0]. Then print the exact output of your C++ FFT for the first frame. They must match up to at least 4 decimal places before you even attempt to feed it to the model.

What C++ library are you currently using to compute the actual Fast Fourier Transform (FFT) step on your Tizen device, and have you verified its raw output against Python's librosa.stft?






###############################################################################################################


Put this at the top of your file. Notice we are assuming you exported the mel_filters.h from Python as discussed previously.


#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// 1. External Single-File Libraries
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

// 2. Your Exported Python Mel Filterbank
// (Generated via: np.savetxt("mel_filters.h", librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128).flatten(), delimiter=","))
#include "mel_filters.h" 

// 3. TensorFlow Lite Headers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// --- CONSTANTS ---
constexpr int SAMPLE_RATE = 22050;
constexpr size_t AUDIO_LENGTH = 27562; 
constexpr int N_MELS = 128;
constexpr int N_FFT = 2048;
constexpr int HOP_LENGTH = 512;
const double PI = 3.14159265358979323846;







std::vector<float> load_audio(const char* file_path) {
    unsigned int channels, sampleRate;
    drwav_uint64 totalSampleCount;
    float* pSampleData = drwav_open_file_and_read_pcm_frames_f32(
        file_path, &channels, &sampleRate, &totalSampleCount, NULL);

    std::vector<float> audio;
    if (!pSampleData) return std::vector<float>(AUDIO_LENGTH, 0.0f);

    // Convert to Mono
    if (channels > 1) {
        audio.resize(totalSampleCount);
        for (size_t i = 0; i < totalSampleCount; ++i) {
            float sum = 0.0f;
            for (unsigned int c = 0; c < channels; ++c) sum += pSampleData[i * channels + c];
            audio[i] = sum / channels;
        }
    } else {
        audio.assign(pSampleData, pSampleData + totalSampleCount);
    }
    drwav_free(pSampleData, NULL);

    // Truncate or Pad to exact AUDIO_LENGTH
    if (audio.size() < AUDIO_LENGTH) audio.resize(AUDIO_LENGTH, 0.0f);
    else audio.resize(AUDIO_LENGTH);

    return audio;
}







// Helper 1: Librosa's Reflection Padding (FIX #1)
std::vector<float> pad_audio_reflect(const std::vector<float>& audio, int n_fft) {
    int pad_len = n_fft / 2;
    std::vector<float> padded(audio.size() + 2 * pad_len);
    for (size_t i = 0; i < audio.size(); ++i) padded[pad_len + i] = audio[i];
    for (int i = 0; i < pad_len; ++i) padded[pad_len - 1 - i] = audio[i + 1]; 
    for (int i = 0; i < pad_len; ++i) padded[pad_len + audio.size() + i] = audio[audio.size() - 2 - i];
    return padded;
}

// Helper 2: Librosa's Hann Window (FIX #2)
std::vector<float> get_hann_window(int n_fft) {
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i) window[i] = 0.5f - 0.5f * std::cos(2.0f * PI * i / n_fft);
    return window;
}

// Main Feature Extractor
std::vector<std::vector<float>> extract_mel_spectrogram(const std::vector<float>& raw_audio) {
    // 1. Pad the audio (Fix #1)
    std::vector<float> padded_audio = pad_audio_reflect(raw_audio, N_FFT);
    std::vector<float> window = get_hann_window(N_FFT); // Fix #2

    int num_frames = 1 + (raw_audio.size() / HOP_LENGTH);
    int num_fft_bins = 1 + (N_FFT / 2);
    
    // We will store the final Mel Spectrogram here: [N_MELS][num_frames]
    std::vector<std::vector<float>> mel_spec(N_MELS, std::vector<float>(num_frames, 0.0f));

    // 2. Loop through frames
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        int start_idx = frame_idx * HOP_LENGTH;
        std::vector<float> frame(N_FFT, 0.0f);
        
        // Apply Hann Window before FFT
        for (int i = 0; i < N_FFT; ++i) {
            frame[i] = padded_audio[start_idx + i] * window[i];
        }

        // --- FAST FOURIER TRANSFORM ---
        // YOU MUST IMPLEMENT THIS using KISS FFT or similar
        // std::vector<float> power_spectrum = compute_kiss_fft_power(frame);
        std::vector<float> power_spectrum(num_fft_bins, 0.0f); // PLACEHOLDER

        // 3. Apply Mel Filterbank (Fix #3)
        for (int m = 0; m < N_MELS; ++m) {
            float mel_sum = 0.0f;
            for (int k = 0; k < num_fft_bins; ++k) {
                // MEL_BASIS comes from your mel_filters.h file
                mel_sum += MEL_BASIS[m * num_fft_bins + k] * power_spectrum[k];
            }
            mel_spec[m][frame_idx] = mel_sum;
        }
    }

    // 4. Power to DB (Exact Librosa Math)
    constexpr float amin = 1e-10f;
    constexpr float top_db = 80.0f;
    float ref_value = amin;

    // Find max for reference
    for (const auto& row : mel_spec) 
        for (float val : row) if (val > ref_value) ref_value = val;

    float log_spec_ref = 10.0f * std::log10(ref_value);
    float max_db = -1e9f;

    // Convert to DB
    for (auto& row : mel_spec) {
        for (float& val : row) {
            float val_db = 10.0f * std::log10(std::max(amin, val)) - log_spec_ref;
            if (val_db > max_db) max_db = val_db;
            val = val_db;
        }
    }

    // Apply Top DB Clipping
    float threshold = max_db - top_db;
    for (auto& row : mel_spec) 
        for (float& val : row) if (val < threshold) val = threshold;

    return mel_spec;
}


int run_audio_inference(const char* model_path, const std::vector<std::vector<float>>& mel_spec) {
    // 1. Load Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();

    // 2. Expand Dims & Fill Input Tensor
    // Flatten the 2D vector [128][Frames] into the 1D TFLite input tensor
    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    int tensor_idx = 0;
    
    for (const auto& row : mel_spec) {
        for (float val : row) {
            input_tensor[tensor_idx++] = val;
        }
    }

    // 3. Run Inference
    interpreter->Invoke();

    // 4. Get Output (np.argmax equivalent)
    float* output_tensor = interpreter->typed_output_tensor<float>(0);
    int num_classes = interpreter->tensor(interpreter->outputs()[0])->dims->data[1];
    
    int best_class = 0;
    float best_prob = output_tensor[0];
    
    for (int i = 1; i < num_classes; ++i) {
        if (output_tensor[i] > best_prob) {
            best_prob = output_tensor[i];
            best_class = i;
        }
    }

    std::cout << "Predicted Class: " << best_class << " with confidence: " << best_prob << std::endl;
    return best_class;
}



import librosa
import numpy as np

# 1. Generate the exact matrix
mel_basis = librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128)

# 2. Save it as a C++ header file
# This formats the numbers so C++ can read them directly as a constant array
np.savetxt(
    "mel_filters.h", 
    mel_basis.flatten(), 
    delimiter=",", 
    header="const float MEL_BASIS[131200] = {", 
    footer="};", 
    comments=""
)

print("Success! mel_filters.h has been created.")
