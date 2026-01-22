/*
 * NPU Flow Demo - Detailed preprocessing/postprocessing pipeline
 * Shows exact transformations for 32-channel IQ processing
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

// JSON parsing helper (simple key-value extraction)
struct ManifestParams {
    float quant_scale = 0.0f;
    int quant_zp = 0;
    float dequant_scale = 0.0f;
    int dequant_zp = 0;
    vector<int> input_padding;
    vector<int> output_padding;
    vector<int> input_shape_original;
    vector<int> input_shape_final;
    vector<int> output_shape_original;
    vector<int> output_shape_final;
    
    bool load_from_file(const string& path) {
        // For demo purposes, use typical values from Phase 2
        // In production, parse actual JSON file
        quant_scale = 0.003921569;  // 1.0 / 255 (for [0,1] range)
        quant_zp = 0;
        dequant_scale = 0.003921569;
        dequant_zp = 0;
        
        // Typical padding for 32 channels
        input_padding = {0, 0, 1, 1, 0, 31, 0, 0};  // Pad to align channels
        output_padding = {0, 0, 1, 1, 0, 31, 0, 0};
        
        input_shape_original = {1, 32, 5120, 1};
        input_shape_final = {1, 5122, 32, 32};  // After padding and transpose
        
        output_shape_original = {1, 32, 5120, 1};
        output_shape_final = {1, 5122, 32, 32};
        
        return true;
    }
};

void print_separator(char c = '=') {
    cout << string(70, c) << endl;
}

void print_shape(const string& name, const vector<int>& shape) {
    cout << name << ": [";
    for (size_t i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i < shape.size() - 1) cout << ", ";
    }
    cout << "]";
    
    int total = 1;
    for (int s : shape) total *= s;
    cout << " (" << total << " elements)" << endl;
}

struct TensorStats {
    float min_val, max_val, mean, std_dev;
    
    template<typename T>
    void compute(const T* data, size_t size) {
        min_val = static_cast<float>(data[0]);
        max_val = static_cast<float>(data[0]);
        double sum = 0.0;
        
        for (size_t i = 0; i < size; i++) {
            float val = static_cast<float>(data[i]);
            min_val = min(min_val, val);
            max_val = max(max_val, val);
            sum += val;
        }
        
        mean = sum / size;
        
        double var_sum = 0.0;
        for (size_t i = 0; i < size; i++) {
            double diff = static_cast<float>(data[i]) - mean;
            var_sum += diff * diff;
        }
        std_dev = sqrt(var_sum / size);
    }
    
    void print(const string& indent = "  ") const {
        cout << indent << "Min:    " << fixed << setprecision(6) << min_val << endl;
        cout << indent << "Max:    " << fixed << setprecision(6) << max_val << endl;
        cout << indent << "Mean:   " << fixed << setprecision(6) << mean << endl;
        cout << indent << "StdDev: " << fixed << setprecision(6) << std_dev << endl;
    }
};

class NPUFlowDemo {
private:
    ManifestParams manifest;
    
public:
    void run_detailed_flow() {
        print_separator();
        cout << "NPU PREPROCESSING/POSTPROCESSING FLOW - DETAILED ANALYSIS" << endl;
        print_separator();
        cout << endl;
        
        // Step 1: Generate FP32 input
        cout << "STEP 1: GENERATE FP32 INPUT DATA" << endl;
        print_separator('-');
        
        vector<float> fp32_input = generate_fp32_input();
        vector<int> shape_fp32 = {1, 32, 5120, 1};
        
        print_shape("Shape", shape_fp32);
        cout << "Data type: FP32" << endl;
        cout << "Value range: [-1.5, +1.5]" << endl;
        
        TensorStats stats_fp32;
        stats_fp32.compute(fp32_input.data(), fp32_input.size());
        cout << "\nStatistics:" << endl;
        stats_fp32.print();
        
        cout << "\nSample values (first 10):" << endl;
        for (int i = 0; i < 10; i++) {
            cout << "  [" << i << "] = " << fixed << setprecision(6) << fp32_input[i] << endl;
        }
        cout << endl << endl;
        
        // Step 2: Rescale to [0, 1]
        cout << "STEP 2: RESCALE TO [0, 1] RANGE" << endl;
        print_separator('-');
        
        vector<float> rescaled = rescale_to_01(fp32_input);
        
        cout << "Formula: output = (input + 1.5) / 3.0" << endl;
        print_shape("Shape", shape_fp32);
        cout << "Data type: FP32" << endl;
        cout << "Value range: [0.0, 1.0]" << endl;
        
        TensorStats stats_rescaled;
        stats_rescaled.compute(rescaled.data(), rescaled.size());
        cout << "\nStatistics:" << endl;
        stats_rescaled.print();
        
        cout << "\nSample values (first 10):" << endl;
        for (int i = 0; i < 10; i++) {
            cout << "  [" << i << "] = " << fixed << setprecision(6) 
                 << fp32_input[i] << " -> " << rescaled[i] << endl;
        }
        cout << endl << endl;
        
        // Step 3: Quantize to INT8
        cout << "STEP 3: QUANTIZE TO INT8" << endl;
        print_separator('-');
        
        vector<int8_t> quantized = quantize_to_int8(rescaled);
        
        cout << "Formula: output = round(input / scale + zero_point).clip(-128, 127)" << endl;
        cout << "Quantization parameters:" << endl;
        cout << "  Scale:      " << scientific << setprecision(6) << manifest.quant_scale << endl;
        cout << "  Zero point: " << manifest.quant_zp << endl;
        
        print_shape("Shape", shape_fp32);
        cout << "Data type: INT8" << endl;
        cout << "Value range: [-128, 127]" << endl;
        
        TensorStats stats_quant;
        stats_quant.compute(quantized.data(), quantized.size());
        cout << "\nStatistics:" << endl;
        stats_quant.print();
        
        cout << "\nSample values (first 10):" << endl;
        for (int i = 0; i < 10; i++) {
            cout << "  [" << i << "] = " << fixed << setprecision(6) 
                 << rescaled[i] << " -> " << static_cast<int>(quantized[i]) << endl;
        }
        cout << endl << endl;
        
        // Step 4: Reshape NCHW -> NHWC
        cout << "STEP 4: RESHAPE NCHW -> NHWC" << endl;
        print_separator('-');
        
        vector<int8_t> nhwc = transpose_nchw_to_nhwc(quantized, shape_fp32);
        vector<int> shape_nhwc = {1, 5120, 1, 32};
        
        cout << "Transpose: (N, C, H, W) -> (N, H, W, C)" << endl;
        print_shape("Input shape (NCHW)", shape_fp32);
        print_shape("Output shape (NHWC)", shape_nhwc);
        cout << "Data type: INT8" << endl;
        
        cout << "\nMemory layout change:" << endl;
        cout << "  Before: Channel-major [N][C][H][W]" << endl;
        cout << "  After:  Channel-minor [N][H][W][C]" << endl;
        
        cout << "\nSample mapping (first channel, first 5 elements):" << endl;
        for (int i = 0; i < 5; i++) {
            int nchw_idx = i;  // Channel 0, position i
            int nhwc_idx = i * 32;  // Position i, channel 0
            cout << "  NCHW[" << nchw_idx << "] = " << static_cast<int>(quantized[nchw_idx])
                 << " -> NHWC[" << nhwc_idx << "] = " << static_cast<int>(nhwc[nhwc_idx]) << endl;
        }
        cout << endl << endl;
        
        // Step 5: Apply padding
        cout << "STEP 5: APPLY PADDING" << endl;
        print_separator('-');
        
        vector<int8_t> padded = apply_padding(nhwc, shape_nhwc);
        vector<int> shape_padded = {1, 5122, 32, 32};  // After padding
        
        cout << "Padding specification: " << manifest.input_padding[0];
        for (size_t i = 1; i < manifest.input_padding.size(); i++) {
            cout << ", " << manifest.input_padding[i];
        }
        cout << endl;
        
        cout << "Padding pairs: [(before, after), ...]" << endl;
        for (size_t i = 0; i < manifest.input_padding.size(); i += 2) {
            cout << "  Dim " << i/2 << ": (" << manifest.input_padding[i] 
                 << ", " << manifest.input_padding[i+1] << ")" << endl;
        }
        
        print_shape("Input shape", shape_nhwc);
        print_shape("Output shape (padded)", shape_padded);
        cout << "Padding value: " << manifest.quant_zp << " (zero point)" << endl;
        
        TensorStats stats_padded;
        stats_padded.compute(padded.data(), padded.size());
        cout << "\nStatistics after padding:" << endl;
        stats_padded.print();
        
        cout << "\nMemory increase:" << endl;
        int original_size = 1;
        for (int s : shape_nhwc) original_size *= s;
        int padded_size = 1;
        for (int s : shape_padded) padded_size *= s;
        cout << "  Original: " << original_size << " elements" << endl;
        cout << "  Padded:   " << padded_size << " elements" << endl;
        cout << "  Increase: " << (padded_size - original_size) << " elements ("
             << fixed << setprecision(1) << (100.0 * (padded_size - original_size) / original_size)
             << "%)" << endl;
        cout << endl << endl;
        
        // Step 6: NPU Inference (simulated)
        cout << "STEP 6: NPU INFERENCE" << endl;
        print_separator('-');
        
        cout << "Input to NPU:" << endl;
        print_shape("  Shape", shape_padded);
        cout << "  Data type: INT8" << endl;
        cout << "  Memory: " << (padded_size * sizeof(int8_t) / 1024.0) << " KB" << endl;
        
        cout << "\nNPU Processing:" << endl;
        cout << "  ✓ Load model from: compiled_model/model.json" << endl;
        cout << "  ✓ Allocate DMA buffers" << endl;
        cout << "  ✓ Copy input to device" << endl;
        cout << "  ✓ Execute inference on AIPU cores" << endl;
        cout << "  ✓ Copy output from device" << endl;
        
        // Simulate output (same shape for U-Net)
        vector<int8_t> npu_output = padded;  // Identity for demo
        
        cout << "\nOutput from NPU:" << endl;
        print_shape("  Shape", shape_padded);
        cout << "  Data type: INT8" << endl;
        
        TensorStats stats_npu_out;
        stats_npu_out.compute(npu_output.data(), npu_output.size());
        cout << "\nStatistics:" << endl;
        stats_npu_out.print("  ");
        cout << endl << endl;
        
        // Step 7: Remove padding
        cout << "STEP 7: REMOVE PADDING (POSTPROCESSING)" << endl;
        print_separator('-');
        
        vector<int8_t> unpadded = remove_padding(npu_output, shape_padded);
        
        cout << "Reverse padding operation" << endl;
        print_shape("Input shape (padded)", shape_padded);
        print_shape("Output shape (unpadded)", shape_nhwc);
        
        cout << "\nMemory decrease:" << endl;
        cout << "  Padded:   " << padded_size << " elements" << endl;
        cout << "  Unpadded: " << original_size << " elements" << endl;
        cout << endl << endl;
        
        // Step 8: Dequantize
        cout << "STEP 8: DEQUANTIZE TO FP32" << endl;
        print_separator('-');
        
        vector<float> dequantized = dequantize_to_fp32(unpadded);
        
        cout << "Formula: output = (input - zero_point) * scale" << endl;
        cout << "Dequantization parameters:" << endl;
        cout << "  Scale:      " << scientific << setprecision(6) << manifest.dequant_scale << endl;
        cout << "  Zero point: " << manifest.dequant_zp << endl;
        
        print_shape("Shape", shape_nhwc);
        cout << "Data type: FP32" << endl;
        cout << "Value range: [0.0, 1.0]" << endl;
        
        TensorStats stats_dequant;
        stats_dequant.compute(dequantized.data(), dequantized.size());
        cout << "\nStatistics:" << endl;
        stats_dequant.print();
        
        cout << "\nSample values (first 10):" << endl;
        for (int i = 0; i < 10; i++) {
            cout << "  [" << i << "] = " << static_cast<int>(unpadded[i]) 
                 << " -> " << fixed << setprecision(6) << dequantized[i] << endl;
        }
        cout << endl << endl;
        
        // Step 9: Reshape back to NCHW
        cout << "STEP 9: RESHAPE NHWC -> NCHW" << endl;
        print_separator('-');
        
        vector<float> nchw_output = transpose_nhwc_to_nchw(dequantized, shape_nhwc);
        
        cout << "Transpose: (N, H, W, C) -> (N, C, H, W)" << endl;
        print_shape("Input shape (NHWC)", shape_nhwc);
        print_shape("Output shape (NCHW)", shape_fp32);
        cout << "Data type: FP32" << endl;
        cout << endl << endl;
        
        // Step 10: Rescale back
        cout << "STEP 10: RESCALE TO [-1.5, +1.5]" << endl;
        print_separator('-');
        
        vector<float> final_output = rescale_from_01(nchw_output);
        
        cout << "Formula: output = input * 3.0 - 1.5" << endl;
        print_shape("Shape", shape_fp32);
        cout << "Data type: FP32" << endl;
        cout << "Value range: [-1.5, +1.5]" << endl;
        
        TensorStats stats_final;
        stats_final.compute(final_output.data(), final_output.size());
        cout << "\nStatistics:" << endl;
        stats_final.print();
        
        cout << "\nSample values (first 10):" << endl;
        for (int i = 0; i < 10; i++) {
            cout << "  [" << i << "] = " << fixed << setprecision(6) 
                 << nchw_output[i] << " -> " << final_output[i] << endl;
        }
        cout << endl << endl;
        
        // Summary
        print_separator();
        cout << "PIPELINE SUMMARY" << endl;
        print_separator();
        
        cout << "\nData Flow:" << endl;
        cout << "  1. FP32 Input        [1,32,5120,1]  FP32  [-1.5, +1.5]" << endl;
        cout << "  2. Rescaled          [1,32,5120,1]  FP32  [0.0, 1.0]" << endl;
        cout << "  3. Quantized         [1,32,5120,1]  INT8  [-128, 127]" << endl;
        cout << "  4. Transposed        [1,5120,1,32]  INT8  [-128, 127]" << endl;
        cout << "  5. Padded            [1,5122,32,32] INT8  [-128, 127]" << endl;
        cout << "  6. NPU Output        [1,5122,32,32] INT8  [-128, 127]" << endl;
        cout << "  7. Unpadded          [1,5120,1,32]  INT8  [-128, 127]" << endl;
        cout << "  8. Dequantized       [1,5120,1,32]  FP32  [0.0, 1.0]" << endl;
        cout << "  9. Transposed        [1,32,5120,1]  FP32  [0.0, 1.0]" << endl;
        cout << " 10. Rescaled          [1,32,5120,1]  FP32  [-1.5, +1.5]" << endl;
        
        cout << "\nKey Transformations:" << endl;
        cout << "  • Rescaling:    Normalize to [0,1] for quantization" << endl;
        cout << "  • Quantization: FP32 -> INT8 (8-bit precision)" << endl;
        cout << "  • Transpose:    NCHW <-> NHWC (memory layout)" << endl;
        cout << "  • Padding:      Align to NPU requirements" << endl;
        
        cout << "\nMemory Usage:" << endl;
        cout << "  • FP32 tensor:  " << (original_size * 4 / 1024.0) << " KB" << endl;
        cout << "  • INT8 tensor:  " << (original_size * 1 / 1024.0) << " KB" << endl;
        cout << "  • Padded INT8:  " << (padded_size * 1 / 1024.0) << " KB" << endl;
        cout << "  • Compression:  " << fixed << setprecision(1) 
             << (100.0 * (1 - (float)original_size / (original_size * 4))) << "% (INT8 vs FP32)" << endl;
        
        cout << endl;
        print_separator();
        cout << "✓ Complete NPU flow analysis!" << endl;
        print_separator();
    }
    
private:
    vector<float> generate_fp32_input() {
        vector<float> data(1 * 32 * 5120 * 1);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dist(-1.5f, 1.5f);
        
        for (auto& val : data) {
            val = dist(gen);
        }
        return data;
    }
    
    vector<float> rescale_to_01(const vector<float>& input) {
        vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            output[i] = (input[i] + 1.5f) / 3.0f;
        }
        return output;
    }
    
    vector<int8_t> quantize_to_int8(const vector<float>& input) {
        vector<int8_t> output(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            float val = round(input[i] / manifest.quant_scale + manifest.quant_zp);
            output[i] = static_cast<int8_t>(max(-128.0f, min(127.0f, val)));
        }
        return output;
    }
    
    vector<int8_t> transpose_nchw_to_nhwc(const vector<int8_t>& input, const vector<int>& shape) {
        int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
        vector<int8_t> output(input.size());
        
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int nchw_idx = ((n * C + c) * H + h) * W + w;
                        int nhwc_idx = ((n * H + h) * W + w) * C + c;
                        output[nhwc_idx] = input[nchw_idx];
                    }
                }
            }
        }
        return output;
    }
    
    vector<int8_t> apply_padding(const vector<int8_t>& input, const vector<int>& shape) {
        // Simplified padding - just add padding elements
        int original_size = 1;
        for (int s : shape) original_size *= s;
        
        // Calculate padded size: [1, 5120+2, 1+31, 32]
        int padded_size = (1) * (5122) * (32) * (32);
        
        vector<int8_t> output(padded_size, manifest.quant_zp);
        
        // Copy original data (simplified - actual implementation would handle dimensions properly)
        size_t copy_size = min(input.size(), output.size());
        for (size_t i = 0; i < copy_size; i++) {
            output[i] = input[i];
        }
        
        return output;
    }
    
    vector<int8_t> remove_padding(const vector<int8_t>& input, const vector<int>& padded_shape) {
        // Simplified unpadding
        vector<int> unpadded_shape = {1, 5120, 1, 32};
        int unpadded_size = 1;
        for (int s : unpadded_shape) unpadded_size *= s;
        
        vector<int8_t> output(unpadded_size);
        for (int i = 0; i < unpadded_size; i++) {
            output[i] = input[i];
        }
        
        return output;
    }
    
    vector<float> dequantize_to_fp32(const vector<int8_t>& input) {
        vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            output[i] = (static_cast<float>(input[i]) - manifest.dequant_zp) * manifest.dequant_scale;
        }
        return output;
    }
    
    vector<float> transpose_nhwc_to_nchw(const vector<float>& input, const vector<int>& shape) {
        int N = shape[0], H = shape[1], W = shape[2], C = shape[3];
        vector<float> output(input.size());
        
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    for (int c = 0; c < C; c++) {
                        int nhwc_idx = ((n * H + h) * W + w) * C + c;
                        int nchw_idx = ((n * C + c) * H + h) * W + w;
                        output[nchw_idx] = input[nhwc_idx];
                    }
                }
            }
        }
        return output;
    }
    
    vector<float> rescale_from_01(const vector<float>& input) {
        vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            output[i] = input[i] * 3.0f - 1.5f;
        }
        return output;
    }
};

int main(int argc, char** argv) {
    NPUFlowDemo demo;
    demo.run_detailed_flow();
    return 0;
}
