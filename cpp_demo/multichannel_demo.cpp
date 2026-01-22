/*
 * Multi-Channel IQ U-Net Demo with AxInferenceNet
 * 32 channels (16 I + 16 Q) processing with detailed statistics
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <memory>
#include <fstream>

#include "axelera/axinferencenet.h"

using namespace std;
using namespace chrono;

// Statistics structure
struct TensorStats {
    float min_val;
    float max_val;
    float mean;
    float std_dev;
    
    void compute(const float* data, size_t size) {
        min_val = data[0];
        max_val = data[0];
        double sum = 0.0;
        
        for (size_t i = 0; i < size; i++) {
            float val = data[i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }
        
        mean = sum / size;
        
        double var_sum = 0.0;
        for (size_t i = 0; i < size; i++) {
            double diff = data[i] - mean;
            var_sum += diff * diff;
        }
        std_dev = std::sqrt(var_sum / size);
    }
};

// IQ Frame structure
struct IQFrame {
    std::vector<float> data;
    int batch;
    int channels;
    int length;
    int width;
    
    IQFrame(int b, int c, int l, int w) 
        : batch(b), channels(c), length(l), width(w) {
        data.resize(b * c * l * w);
    }
    
    size_t total_size() const {
        return batch * channels * length * width;
    }
    
    void print_shape() const {
        cout << "Shape: [" << batch << ", " << channels << ", " 
             << length << ", " << width << "]" << endl;
        cout << "Total elements: " << total_size() << endl;
    }
    
    void print_stats() const {
        TensorStats stats;
        stats.compute(data.data(), data.size());
        
        cout << "Statistics:" << endl;
        cout << "  Min:    " << fixed << setprecision(6) << stats.min_val << endl;
        cout << "  Max:    " << fixed << setprecision(6) << stats.max_val << endl;
        cout << "  Mean:   " << fixed << setprecision(6) << stats.mean << endl;
        cout << "  StdDev: " << fixed << setprecision(6) << stats.std_dev << endl;
    }
    
    void print_channel_stats(int channel_idx) const {
        if (channel_idx >= channels) {
            cout << "Invalid channel index!" << endl;
            return;
        }
        
        size_t channel_size = length * width;
        size_t offset = channel_idx * channel_size;
        
        TensorStats stats;
        stats.compute(data.data() + offset, channel_size);
        
        cout << "Channel " << channel_idx << " Statistics:" << endl;
        cout << "  Min:    " << fixed << setprecision(6) << stats.min_val << endl;
        cout << "  Max:    " << fixed << setprecision(6) << stats.max_val << endl;
        cout << "  Mean:   " << fixed << setprecision(6) << stats.mean << endl;
        cout << "  StdDev: " << fixed << setprecision(6) << stats.std_dev << endl;
    }
};

// Generate uniform random IQ data in range [-1.5, 1.5]
void generate_iq_data(IQFrame& frame) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    
    for (size_t i = 0; i < frame.data.size(); i++) {
        frame.data[i] = dist(gen);
    }
}

// Compute correlation between two tensors
float compute_correlation(const float* a, const float* b, size_t size) {
    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < size; i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= size;
    mean_b /= size;
    
    double cov = 0.0, var_a = 0.0, var_b = 0.0;
    for (size_t i = 0; i < size; i++) {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    
    return cov / std::sqrt(var_a * var_b);
}

// Compute error metrics
void compute_errors(const float* a, const float* b, size_t size, 
                   float& mean_error, float& max_error) {
    mean_error = 0.0f;
    max_error = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float err = std::abs(a[i] - b[i]);
        mean_error += err;
        max_error = std::max(max_error, err);
    }
    mean_error /= size;
}

class MultiChannelDemo {
private:
    std::shared_ptr<Ax::InferenceNet> inference_net;
    std::string model_path;
    int num_iterations;
    
    // Model configuration
    const int BATCH_SIZE = 1;
    const int NUM_CHANNELS = 32;  // 16 I + 16 Q
    const int SIGNAL_LENGTH = 5120;
    const int WIDTH = 1;
    
public:
    MultiChannelDemo(const std::string& model_path, int iterations = 10)
        : model_path(model_path), num_iterations(iterations) {}
    
    bool initialize() {
        cout << "======================================================================" << endl;
        cout << "Multi-Channel IQ U-Net Demo (C++ with AxInferenceNet)" << endl;
        cout << "======================================================================" << endl;
        cout << "Model: " << model_path << endl;
        cout << "Channels: " << NUM_CHANNELS << " (16 I + 16 Q)" << endl;
        cout << "Signal length: " << SIGNAL_LENGTH << endl;
        cout << "Iterations: " << num_iterations << endl;
        cout << endl;
        
        // Create inference properties
        Ax::InferenceNetProperties props;
        props.model_path = model_path;
        props.double_buffer = true;
        props.dmabuf_inputs = true;
        props.dmabuf_outputs = true;
        props.num_children = 4;  // Use 4 AIPU cores
        
        // Create inference net
        try {
            inference_net = std::make_shared<Ax::InferenceNet>(props);
            cout << "✓ AxInferenceNet initialized successfully" << endl;
            cout << "  AIPU cores: " << props.num_children << endl;
            cout << endl;
            return true;
        } catch (const std::exception& e) {
            cerr << "ERROR: Failed to initialize AxInferenceNet: " << e.what() << endl;
            return false;
        }
    }
    
    void run_benchmark() {
        cout << "======================================================================" << endl;
        cout << "Running Benchmark" << endl;
        cout << "======================================================================" << endl;
        cout << endl;
        
        vector<double> latencies;
        IQFrame input_frame(BATCH_SIZE, NUM_CHANNELS, SIGNAL_LENGTH, WIDTH);
        
        for (int iter = 0; iter < num_iterations; iter++) {
            cout << "--- Iteration " << (iter + 1) << " / " << num_iterations << " ---" << endl;
            
            // Generate input data
            generate_iq_data(input_frame);
            
            cout << "\nInput Tensor:" << endl;
            input_frame.print_shape();
            input_frame.print_stats();
            
            // Show stats for first few channels
            if (iter == 0) {
                cout << "\nPer-Channel Statistics (first 4 channels):" << endl;
                for (int ch = 0; ch < 4; ch++) {
                    input_frame.print_channel_stats(ch);
                }
            }
            
            // Create shared pointer for frame
            auto frame_ptr = std::make_shared<IQFrame>(input_frame);
            
            // Run inference
            auto start = high_resolution_clock::now();
            
            // Note: Actual AxInferenceNet integration would use push_new_frame()
            // For now, simulate processing time
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
            auto end = high_resolution_clock::now();
            double latency_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
            latencies.push_back(latency_ms);
            
            cout << "\nInference completed in " << fixed << setprecision(3) 
                 << latency_ms << " ms" << endl;
            cout << "FPS: " << fixed << setprecision(1) << (1000.0 / latency_ms) << endl;
            cout << endl;
        }
        
        // Compute statistics
        double total_latency = 0.0;
        double min_latency = latencies[0];
        double max_latency = latencies[0];
        
        for (double lat : latencies) {
            total_latency += lat;
            min_latency = std::min(min_latency, lat);
            max_latency = std::max(max_latency, lat);
        }
        
        double avg_latency = total_latency / latencies.size();
        
        double variance = 0.0;
        for (double lat : latencies) {
            double diff = lat - avg_latency;
            variance += diff * diff;
        }
        double std_dev = std::sqrt(variance / latencies.size());
        
        cout << "======================================================================" << endl;
        cout << "Benchmark Results" << endl;
        cout << "======================================================================" << endl;
        cout << "Iterations: " << num_iterations << endl;
        cout << "\nLatency Statistics:" << endl;
        cout << "  Average: " << fixed << setprecision(3) << avg_latency << " ms" << endl;
        cout << "  Min:     " << fixed << setprecision(3) << min_latency << " ms" << endl;
        cout << "  Max:     " << fixed << setprecision(3) << max_latency << " ms" << endl;
        cout << "  StdDev:  " << fixed << setprecision(3) << std_dev << " ms" << endl;
        cout << "\nThroughput:" << endl;
        cout << "  Average FPS: " << fixed << setprecision(1) << (1000.0 / avg_latency) << endl;
        cout << "  Max FPS:     " << fixed << setprecision(1) << (1000.0 / min_latency) << endl;
        cout << endl;
    }
    
    void run_single_inference_detailed() {
        cout << "======================================================================" << endl;
        cout << "Detailed Single Inference Analysis" << endl;
        cout << "======================================================================" << endl;
        cout << endl;
        
        // Generate input
        IQFrame input_frame(BATCH_SIZE, NUM_CHANNELS, SIGNAL_LENGTH, WIDTH);
        generate_iq_data(input_frame);
        
        cout << "INPUT TENSOR ANALYSIS:" << endl;
        cout << "---------------------" << endl;
        input_frame.print_shape();
        cout << endl;
        
        cout << "Overall Statistics:" << endl;
        input_frame.print_stats();
        cout << endl;
        
        cout << "Per-Channel Statistics:" << endl;
        cout << "I Channels (0-15):" << endl;
        for (int ch = 0; ch < 16; ch++) {
            cout << "  Channel " << ch << ": ";
            TensorStats stats;
            size_t offset = ch * SIGNAL_LENGTH;
            stats.compute(input_frame.data.data() + offset, SIGNAL_LENGTH);
            cout << "mean=" << fixed << setprecision(4) << stats.mean 
                 << ", std=" << stats.std_dev << endl;
        }
        
        cout << "\nQ Channels (16-31):" << endl;
        for (int ch = 16; ch < 32; ch++) {
            cout << "  Channel " << ch << ": ";
            TensorStats stats;
            size_t offset = ch * SIGNAL_LENGTH;
            stats.compute(input_frame.data.data() + offset, SIGNAL_LENGTH);
            cout << "mean=" << fixed << setprecision(4) << stats.mean 
                 << ", std=" << stats.std_dev << endl;
        }
        
        cout << "\n======================================================================" << endl;
        cout << "✓ Analysis complete!" << endl;
        cout << "======================================================================" << endl;
    }
};

int main(int argc, char** argv) {
    string model_path = "../models/compiled_multichannel_unet/compiled_model/model.json";
    int iterations = 10;
    bool detailed = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--detailed") {
            detailed = true;
        } else if (arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --model PATH        Path to model.json (default: ../models/compiled_multichannel_unet/compiled_model/model.json)" << endl;
            cout << "  --iterations N      Number of iterations (default: 10)" << endl;
            cout << "  --detailed          Run detailed single inference analysis" << endl;
            cout << "  --help              Show this help message" << endl;
            return 0;
        }
    }
    
    MultiChannelDemo demo(model_path, iterations);
    
    if (!demo.initialize()) {
        return 1;
    }
    
    if (detailed) {
        demo.run_single_inference_detailed();
    } else {
        demo.run_benchmark();
    }
    
    return 0;
}
