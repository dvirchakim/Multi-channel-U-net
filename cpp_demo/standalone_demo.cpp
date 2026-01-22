/*
 * Standalone Multi-Channel IQ Demo
 * Shows raw stats and shapes without requiring AxInferenceNet headers
 * This version generates data and shows statistics only
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <fstream>

using namespace std;
using namespace chrono;

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
    
    void print() const {
        cout << "  Min:    " << fixed << setprecision(6) << min_val << endl;
        cout << "  Max:    " << fixed << setprecision(6) << max_val << endl;
        cout << "  Mean:   " << fixed << setprecision(6) << mean << endl;
        cout << "  StdDev: " << fixed << setprecision(6) << std_dev << endl;
    }
};

struct IQTensor {
    vector<float> data;
    int batch;
    int channels;
    int length;
    int width;
    
    IQTensor(int b, int c, int l, int w) 
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
        cout << "Memory size: " << (total_size() * sizeof(float) / 1024.0) << " KB" << endl;
    }
    
    void generate_random_data(float min_val = -1.5f, float max_val = 1.5f) {
        static random_device rd;
        static mt19937 gen(rd());
        uniform_real_distribution<float> dist(min_val, max_val);
        
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dist(gen);
        }
    }
    
    TensorStats get_overall_stats() const {
        TensorStats stats;
        stats.compute(data.data(), data.size());
        return stats;
    }
    
    TensorStats get_channel_stats(int channel_idx) const {
        if (channel_idx >= channels) {
            throw runtime_error("Invalid channel index");
        }
        
        size_t channel_size = length * width;
        size_t offset = channel_idx * channel_size;
        
        TensorStats stats;
        stats.compute(data.data() + offset, channel_size);
        return stats;
    }
    
    void save_to_file(const string& filename) const {
        ofstream file(filename, ios::binary);
        if (!file) {
            throw runtime_error("Cannot open file: " + filename);
        }
        
        // Write shape
        file.write(reinterpret_cast<const char*>(&batch), sizeof(int));
        file.write(reinterpret_cast<const char*>(&channels), sizeof(int));
        file.write(reinterpret_cast<const char*>(&length), sizeof(int));
        file.write(reinterpret_cast<const char*>(&width), sizeof(int));
        
        // Write data
        file.write(reinterpret_cast<const char*>(data.data()), 
                   data.size() * sizeof(float));
        
        file.close();
    }
};

void print_separator() {
    cout << "======================================================================" << endl;
}

void analyze_tensor_detailed(const IQTensor& tensor) {
    print_separator();
    cout << "DETAILED TENSOR ANALYSIS" << endl;
    print_separator();
    cout << endl;
    
    // Shape information
    cout << "TENSOR SHAPE:" << endl;
    tensor.print_shape();
    cout << endl;
    
    // Overall statistics
    cout << "OVERALL STATISTICS:" << endl;
    TensorStats overall = tensor.get_overall_stats();
    overall.print();
    cout << endl;
    
    // I Channels (0-15)
    cout << "I CHANNELS (0-15) STATISTICS:" << endl;
    cout << string(70, '-') << endl;
    for (int ch = 0; ch < 16; ch++) {
        TensorStats stats = tensor.get_channel_stats(ch);
        cout << "Channel " << setw(2) << ch << ": "
             << "mean=" << fixed << setprecision(4) << setw(8) << stats.mean 
             << ", std=" << setw(8) << stats.std_dev
             << ", range=[" << setw(8) << stats.min_val 
             << ", " << setw(8) << stats.max_val << "]" << endl;
    }
    cout << endl;
    
    // Q Channels (16-31)
    cout << "Q CHANNELS (16-31) STATISTICS:" << endl;
    cout << string(70, '-') << endl;
    for (int ch = 16; ch < 32; ch++) {
        TensorStats stats = tensor.get_channel_stats(ch);
        cout << "Channel " << setw(2) << ch << ": "
             << "mean=" << fixed << setprecision(4) << setw(8) << stats.mean 
             << ", std=" << setw(8) << stats.std_dev
             << ", range=[" << setw(8) << stats.min_val 
             << ", " << setw(8) << stats.max_val << "]" << endl;
    }
    cout << endl;
    
    // Channel pair analysis
    cout << "IQ CHANNEL PAIR ANALYSIS:" << endl;
    cout << string(70, '-') << endl;
    for (int pair = 0; pair < 16; pair++) {
        int i_ch = pair;
        int q_ch = pair + 16;
        
        TensorStats i_stats = tensor.get_channel_stats(i_ch);
        TensorStats q_stats = tensor.get_channel_stats(q_ch);
        
        cout << "Pair " << setw(2) << pair << " (I=" << setw(2) << i_ch 
             << ", Q=" << setw(2) << q_ch << "):" << endl;
        cout << "  I: mean=" << fixed << setprecision(4) << setw(8) << i_stats.mean 
             << ", std=" << setw(8) << i_stats.std_dev << endl;
        cout << "  Q: mean=" << fixed << setprecision(4) << setw(8) << q_stats.mean 
             << ", std=" << setw(8) << q_stats.std_dev << endl;
    }
    cout << endl;
}

void run_benchmark(int iterations) {
    print_separator();
    cout << "MULTI-CHANNEL IQ BENCHMARK" << endl;
    print_separator();
    cout << "Iterations: " << iterations << endl;
    cout << "Channels: 32 (16 I + 16 Q)" << endl;
    cout << "Signal length: 5120 samples" << endl;
    cout << endl;
    
    vector<double> generation_times;
    
    for (int iter = 0; iter < iterations; iter++) {
        cout << "Iteration " << (iter + 1) << " / " << iterations << "..." << endl;
        
        IQTensor tensor(1, 32, 5120, 1);
        
        auto start = high_resolution_clock::now();
        tensor.generate_random_data(-1.5f, 1.5f);
        auto end = high_resolution_clock::now();
        
        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        generation_times.push_back(time_ms);
        
        if (iter == 0) {
            cout << "\nFirst iteration details:" << endl;
            tensor.print_shape();
            TensorStats stats = tensor.get_overall_stats();
            cout << "\nStatistics:" << endl;
            stats.print();
        }
        
        cout << "  Generation time: " << fixed << setprecision(3) << time_ms << " ms" << endl;
        cout << endl;
    }
    
    // Compute statistics
    double total = 0.0, min_time = generation_times[0], max_time = generation_times[0];
    for (double t : generation_times) {
        total += t;
        min_time = min(min_time, t);
        max_time = max(max_time, t);
    }
    double avg = total / generation_times.size();
    
    double variance = 0.0;
    for (double t : generation_times) {
        double diff = t - avg;
        variance += diff * diff;
    }
    double std_dev = sqrt(variance / generation_times.size());
    
    print_separator();
    cout << "BENCHMARK RESULTS" << endl;
    print_separator();
    cout << "\nData Generation Performance:" << endl;
    cout << "  Average: " << fixed << setprecision(3) << avg << " ms" << endl;
    cout << "  Min:     " << fixed << setprecision(3) << min_time << " ms" << endl;
    cout << "  Max:     " << fixed << setprecision(3) << max_time << " ms" << endl;
    cout << "  StdDev:  " << fixed << setprecision(3) << std_dev << " ms" << endl;
    cout << endl;
}

int main(int argc, char** argv) {
    bool detailed = false;
    int iterations = 10;
    bool save_data = false;
    string output_file = "tensor_data.bin";
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--detailed") {
            detailed = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = stoi(argv[++i]);
        } else if (arg == "--save" && i + 1 < argc) {
            save_data = true;
            output_file = argv[++i];
        } else if (arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --detailed          Run detailed single tensor analysis" << endl;
            cout << "  --iterations N      Number of benchmark iterations (default: 10)" << endl;
            cout << "  --save FILE         Save tensor data to binary file" << endl;
            cout << "  --help              Show this help message" << endl;
            return 0;
        }
    }
    
    try {
        if (detailed) {
            IQTensor tensor(1, 32, 5120, 1);
            tensor.generate_random_data(-1.5f, 1.5f);
            analyze_tensor_detailed(tensor);
            
            if (save_data) {
                cout << "Saving tensor to: " << output_file << endl;
                tensor.save_to_file(output_file);
                cout << "✓ Tensor saved successfully" << endl;
            }
        } else {
            run_benchmark(iterations);
        }
        
        print_separator();
        cout << "✓ Complete!" << endl;
        print_separator();
        
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
