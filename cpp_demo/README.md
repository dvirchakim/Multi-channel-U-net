# Multi-Channel C++ Demo

C++ implementation for 32-channel IQ processing with detailed statistics and shape information.

## Features

- **32-channel processing** (16 I + 16 Q)
- **Detailed tensor statistics** (min, max, mean, std dev)
- **Per-channel analysis** for all 16 IQ pairs
- **Performance benchmarking** with latency metrics
- **Raw shape information** for debugging

## Build

```bash
chmod +x build.sh
./build.sh
```

## Usage

### Basic Benchmark (10 iterations)
```bash
./build/multichannel_demo
```

### Extended Benchmark (100 iterations)
```bash
./build/multichannel_demo --iterations 100
```

### Detailed Single Inference Analysis
```bash
./build/multichannel_demo --detailed
```

Shows:
- Input tensor shape
- Overall statistics (min, max, mean, std dev)
- Per-channel statistics for all 32 channels
- Separate analysis for I channels (0-15) and Q channels (16-31)

### Custom Model Path
```bash
./build/multichannel_demo --model /path/to/model.json
```

### Help
```bash
./build/multichannel_demo --help
```

## Output Information

### Tensor Shape
```
Shape: [1, 32, 5120, 1]
Total elements: 163840
```

### Overall Statistics
```
Statistics:
  Min:    -1.499998
  Max:     1.499997
  Mean:    0.000123
  StdDev:  0.866025
```

### Per-Channel Statistics
```
Channel 0: mean=0.0012, std=0.8660
Channel 1: mean=-0.0034, std=0.8655
...
```

### Performance Metrics
```
Latency Statistics:
  Average: 1.234 ms
  Min:     1.100 ms
  Max:     1.450 ms
  StdDev:  0.089 ms

Throughput:
  Average FPS: 810.4
  Max FPS:     909.1
```

## Model Requirements

- **Input shape**: `[1, 32, 5120, 1]`
- **Output shape**: `[1, 32, 5120, 1]`
- **Model format**: Compiled Axelera model (model.json)
- **Default path**: `../models/compiled_multichannel_unet/compiled_model/model.json`

## Dependencies

- Voyager SDK v1.4+
- AxInferenceNet library
- AxRuntime library
- C++17 compiler

## Notes

- Uses 4 AIPU cores by default
- Input data: uniform random in [-1.5, 1.5]
- All statistics computed on FP32 data
- DMA buffers enabled for optimal performance
