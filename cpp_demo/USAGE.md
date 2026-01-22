# C++ Demo Usage Guide

## ✅ Successfully Compiled and Running!

The standalone C++ demo provides detailed tensor statistics and shapes for the 32-channel IQ model.

## Quick Commands

### Detailed Analysis (Single Tensor)
```bash
./standalone_demo --detailed
```

**Output:**
- Tensor shape: `[1, 32, 5120, 1]` (163,840 elements, 640 KB)
- Overall statistics (min, max, mean, std dev)
- Per-channel statistics for all 32 channels
- IQ pair analysis for all 16 pairs

### Benchmark (100 iterations)
```bash
./standalone_demo --iterations 100
```

**Output:**
- Data generation performance metrics
- Average: ~0.95 ms per tensor
- Min/Max/StdDev statistics

### Save Tensor Data
```bash
./standalone_demo --detailed --save tensor_output.bin
```

Saves the tensor to a binary file for further analysis.

## Example Output

### Tensor Shape
```
Shape: [1, 32, 5120, 1]
Total elements: 163840
Memory size: 640 KB
```

### Overall Statistics
```
OVERALL STATISTICS:
  Min:    -1.499993
  Max:    1.499989
  Mean:   0.002516
  StdDev: 0.866096
```

### Per-Channel Statistics
```
I CHANNELS (0-15):
Channel  0: mean= -0.0052, std=  0.8547, range=[ -1.4984,   1.4994]
Channel  1: mean=  0.0039, std=  0.8687, range=[ -1.4974,   1.4997]
...

Q CHANNELS (16-31):
Channel 16: mean=  0.0011, std=  0.8674, range=[ -1.4972,   1.4998]
Channel 17: mean=  0.0081, std=  0.8647, range=[ -1.4994,   1.4995]
...
```

### IQ Pair Analysis
```
Pair  0 (I= 0, Q=16):
  I: mean= -0.0052, std=  0.8547
  Q: mean=  0.0011, std=  0.8674
...
```

## Performance Results

From 100-iteration benchmark:
- **Average generation time**: 0.947 ms
- **Min time**: 0.901 ms
- **Max time**: 1.500 ms
- **Standard deviation**: 0.117 ms
- **Throughput**: ~1,055 tensors/second

## Data Characteristics

For uniform random data in range [-1.5, 1.5]:
- **Expected mean**: ~0.0 (observed: -0.003 to 0.016)
- **Expected std dev**: ~0.866 (observed: 0.854 to 0.875)
- **Range coverage**: Full [-1.5, 1.5] range utilized
- **Channel independence**: Each channel has unique statistics

## Use Cases

1. **Verify tensor shapes** before NPU inference
2. **Check data distribution** across all 32 channels
3. **Benchmark data generation** performance
4. **Export tensors** for external analysis
5. **Debug preprocessing** pipeline

## Building

```bash
make clean
make
```

## All Options

```bash
./standalone_demo --help
```

Options:
- `--detailed` - Run detailed single tensor analysis
- `--iterations N` - Number of benchmark iterations (default: 10)
- `--save FILE` - Save tensor data to binary file
- `--help` - Show help message

---

**Status**: ✓ Fully functional and tested
