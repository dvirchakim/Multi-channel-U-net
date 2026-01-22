# Hardware Specifications - AXE-BME20P1AJ04A02

## Card Information

**Model**: AXE-BME20P1AJ04A02  
**Type**: Axelera Metis PCIe Accelerator Card  
**Form Factor**: PCIe Gen4 x16  
**Target**: Edge AI / Data Center Inference

## Performance Specifications

### Compute Capacity
- **INT8 Performance**: 214 TOPS
- **Architecture**: Metis AIPU (AI Processing Unit)
- **Cores**: Multiple AIPU clusters
- **Precision**: INT8 optimized, FP16 support

### Memory Hierarchy

#### On-Chip SRAM
- **Size**: ~10-20 MB per AIPU cluster
- **Bandwidth**: 500+ GB/s (internal)
- **Purpose**: Activation storage during inference
- **Latency**: <1 ns

#### Device DRAM
- **Size**: 4-8 GB LPDDR4/DDR4
- **Bandwidth**: 50-100 GB/s
- **Purpose**: Model weights, I/O buffers
- **Latency**: ~100 ns

#### Host Interface
- **PCIe**: Gen4 x16
- **Bandwidth**: ~32 GB/s bidirectional
- **Latency**: ~1-2 μs

## Channel Scaling Analysis for U-Net

### Memory Constraints

For input shape `[1, C, 5120, 1]`:

| Channels | Input Size (INT8) | Padded Size | Weights | Total DRAM | Status |
|----------|-------------------|-------------|---------|------------|--------|
| 2 | 10 KB | 13 KB | ~5 MB | ~5 MB | ✅ Tested |
| 32 | 160 KB | 524 KB | ~10 MB | ~11 MB | ✅ Tested |
| 64 | 320 KB | ~1 MB | ~15 MB | ~16 MB | ✅ Recommended |
| 128 | 640 KB | ~2 MB | ~25 MB | ~27 MB | ✅ Safe |
| 256 | 1.3 MB | ~4 MB | ~50 MB | ~54 MB | ✅ Good |
| 512 | 2.6 MB | ~8 MB | ~100 MB | ~108 MB | ✅ Likely works |
| 1024 | 5.1 MB | ~16 MB | ~200 MB | ~216 MB | ⚠️ Test needed |
| 2048 | 10.2 MB | ~32 MB | ~400 MB | ~432 MB | ⚠️ May work |
| 4096 | 20.5 MB | ~64 MB | ~800 MB | ~864 MB | ❌ Compiler limits |

### Performance Estimates

Based on 214 TOPS compute capacity:

| Channels | Compute (GOPS) | Expected FPS | Throughput | Notes |
|----------|----------------|--------------|------------|-------|
| 2 | ~5 | 3000-4000 | High | Tested: 3921 FPS |
| 32 | ~50 | 700-900 | High | Tested: 736 FPS |
| 64 | ~100 | 400-600 | Good | 2x current |
| 128 | ~200 | 200-400 | Good | 4x current |
| 256 | ~400 | 100-250 | Medium | 8x current |
| 512 | ~800 | 50-150 | Medium | 16x current |
| 1024 | ~1600 | 25-100 | Low | 32x current |
| 2048 | ~3200 | 10-50 | Low | 64x current |

### Bottleneck Analysis

#### 1. **On-Chip SRAM** (10-20 MB)
- **Stores**: Intermediate activations
- **U-Net peak**: ~2-3 MB (independent of input channels)
- **Verdict**: ✅ Not a bottleneck for channel scaling

#### 2. **Device DRAM** (4-8 GB)
- **Stores**: Model weights, I/O buffers
- **Usage**: ~5 MB base + 0.2 MB per channel
- **Verdict**: ✅ Can handle 1000+ channels easily

#### 3. **PCIe Bandwidth** (32 GB/s)
- **Transfer time**: 
  - 32 channels: 0.16 MB → 5 μs
  - 1024 channels: 5.1 MB → 160 μs
- **Verdict**: ⚠️ May become bottleneck at 1000+ channels

#### 4. **Compiler Constraints**
- **Max tensor size**: Implementation dependent
- **Compilation time**: Increases with model size
- **Verdict**: ⚠️ Likely limits at 2000-4000 channels

## Recommended Channel Configurations

### **Production Use**
- **64-256 channels**: Optimal balance
- **Performance**: 100-600 FPS
- **Memory**: <100 MB
- **Reliability**: High

### **High Channel Count**
- **512-1024 channels**: Feasible
- **Performance**: 25-150 FPS
- **Memory**: 100-250 MB
- **Reliability**: Medium (needs testing)

### **Extreme Testing**
- **2048+ channels**: Experimental
- **Performance**: 10-50 FPS
- **Memory**: 400+ MB
- **Reliability**: Low (compiler may fail)

## Practical Limits

### **Hardware Limit**: ~4000 channels
- Constrained by 4-8 GB DRAM
- Assumes 1 GB for model, 1 GB for I/O

### **Compiler Limit**: ~2000 channels
- Typical framework constraints
- Compilation time becomes impractical

### **Performance Limit**: ~1000 channels
- Below 25 FPS not useful for most applications
- PCIe transfer overhead dominates

### **Recommended Maximum**: 512 channels
- Good performance (50-150 FPS)
- Reliable compilation
- Reasonable memory usage
- 16x increase from current 32 channels

## Optimization Strategies

### For More Channels
1. **Reduce sequence length**: 5120 → 2560 samples
2. **Use model pruning**: Reduce filter counts
3. **Batch processing**: Process multiple samples
4. **Channel grouping**: Process in 32-channel groups

### For Better Performance
1. **Increase batch size**: Process multiple samples together
2. **Pipeline stages**: Overlap preprocessing/inference/postprocessing
3. **Use C++ implementation**: 3-5x faster than Python
4. **Optimize data transfers**: Minimize PCIe overhead

## Testing Recommendations

### Phase 1: Validate Compilation
Test channels: 64, 128, 256, 512, 1024
- Export ONNX models
- Attempt compilation
- Record success/failure

### Phase 2: Measure Performance
For successful compilations:
- Run inference benchmarks
- Measure FPS and latency
- Check accuracy vs FP32

### Phase 3: Optimize
For target channel count:
- Tune batch size
- Optimize preprocessing
- Implement C++ pipeline

## Summary

**Your AXE-BME20P1AJ04A02 card can handle:**

✅ **Confidently**: Up to 512 channels (50-150 FPS)  
⚠️ **Likely**: Up to 1024 channels (25-100 FPS)  
❓ **Maybe**: Up to 2048 channels (10-50 FPS)  
❌ **Unlikely**: Above 4096 channels (compiler limits)

**Recommended next step**: Test 128 channels (4x current), then 256 channels (8x current).

---

**Hardware**: AXE-BME20P1AJ04A02 (214 TOPS)  
**Current**: 32 channels @ 736 FPS  
**Target**: 128-512 channels @ 50-400 FPS  
**Date**: January 2026
