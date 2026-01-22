# Phase 3: Multi-Channel IQ Signal Processing (16 Channels)

## Overview

Phase 3 extends the U-Net architecture to process **16 IQ channels simultaneously** (32 total channels: 16 I + 16 Q). This implementation provides:

- **Multi-channel U-Net model** with 32 input/output channels
- **Live visualization** with interactive channel selection
- **Real-time comparison** between PyTorch (FP32) and Axelera NPU (INT8)
- **Complete training pipeline** from scratch to deployment

## Architecture

### Model Specifications
- **Input shape**: `(1, 32, 5120, 1)` - 16 I channels + 16 Q channels
- **Output shape**: `(1, 32, 5120, 1)` - 16 I channels + 16 Q channels
- **Base filters**: 64 (increased from 32 in Phase 2)
- **Architecture**: U-Net with 3 encoder/decoder stages
- **Task**: Identity mapping (autoencoder)

### Channel Organization
```
Channels 0-15:  I channels (In-phase)
Channels 16-31: Q channels (Quadrature)
```

## Quick Start

### 1. Train the Model

```bash
python3 train_unet.py
```

**Training Configuration:**
- Dataset: 10,000 samples
- Batch size: 16
- Learning rate: 0.001
- Early stopping: 5 epochs patience
- Data range: [-1.5, +1.5]

**Output:**
- `models/multichannel_unet_trained.pth` - Trained PyTorch model
- `results/training_history.png` - Training curves

### 2. Export to ONNX

```bash
python3 export_onnx.py
```

**Output:**
- `models/multichannel_unet_model.onnx` - ONNX model for NPU compilation

### 3. Compile for Axelera NPU

```bash
bash compile_for_npu.sh
```

**Output:**
- `models/compiled_multichannel_unet/compiled_model/model.json` - NPU model
- `models/compiled_multichannel_unet/compiled_model/manifest.json` - Quantization parameters

### 4. Run Live Visualization

```bash
python3 demo_live_multichannel.py
```

**Features:**
- Real-time PyTorch vs NPU comparison
- Interactive channel selection (0-15)
- Live performance metrics
- Correlation and error statistics

## Live Demo Controls

### Keyboard Shortcuts
- **+/-**: Zoom in/out
- **←/→**: Pan left/right
- **↑/↓**: Change channel
- **q**: Quit

### Interactive Controls
- **Channel Slider**: Select channel 0-15
- **Real-time Stats**: FPS, correlation, errors

## File Structure

```
phase_three/
├── model.py                          # Multi-channel U-Net model (32 channels)
├── train_unet.py                     # Training script
├── export_onnx.py                    # ONNX export
├── compile_for_npu.sh                # NPU compilation script
├── demo_live_multichannel.py         # Live visualization demo
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── models/                           # Model artifacts
│   ├── multichannel_unet_trained.pth
│   ├── multichannel_unet_model.onnx
│   └── compiled_multichannel_unet/
└── results/                          # Training results
    └── training_history.png
```

## Key Differences from Phase 2

### Model Changes
- **Input channels**: 2 → 32 (16x increase)
- **Base filters**: 32 → 64 (2x increase)
- **Parameters**: ~500K → ~2.5M (5x increase)
- **Batch size**: 32 → 16 (reduced for memory)

### Visualization Changes
- **Channel selector**: Interactive slider for 16 channels
- **Keyboard navigation**: Arrow keys to switch channels
- **Multi-channel stats**: Per-channel performance tracking
- **Enhanced UI**: Channel-specific titles and indicators

## Performance Expectations

### Training
- **Time**: ~5-10 minutes for 100 epochs (with early stopping)
- **Memory**: ~4-6 GB GPU/RAM
- **Convergence**: Typically within 20-30 epochs

### Inference
- **PyTorch (CPU)**: ~50-100 FPS
- **Axelera NPU**: ~1000-2000 FPS
- **Expected speedup**: 10-20x

### Accuracy
- **Correlation**: >0.999 (very high similarity)
- **Mean error**: <0.01 (low quantization error)
- **Max error**: <0.1 (acceptable range)

## Technical Details

### Quantization Pipeline

**Input Preprocessing:**
1. Rescale: [-1.5, +1.5] → [0, 1]
2. Quantize: FP32 → INT8 using manifest parameters
3. Reshape: NCHW → NHWC
4. Pad: Apply padding from manifest

**Output Postprocessing:**
1. Unpad: Remove padding
2. Dequantize: INT8 → FP32
3. Reshape: NHWC → NCHW
4. Rescale: [0, 1] → [-1.5, +1.5]

### Model Architecture

```
Encoder:
  conv1 (32→64) → pool1 (5120→2560)
  conv2 (64→128) → pool2 (2560→1280)
  conv3 (128→256) → pool3 (1280→640)

Bottleneck:
  conv_middle (256→256)

Decoder:
  up3 + conv4 (256→256) → 1280
  up2 + conv5 (256→128) → 2560
  up1 + conv6 (128→64) → 5120
  
Output:
  output_layer (64→32)
```

## Troubleshooting

### Training Issues
- **Out of memory**: Reduce batch size to 8 or 4
- **Slow convergence**: Increase learning rate to 0.002
- **Overfitting**: Increase dropout rate to 0.2

### Compilation Issues
- **ONNX export fails**: Check PyTorch model loads correctly
- **NPU compilation fails**: Verify input shape is `1,32,5120,1`
- **Manifest missing**: Ensure compilation completed successfully

### Visualization Issues
- **AxRuntime not found**: Activate Voyager SDK environment
- **Slow FPS**: Reduce visualization update rate
- **Channel out of range**: Use slider or arrow keys (0-15)

## Next Steps

### Optimization
1. **Quantization-aware training**: Train with INT8 simulation
2. **Model pruning**: Reduce parameters for faster inference
3. **Batch processing**: Process multiple samples simultaneously

### Extensions
1. **More channels**: Extend to 32 or 64 IQ channels
2. **Different tasks**: Classification, detection, filtering
3. **Real data**: Replace synthetic data with actual IQ samples

## References

- Phase 2: Single-channel IQ processing (2 channels)
- Axelera SDK: Voyager v1.4+
- U-Net: Original paper by Ronneberger et al.

## Support

For issues or questions:
1. Check Phase 2 implementation for reference
2. Review Axelera SDK documentation
3. Verify all dependencies are installed
4. Check model shapes and data formats

---

**Phase 3 Status**: ✓ Complete and ready for deployment
