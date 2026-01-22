# Phase 3 Quick Start Guide

## 🚀 Complete Workflow (5 Steps)

### Step 1: Train the Model (5-10 minutes)
```bash
cd "/home/axelera/dvir/Gilat_Project/simple net/phase_three"
python3 train_unet.py
```
**Expected output**: `models/multichannel_unet_trained.pth`

### Step 2: Export to ONNX (30 seconds)
```bash
python3 export_onnx.py
```
**Expected output**: `models/multichannel_unet_model.onnx`

### Step 3: Compile for NPU (2-5 minutes)
```bash
bash compile_for_npu.sh
```
**Expected output**: `models/compiled_multichannel_unet/compiled_model/`

### Step 4: Run Live Demo
```bash
python3 demo_live_multichannel.py
```

### Step 5: Interact with Live Visualization
- **Slider**: Select channel 0-15
- **↑/↓**: Navigate channels
- **+/-**: Zoom in/out
- **←/→**: Pan signal view
- **q**: Quit

---

## 📊 What You'll See

### Live Visualization Features
1. **Real-time inference** on both PyTorch and NPU
2. **16 IQ channels** (32 total: 16 I + 16 Q)
3. **Performance metrics**: FPS, speedup, correlation
4. **Interactive controls**: Channel selection, zoom, pan

### Expected Performance
- **PyTorch**: ~50-100 FPS
- **NPU**: ~1000-2000 FPS
- **Speedup**: 10-20x
- **Correlation**: >0.999

---

## 🔧 One-Line Commands

### Full Pipeline
```bash
cd "/home/axelera/dvir/Gilat_Project/simple net/phase_three" && \
python3 train_unet.py && \
python3 export_onnx.py && \
bash compile_for_npu.sh && \
python3 demo_live_multichannel.py
```

### Test Model Only
```bash
python3 model.py
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `model.py` | 32-channel U-Net architecture |
| `train_unet.py` | Training script |
| `export_onnx.py` | ONNX export |
| `compile_for_npu.sh` | NPU compilation |
| `demo_live_multichannel.py` | Live visualization |

---

## ⚡ Quick Tips

1. **First time?** Run model test: `python3 model.py`
2. **Skip training?** Use pre-trained model from Phase 2 as reference
3. **Slow training?** Reduce batch size in `train_unet.py`
4. **NPU issues?** Check SDK environment is activated

---

## 🎯 Key Differences from Phase 2

| Feature | Phase 2 | Phase 3 |
|---------|---------|---------|
| Channels | 2 (1 I + 1 Q) | 32 (16 I + 16 Q) |
| Base Filters | 32 | 64 |
| Parameters | ~500K | ~2.5M |
| Batch Size | 32 | 16 |
| Channel Selection | N/A | Interactive slider |

---

**Ready to start?** Run Step 1 above! 🚀
