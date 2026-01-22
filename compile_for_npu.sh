#!/bin/bash
# Compile Multi-Channel IQUNet ONNX model for Axelera NPU
# 32 channels (16 I + 16 Q)

set -e

echo "======================================================================"
echo "Compiling Multi-Channel IQUNet for Axelera NPU"
echo "======================================================================"
echo ""

# Check if ONNX model exists
ONNX_MODEL="models/multichannel_unet_model.onnx"
if [ ! -f "$ONNX_MODEL" ]; then
    echo "ERROR: ONNX model not found: $ONNX_MODEL"
    echo "Please run: python3 export_onnx.py"
    exit 1
fi

echo "ONNX model: $ONNX_MODEL"
echo "Input shape: 1,32,5120,1 (32 channels = 16 I + 16 Q)"
echo "Output directory: models/compiled_multichannel_unet"
echo ""

# Compile using Axelera compiler
echo "Running Axelera compiler..."
echo ""

compile \
    --input "$ONNX_MODEL" \
    --output models/compiled_multichannel_unet \
    --overwrite \
    --input-shape 1,32,5120,1 \
    --config /home/axelera/dvir/Gilat_Project/workdir_0/conf.json

echo ""
echo "======================================================================"
echo "✓ Compilation complete!"
echo "======================================================================"
echo ""
echo "Compiled model artifacts:"
echo "  - models/compiled_multichannel_unet/compiled_model/model.json"
echo "  - models/compiled_multichannel_unet/compiled_model/manifest.json"
echo ""
echo "Next step:"
echo "  Run live visualization:"
echo "  python3 demo_live_multichannel.py"
echo ""
