#!/usr/bin/env python3
"""
Export trained Multi-Channel IQUNet model to ONNX format
"""

import torch
from pathlib import Path
from model import MultiChannelIQUNet


def export_to_onnx(
    model_path='models/multichannel_unet_trained.pth',
    onnx_path='models/multichannel_unet_model.onnx'
):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model_path: Path to trained PyTorch model (.pth)
        onnx_path: Path to save ONNX model (.onnx)
    """
    print("="*70)
    print("Exporting Multi-Channel IQUNet to ONNX")
    print("="*70)
    print(f"PyTorch model: {model_path}")
    print(f"ONNX output: {onnx_path}")
    print()
    
    # Load model
    print("Loading PyTorch model...")
    model = MultiChannelIQUNet(input_channels=32, num_filters=64, dropout_rate=0.1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("✓ Model loaded")
    
    # Create dummy input (32 channels = 16 I + 16 Q)
    dummy_input = torch.randn(1, 32, 5120, 1)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Input channels: 32 (16 I + 16 Q)")
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print()
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX model exported to: {onnx_path}")
    print()
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
    except ImportError:
        print("⚠ ONNX package not available for verification (optional)")
    except Exception as e:
        print(f"⚠ ONNX verification warning: {e}")
    
    print()
    print("="*70)
    print("✓ Export complete!")
    print("="*70)
    print()
    print("Next step:")
    print("  Compile for Axelera NPU:")
    print("  bash compile_for_npu.sh")
    print()


def main():
    # Ensure models directory exists
    Path('models').mkdir(exist_ok=True)
    
    # Export
    export_to_onnx(
        model_path='models/multichannel_unet_trained.pth',
        onnx_path='models/multichannel_unet_model.onnx'
    )


if __name__ == "__main__":
    main()
