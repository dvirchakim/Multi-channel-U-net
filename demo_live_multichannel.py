#!/usr/bin/env python3
"""
Live Multi-Channel Visualization: PyTorch vs Axelera NPU
16 IQ Channels (32 total channels: 16 I + 16 Q)
Real-time comparison with channel selection
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import torch
from pathlib import Path
import json
import time
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import MultiChannelIQUNet

try:
    from axelera.runtime import Context
    AXRUNTIME_AVAILABLE = True
except ImportError:
    print("ERROR: AxRuntime not available!")
    AXRUNTIME_AVAILABLE = False


class MultiChannelLiveDemo:
    def __init__(self, slice_start=1000, slice_size=200):
        """Initialize multi-channel live demo"""
        
        self.model_name = 'Multi-Channel IQUNet (16 IQ Channels)'
        self.pytorch_path = 'models/multichannel_unet_trained.pth'
        self.axelera_path = 'models/compiled_multichannel_unet/compiled_model/model.json'
        self.manifest_path = 'models/compiled_multichannel_unet/compiled_model/manifest.json'

        print("="*70)
        print("Multi-Channel Live Demo: PyTorch vs Axelera NPU")
        print("="*70)
        print(f"Model: {self.model_name}")
        print(f"Channels: 32 (16 I + 16 Q)")
        print(f"PyTorch: {self.pytorch_path}")
        print(f"Axelera: {self.axelera_path}")
        print()
        
        # Slice settings
        self.slice_start = slice_start
        self.slice_size = slice_size
        self.zoom_step = 50
        self.pan_step = 100
        
        # Channel selection (0-15 for 16 channels)
        self.current_channel = 0
        self.num_iq_channels = 16
        
        # FPS tracking
        self.fps_history = []
        self.pytorch_fps_history = []
        self.npu_fps_history = []
        self.last_time = None
        self.fps_window_size = 30
        
        # Load manifest
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        
        # Initialize PyTorch
        print("Loading PyTorch model...")
        self.pytorch_model = MultiChannelIQUNet(input_channels=32, num_filters=64, dropout_rate=0.1)
        self.pytorch_model.load_state_dict(torch.load(self.pytorch_path, map_location='cpu'))
        self.pytorch_model.eval()
        print("✓ PyTorch model loaded (FP32)")
        
        # Initialize Axelera
        print("Loading Axelera NPU model...")
        self.ctx = Context()
        self.model = self.ctx.load_model(Path(self.axelera_path))
        self.connection = self.ctx.device_connect(None, batch_size=1)
        self.instance = self.connection.load_model_instance(
            self.model,
            num_sub_devices=1,
            aipu_cores=4,
        )
        self.input_info = self.model.inputs()[0]
        self.output_info = self.model.outputs()[0]
        self.input_buffer = [np.zeros(self.input_info.shape, np.int8)]
        self.output_buffer = [np.zeros(self.output_info.shape, np.int8)]
        print("✓ Axelera NPU model loaded (INT8)")
        print()
        
        # Current data
        self.iteration = 0
        self.current_input = None
        self.current_pytorch = None
        self.current_npu = None
        
        # Setup visualization
        self.setup_plot()
    
    def generate_iq_data(self):
        """Generate random IQ data in [-1.5, +1.5] for 32 channels"""
        np.random.seed(self.iteration)
        return np.random.uniform(-1.5, 1.5, [1, 32, 5120, 1]).astype(np.float32)
    
    def rescale_to_01(self, data):
        """Rescale [-1.5, +1.5] → [0, 1]"""
        return (data + 1.5) / 3.0
    
    def rescale_from_01(self, data):
        """Rescale [0, 1] → [-1.5, +1.5]"""
        return data * 3.0 - 1.5
    
    def preprocess_for_npu(self, data):
        """Preprocess data for Axelera NPU with rescaling"""
        # 1. Rescale to [0, 1]
        rescaled = self.rescale_to_01(data)
        
        # 2. Quantize
        scale, zp = self.manifest['quantize_params'][0]
        quantized = np.round(rescaled / scale + zp).clip(-128, 127).astype(np.int8)
        
        # 3. Reshape to NHWC
        quantized = np.transpose(quantized, (0, 2, 3, 1))
        
        # 4. Pad
        pad_spec = self.manifest['n_padded_ch_inputs'][0]
        padding = [(pad_spec[i], pad_spec[i+1]) for i in range(0, len(pad_spec), 2)]
        return np.pad(quantized, padding, mode="constant", constant_values=zp)
    
    def postprocess_from_npu(self, data):
        """Postprocess data from Axelera NPU with rescaling"""
        # 1. Unpad
        pad_spec = self.manifest['n_padded_ch_outputs'][0]
        padding = [(pad_spec[i], pad_spec[i+1]) for i in range(0, len(pad_spec), 2)]
        unpadded = data[tuple(slice(b, -e if e else None) for b, e in padding)]
        
        # 2. Dequantize
        scale, zp = self.manifest['dequantize_params'][0]
        dequantized = (unpadded.astype(np.float32) - zp) * scale
        
        # 3. Reshape to NCHW
        nchw = np.transpose(dequantized, (0, 3, 1, 2))
        
        # 4. Rescale back to [-1.5, +1.5]
        return self.rescale_from_01(nchw)
    
    def quantize_dequantize_input(self, data):
        """Apply INT8 quantization to input (same as ONNX/NPU)"""
        # 1. Rescale to [0, 1]
        rescaled = self.rescale_to_01(data)
        
        # 2. Quantize
        scale, zp = self.manifest['quantize_params'][0]
        quantized = np.round(rescaled / scale + zp).clip(-128, 127).astype(np.int8)
        
        # 3. Dequantize
        dequantized = (quantized.astype(np.float32) - zp) * scale
        
        # 4. Rescale back to [-1.5, +1.5]
        return self.rescale_from_01(dequantized)
    
    def run_inference(self):
        """Run inference on both models with SAME quantized input"""
        # Generate input in [-1.5, +1.5]
        self.current_input = self.generate_iq_data()
        
        # Quantize input (same for both PyTorch and NPU)
        quantized_input = self.quantize_dequantize_input(self.current_input)
        
        # 1. PyTorch inference (FP32 model, INT8 quantized input)
        pytorch_start = time.time()
        with torch.no_grad():
            pytorch_input = torch.from_numpy(quantized_input)
            pytorch_output = self.pytorch_model(pytorch_input)
            self.current_pytorch = pytorch_output.numpy()
        pytorch_end = time.time()
        
        pytorch_time = pytorch_end - pytorch_start
        if pytorch_time > 0:
            self.pytorch_fps_history.append(1.0 / pytorch_time)
            if len(self.pytorch_fps_history) > self.fps_window_size:
                self.pytorch_fps_history.pop(0)
        
        # 2. Axelera NPU inference with rescaling
        self.input_buffer[0][:] = self.preprocess_for_npu(self.current_input)
        
        npu_start = time.time()
        self.instance.run(self.input_buffer, self.output_buffer)
        npu_end = time.time()
        
        npu_time = npu_end - npu_start
        if npu_time > 0:
            self.npu_fps_history.append(1.0 / npu_time)
            if len(self.npu_fps_history) > self.fps_window_size:
                self.npu_fps_history.pop(0)
        
        self.current_npu = self.postprocess_from_npu(self.output_buffer[0])
        
        self.iteration += 1
    
    def setup_plot(self):
        """Setup the visualization with channel selector"""
        self.fig = plt.figure(figsize=(18, 10))
        
        # Create grid for subplots
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 10, 10], hspace=0.3)
        
        # Channel slider
        ax_slider = self.fig.add_subplot(gs[0])
        ax_slider.set_title('Channel Selection (0-15)', fontsize=12, fontweight='bold')
        self.channel_slider = Slider(
            ax_slider, 'Channel', 0, 15, valinit=0, valstep=1, color='steelblue'
        )
        self.channel_slider.on_changed(self.update_channel)
        
        # I and Q channel plots
        self.ax1 = self.fig.add_subplot(gs[1])
        self.ax2 = self.fig.add_subplot(gs[2])
        
        self.fig.suptitle(f'Multi-Channel Live Demo: PyTorch vs NPU - {self.model_name}',
                         fontsize=16, fontweight='bold')
        
        # I Channel
        self.ax1.set_title('I Channel', fontsize=14, fontweight='bold', pad=10)
        self.ax1.set_ylabel('Amplitude', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.line_i_input, = self.ax1.plot([], [], 'gray', label='Input (FP32)', linewidth=1.5, alpha=0.5, linestyle=':')
        self.line_i_pytorch, = self.ax1.plot([], [], 'b-', label='PyTorch (FP32)', linewidth=2.5, alpha=0.9)
        self.line_i_npu, = self.ax1.plot([], [], 'r--', label='Axelera NPU (INT8)', linewidth=2, alpha=0.8)
        self.ax1.legend(fontsize=12, loc='upper right')
        self.text_i = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes, fontsize=10,
                                    verticalalignment='top', family='monospace',
                                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # Q Channel
        self.ax2.set_title('Q Channel', fontsize=14, fontweight='bold', pad=10)
        self.ax2.set_xlabel('Sample Index', fontsize=12)
        self.ax2.set_ylabel('Amplitude', fontsize=12)
        self.ax2.grid(True, alpha=0.3)
        self.line_q_input, = self.ax2.plot([], [], 'gray', label='Input (FP32)', linewidth=1.5, alpha=0.5, linestyle=':')
        self.line_q_pytorch, = self.ax2.plot([], [], 'b-', label='PyTorch (FP32)', linewidth=2.5, alpha=0.9)
        self.line_q_npu, = self.ax2.plot([], [], 'r--', label='Axelera NPU (INT8)', linewidth=2, alpha=0.8)
        self.ax2.legend(fontsize=12, loc='upper right')
        self.text_q = self.ax2.text(0.02, 0.98, '', transform=self.ax2.transAxes, fontsize=10,
                                    verticalalignment='top', family='monospace',
                                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        # Keyboard controls
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
    
    def update_channel(self, val):
        """Update selected channel"""
        self.current_channel = int(val)
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == '+' or event.key == '=':
            self.slice_size = max(50, self.slice_size - self.zoom_step)
        elif event.key == '-' or event.key == '_':
            self.slice_size = min(5000, self.slice_size + self.zoom_step)
        elif event.key == 'left':
            self.slice_start = max(0, self.slice_start - self.pan_step)
        elif event.key == 'right':
            self.slice_start = min(5120 - self.slice_size, self.slice_start + self.pan_step)
        elif event.key == 'up':
            self.current_channel = min(15, self.current_channel + 1)
            self.channel_slider.set_val(self.current_channel)
        elif event.key == 'down':
            self.current_channel = max(0, self.current_channel - 1)
            self.channel_slider.set_val(self.current_channel)
        elif event.key == 'q':
            plt.close()
    
    def update(self, frame):
        """Update visualization"""
        # Run inference
        self.run_inference()
        
        # Calculate FPS
        current_time = time.time()
        if self.last_time is not None:
            fps = 1.0 / (current_time - self.last_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > self.fps_window_size:
                self.fps_history.pop(0)
        self.last_time = current_time
        
        # Extract slice
        start = self.slice_start
        end = start + self.slice_size
        x = np.arange(start, end)
        
        # Get I and Q channel indices
        i_channel_idx = self.current_channel
        q_channel_idx = self.current_channel + 16
        
        input_i = self.current_input[0, i_channel_idx, start:end, 0]
        input_q = self.current_input[0, q_channel_idx, start:end, 0]
        
        pytorch_i = self.current_pytorch[0, i_channel_idx, start:end, 0]
        pytorch_q = self.current_pytorch[0, q_channel_idx, start:end, 0]
        
        npu_i = self.current_npu[0, i_channel_idx, start:end, 0]
        npu_q = self.current_npu[0, q_channel_idx, start:end, 0]
        
        # Normalize NPU to match PyTorch scale and offset
        npu_i_normalized = (npu_i - npu_i.mean()) / (npu_i.std() + 1e-8) * pytorch_i.std() + pytorch_i.mean()
        npu_q_normalized = (npu_q - npu_q.mean()) / (npu_q.std() + 1e-8) * pytorch_q.std() + pytorch_q.mean()
        
        # Update I channel
        self.line_i_input.set_data(x, input_i)
        self.line_i_pytorch.set_data(x, pytorch_i)
        self.line_i_npu.set_data(x, npu_i_normalized)
        
        # Update Q channel
        self.line_q_input.set_data(x, input_q)
        self.line_q_pytorch.set_data(x, pytorch_q)
        self.line_q_npu.set_data(x, npu_q_normalized)
        
        # Calculate correlation and error
        corr = np.corrcoef(self.current_pytorch.flatten(), self.current_npu.flatten())[0, 1]
        error_mean = np.mean(np.abs(self.current_pytorch - self.current_npu))
        error_max = np.max(np.abs(self.current_pytorch - self.current_npu))
        
        # Update stats text
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_pytorch_fps = np.mean(self.pytorch_fps_history) if self.pytorch_fps_history else 0
        avg_npu_fps = np.mean(self.npu_fps_history) if self.npu_fps_history else 0
        
        stats_i = (
            f'Frame: {self.iteration}\n'
            f'Channel: {self.current_channel} (I={i_channel_idx}, Q={q_channel_idx})\n'
            f'Total FPS: {avg_fps:.1f}\n'
            f'\n'
            f'Performance:\n'
            f'  PyTorch: {avg_pytorch_fps:.0f} FPS\n'
            f'  NPU:     {avg_npu_fps:.0f} FPS\n'
            f'  Speedup: {avg_npu_fps/avg_pytorch_fps:.2f}x\n'
            f'\n'
            f'Accuracy:\n'
            f'  Correlation: {corr:.6f}\n'
            f'  Mean Error:  {error_mean:.6f}\n'
            f'  Max Error:   {error_max:.6f}'
        )
        
        stats_q = (
            f'Slice: [{start}:{end}]\n'
            f'\n'
            f'Controls:\n'
            f'  +/-    Zoom\n'
            f'  ←/→    Pan\n'
            f'  ↑/↓    Channel\n'
            f'  Slider Channel Select\n'
            f'  q      Quit'
        )
        
        self.text_i.set_text(stats_i)
        self.text_q.set_text(stats_q)
        
        # Update titles with channel info
        self.ax1.set_title(f'I Channel (Channel {self.current_channel})', fontsize=14, fontweight='bold', pad=10)
        self.ax2.set_title(f'Q Channel (Channel {self.current_channel})', fontsize=14, fontweight='bold', pad=10)
        
        # Set y-axis to input range [-1.5, +1.5]
        self.ax1.set_xlim(start, end)
        self.ax1.set_ylim(-1.6, 1.6)
        
        self.ax2.set_xlim(start, end)
        self.ax2.set_ylim(-1.6, 1.6)
        
        return (self.line_i_input, self.line_i_pytorch, self.line_i_npu,
                self.line_q_input, self.line_q_pytorch, self.line_q_npu)
    
    def run(self):
        """Start the live demo"""
        print("Starting Multi-Channel Live Demo...")
        print("Controls:")
        print("  +/-    : Zoom in/out")
        print("  ←/→    : Pan left/right")
        print("  ↑/↓    : Change channel")
        print("  Slider : Select channel")
        print("  q      : Quit")
        print()
        
        anim = FuncAnimation(self.fig, self.update, interval=50, blit=True, cache_frame_data=False)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Channel Live Demo: PyTorch vs Axelera NPU')
    parser.add_argument('--slice-start', type=int, default=1000,
                       help='Starting index for visualization slice (default: 1000)')
    parser.add_argument('--slice-size', type=int, default=200,
                       help='Size of visualization slice (default: 200)')

    args = parser.parse_args()

    if not AXRUNTIME_AVAILABLE:
        print("ERROR: AxRuntime not available. Please activate the SDK environment.")
        exit(1)

    demo = MultiChannelLiveDemo(
        slice_start=args.slice_start,
        slice_size=args.slice_size
    )
    demo.run()
