#!/usr/bin/env python3
"""
Multi-Channel IQUNet Model for 16-Channel IQ Signal Processing
Architecture: U-Net with 1D convolutions
Input: (N, 32, 5120, 1) - 16 I channels + 16 Q channels
Output: (N, 32, 5120, 1) - 16 I channels + 16 Q channels
"""

import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_height=3, dropout_rate=0.1):
    """Conv2d-BN-ReLU-Dropout-Conv2d-BN-ReLU block operating on (N, C, L, 1)."""
    padding = (kernel_height // 2, 0)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height, 1), padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_height, 1), padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class MultiChannelIQUNet(nn.Module):
    """
    U-Net for Multi-Channel IQ Signal Processing
    
    Architecture:
    - Encoder: 3 downsampling stages (MaxPool1d)
    - Bottleneck: Middle conv block
    - Decoder: 3 upsampling stages (ConvTranspose1d) with skip connections
    - Output: 1x1 Conv to restore 32 channels
    
    Input shape: (N, 32, 5120, 1) - 16 I channels + 16 Q channels
    Output shape: (N, 32, 5120, 1) - 16 I channels + 16 Q channels
    """
    
    def __init__(self, input_channels=32, num_filters=64, dropout_rate=0.1):
        super(MultiChannelIQUNet, self).__init__()
        
        # Downsample path (Encoder)
        self.conv1 = conv_block(input_channels, num_filters, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv2 = conv_block(num_filters, num_filters * 2, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv3 = conv_block(num_filters * 2, num_filters * 4, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Bottleneck (Middle layer)
        self.conv_middle = conv_block(num_filters * 4, num_filters * 4, dropout_rate=dropout_rate)
        
        # Upsample path (Decoder)
        self.up3 = nn.ConvTranspose2d(num_filters * 4, num_filters * 4, kernel_size=(2, 1), stride=(2, 1))
        self.conv4 = conv_block(num_filters * 8, num_filters * 4, dropout_rate=dropout_rate)
        
        self.up2 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=(2, 1), stride=(2, 1))
        self.conv5 = conv_block(num_filters * 4, num_filters * 2, dropout_rate=dropout_rate)
        
        self.up1 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=(2, 1), stride=(2, 1))
        self.conv6 = conv_block(num_filters * 2, num_filters, dropout_rate=dropout_rate)
        
        # Output Layer
        self.output_layer = nn.Conv2d(num_filters, input_channels, kernel_size=(1, 1))

    def forward(self, x):
        """
        Forward pass with automatic shape handling
        
        Args:
            x: Input tensor of shape (N, 32, 5120, 1)
        
        Returns:
            Output tensor of shape (N, 32, 5120, 1)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (N, C, L, 1), got shape {x.shape}")
        
        # Encoder (Downsample path)
        conv1 = self.conv1(x)          # (N, 64, 5120, 1)
        pool1 = self.pool1(conv1)      # (N, 64, 2560, 1)
        
        conv2 = self.conv2(pool1)      # (N, 128, 2560, 1)
        pool2 = self.pool2(conv2)      # (N, 128, 1280, 1)
        
        conv3 = self.conv3(pool2)      # (N, 256, 1280, 1)
        pool3 = self.pool3(conv3)      # (N, 256, 640, 1)
        
        # Bottleneck
        conv_middle = self.conv_middle(pool3)  # (N, 256, 640, 1)
        
        # Decoder (Upsample path with skip connections)
        up3 = self.up3(conv_middle)    # (N, 256, 1280, 1)
        merge3 = torch.cat([conv3, up3], dim=1)  # (N, 512, 1280, 1)
        conv4 = self.conv4(merge3)     # (N, 256, 1280, 1)
        
        up2 = self.up2(conv4)          # (N, 128, 2560, 1)
        merge2 = torch.cat([conv2, up2], dim=1)  # (N, 256, 2560, 1)
        conv5 = self.conv5(merge2)     # (N, 128, 2560, 1)
        
        up1 = self.up1(conv5)          # (N, 64, 5120, 1)
        merge1 = torch.cat([conv1, up1], dim=1)  # (N, 128, 5120, 1)
        conv6 = self.conv6(merge1)     # (N, 64, 5120, 1)
        
        # Output Layer
        outputs = self.output_layer(conv6)  # (N, 32, 5120, 1)
        
        return outputs


def test_model():
    """Test the model with dummy input"""
    print("="*70)
    print("Testing Multi-Channel IQUNet Model")
    print("="*70)
    
    model = MultiChannelIQUNet(input_channels=32, num_filters=64)
    model.eval()
    
    # Test input
    test_input = torch.randn(1, 32, 5120, 1)
    print(f"Input shape: {test_input.shape}")
    print(f"Input channels: 32 (16 I + 16 Q)")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    
    # Verify shape
    assert output.shape == test_input.shape, f"Shape mismatch! Expected {test_input.shape}, got {output.shape}"
    print("✓ Shape test passed!")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    print()
    return model


if __name__ == "__main__":
    test_model()
