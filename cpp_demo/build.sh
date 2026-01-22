#!/bin/bash
# Build script for Multi-Channel Demo

set -e

echo "======================================================================"
echo "Building Multi-Channel Demo (C++)"
echo "======================================================================"
echo ""

# Check if Voyager SDK is available
VOYAGER_SDK="${HOME}/dvir/voyager-sdk"
if [ ! -d "$VOYAGER_SDK" ]; then
    echo "ERROR: Voyager SDK not found at $VOYAGER_SDK"
    exit 1
fi

echo "Voyager SDK: $VOYAGER_SDK"
echo ""

# Create build directory
mkdir -p build
cd build

# Run CMake
echo "Running CMake..."
cmake .. -DVOYAGER_SDK_PATH="$VOYAGER_SDK"

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "======================================================================"
echo "✓ Build complete!"
echo "======================================================================"
echo ""
echo "Executable: build/multichannel_demo"
echo ""
echo "Usage:"
echo "  ./build/multichannel_demo                    # Run benchmark (10 iterations)"
echo "  ./build/multichannel_demo --iterations 100   # Run 100 iterations"
echo "  ./build/multichannel_demo --detailed         # Detailed single inference"
echo "  ./build/multichannel_demo --help             # Show help"
echo ""
