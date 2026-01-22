# Simple Makefile for standalone demo

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
TARGET = standalone_demo
NPU_TARGET = npu_flow_demo

all: $(TARGET) $(NPU_TARGET)

$(TARGET): standalone_demo.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) standalone_demo.cpp

$(NPU_TARGET): npu_flow_demo.cpp
	$(CXX) $(CXXFLAGS) -o $(NPU_TARGET) npu_flow_demo.cpp

clean:
	rm -f $(TARGET) $(NPU_TARGET)

run: $(TARGET)
	./$(TARGET)

detailed: $(TARGET)
	./$(TARGET) --detailed

benchmark: $(TARGET)
	./$(TARGET) --iterations 100

npu: $(NPU_TARGET)
	./$(NPU_TARGET)

.PHONY: all clean run detailed benchmark npu
