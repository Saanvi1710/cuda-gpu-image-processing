#!/bin/bash

set -e

mkdir -p output results

echo "Compiling CUDA program..."
nvcc -O2 image_processing.cu -o image_processor

echo "Running GPU image processing..."
./image_processor data output | tee results/execution_log.txt

echo "Execution completed."
