# CUDA GPU Image Processing

## Project Overview

This project implements GPU-accelerated image processing using CUDA.

The program processes a large dataset of 200 grayscale PGM images. Each image is 128 × 128 pixels. Image pixels are processed in parallel using a CUDA kernel.

The image processing operation used in this project is pixel-intensity inversion:

    output_pixel = 255 - input_pixel

This demonstrates how CUDA can be used to perform image processing on a large number of independent inputs.

## GPU Computation

The project uses a custom CUDA kernel called `processImage`.

Each CUDA thread processes one pixel of an image.

A two-dimensional CUDA configuration is used:

- Threads per block: 16 × 16
- Each thread processes one pixel
- Blocks are calculated based on the image dimensions

The program transfers image data from CPU memory to GPU memory, executes the CUDA kernel, copies the processed image back to CPU memory, and saves the result.

## Dataset

The demonstration dataset contains:

- 200 grayscale images
- Image format: PGM
- Image resolution: 128 × 128 pixels
- 16,384 pixels per image

The program processes all images in the input directory.

## Requirements

- NVIDIA GPU
- CUDA Toolkit
- `nvcc`
- Linux environment

## Compilation

Compile the CUDA program using:

```bash
nvcc -O2 image_processing.cu -o image_processor
