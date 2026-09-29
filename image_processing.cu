#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

__global__ void processImage(unsigned char *input,
                             unsigned char *output,
                             int width,
                             int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;

        // Simple GPU image processing:
        // invert pixel intensity
        output[idx] = 255 - input[idx];
    }
}

bool readPGM(const std::string &filename,
             std::vector<unsigned char> &data,
             int &width,
             int &height)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
        return false;

    std::string magic;
    int maxValue;

    file >> magic;

    if (magic != "P5")
        return false;

    file >> width >> height >> maxValue;
    file.get();

    data.resize(width * height);
    file.read(reinterpret_cast<char *>(data.data()),
              width * height);

    return true;
}

bool writePGM(const std::string &filename,
              const std::vector<unsigned char> &data,
              int width,
              int height)
{
    std::ofstream file(filename, std::ios::binary);

    if (!file)
        return false;

    file << "P5\n";
    file << width << " " << height << "\n";
    file << "255\n";

    file.write(reinterpret_cast<const char *>(data.data()),
               data.size());

    return true;
}

int main(int argc, char **argv)
{
    std::string inputDir = "data";
    std::string outputDir = "output";

    if (argc > 1)
        inputDir = argv[1];

    if (argc > 2)
        outputDir = argv[2];

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "GPU: " << prop.name << "\n";

    DIR *dir = opendir(inputDir.c_str());

    if (!dir)
    {
        std::cerr << "Cannot open input directory\n";
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int imageCount = 0;

    cudaEventRecord(start);

    struct dirent *entry;

    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;

        if (filename.size() < 4 ||
            filename.substr(filename.size() - 4) != ".pgm")
            continue;

        std::string inputPath = inputDir + "/" + filename;
        std::string outputPath = outputDir + "/" + filename;

        std::vector<unsigned char> hostInput;
        std::vector<unsigned char> hostOutput;

        int width, height;

        if (!readPGM(inputPath, hostInput, width, height))
            continue;

        hostOutput.resize(width * height);

        unsigned char *deviceInput;
        unsigned char *deviceOutput;

        size_t bytes = width * height * sizeof(unsigned char);

        cudaMalloc(&deviceInput, bytes);
        cudaMalloc(&deviceOutput, bytes);

        cudaMemcpy(deviceInput,
                   hostInput.data(),
                   bytes,
                   cudaMemcpyHostToDevice);

        dim3 threads(16, 16);
        dim3 blocks(
            (width + threads.x - 1) / threads.x,
            (height + threads.y - 1) / threads.y
        );

        processImage<<<blocks, threads>>>(
            deviceInput,
            deviceOutput,
            width,
            height
        );

        cudaDeviceSynchronize();

        cudaMemcpy(hostOutput.data(),
                   deviceOutput,
                   bytes,
                   cudaMemcpyDeviceToHost);

        writePGM(outputPath,
                 hostOutput,
                 width,
                 height);

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

        imageCount++;
    }

    closedir(dir);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    std::cout << "Images processed: "
              << imageCount << "\n";

    std::cout << "CUDA processing time: "
              << milliseconds << " ms\n";

    std::cout << "Processing completed successfully\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
