#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Pixel {
    unsigned char r, g, b;
};

__global__ void applyGaussianBlurKernel(Pixel* d_input, Pixel* d_output, float* d_kernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfSize = kernelSize / 2;

    if (x < width && y < height) {
        float rSum = 0.0f, gSum = 0.0f, bSum = 0.0f;
        for (int ky = -halfSize; ky <= halfSize; ++ky) {
            for (int kx = -halfSize; kx <= halfSize; ++kx) {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                float kernelValue = d_kernel[(ky + halfSize) * kernelSize + (kx + halfSize)];
                Pixel p = d_input[ny * width + nx];
                rSum += kernelValue * p.r;
                gSum += kernelValue * p.g;
                bSum += kernelValue * p.b;
            }
        }
        d_output[y * width + x].r = static_cast<unsigned char>(rSum);
        d_output[y * width + x].g = static_cast<unsigned char>(gSum);
        d_output[y * width + x].b = static_cast<unsigned char>(bSum);
    }
}

std::vector<std::vector<float>> generateGaussianKernel(int kernelSize, float sigma) {
    std::vector<std::vector<float>> kernel(kernelSize, std::vector<float>(kernelSize));
    float sum = 0.0f;
    int halfSize = kernelSize / 2;
    float twoSigmaSquare = 2.0f * sigma * sigma;

    for (int x = -halfSize; x <= halfSize; ++x) {
        for (int y = -halfSize; y <= halfSize; ++y) {
            float exponent = -(x * x + y * y) / twoSigmaSquare;
            kernel[x + halfSize][y + halfSize] = exp(exponent) / (M_PI * twoSigmaSquare);
            sum += kernel[x + halfSize][y + halfSize];
        }
    }

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i][j] /= sum;
        }
    }
    return kernel;
}

int main() {
    try {
        std::string inputPath  = "./ex_pics/doggo.jpg";
        std::string outputPath = "./ex_pics/doggo_blurred.png";
        int width, height, channels;

        unsigned char* imageData = stbi_load(inputPath.c_str(), &width, &height, &channels, 3);
        if (!imageData) {
            throw std::runtime_error("Failed to load image");
        }

        std::vector<Pixel> image(width * height);
        for (int i = 0; i < width * height; i++) {
            image[i].r = imageData[i * 3];
            image[i].g = imageData[i * 3 + 1];
            image[i].b = imageData[i * 3 + 2];
        }
        stbi_image_free(imageData);

        int kernelSize;
        float sigma;

        std::cout << "Enter the kernel size >1 (suggested 15): ";
        std::cin >> kernelSize;
        if (kernelSize <= 1 || kernelSize % 2 == 0) {
            std::cerr << "Kernel size must be an odd number greater than 1." << std::endl;
            return -1;
        }

        std::cout << "Enter the sigma value (suggested 5): ";
        std::cin >> sigma;

        auto kernel = generateGaussianKernel(kernelSize, sigma);
        std::vector<float> flatKernel(kernelSize * kernelSize);
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                flatKernel[i * kernelSize + j] = kernel[i][j];
            }
        }

        Pixel *d_input, *d_output;
        float *d_kernel;
        cudaMalloc(&d_input, width * height * sizeof(Pixel));
        cudaMalloc(&d_output, width * height * sizeof(Pixel));
        cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));

        cudaMemcpy(d_input, image.data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, flatKernel.data(), kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + 15) / 16, (height + 15) / 16);

        std::clock_t start = std::clock();
        applyGaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, width, height, kernelSize);
        cudaDeviceSynchronize();
        std::clock_t end = std::clock();

        std::vector<Pixel> blurredImage(width * height);
        cudaMemcpy(blurredImage.data(), d_output, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);

        std::vector<unsigned char> outputData(width * height * 3);
        for (int i = 0; i < width * height; i++) {
            outputData[i * 3] = blurredImage[i].r;
            outputData[i * 3 + 1] = blurredImage[i].g;
            outputData[i * 3 + 2] = blurredImage[i].b;
        }

        stbi_write_png(outputPath.c_str(), width, height, 3, outputData.data(), width * 3);
        double elapsedSeconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        std::cout << "Blurred image saved to: " << outputPath << std::endl;
        std::cout << "Time taken for Gaussian blur: " << elapsedSeconds << " seconds." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return -1;
    }
    return 0;
}

