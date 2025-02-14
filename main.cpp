
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>        
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct Pixel {
    unsigned char r, g, b;
};

std::vector<std::vector<Pixel>> readPPM(const std::string& filename, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    std::string header;
    file >> header;
    if (header != "P6") {
        throw std::runtime_error("Invalid PPM file format");
    }
    file >> width >> height;
    int maxVal;
    file >> maxVal;
    file.ignore();

    std::vector<std::vector<Pixel>> image(height, std::vector<Pixel>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            file.read(reinterpret_cast<char*>(&image[y][x]), 3);
        }
    }
    return image;
}

void writePPM(const std::string& filename, const std::vector<std::vector<Pixel>>& image) {
    int height = image.size();
    int width  = image[0].size();
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            file.write(reinterpret_cast<const char*>(&image[y][x]), 3);
        }
    }
}

// ------------------------------------------
//  STB image reading for JPG, PNG, etc.
// ------------------------------------------
std::vector<std::vector<Pixel>> readImageStb(const std::string &filename, int &width, int &height) {
    int channels;
    // Force 3 channels (RGB) so everything aligns with your Pixel struct
    unsigned char *data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    // Convert the raw data into your 2D Pixel vector
    std::vector<std::vector<Pixel>> image(height, std::vector<Pixel>(width));
    int index = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Pixel p;
            p.r = data[index + 0];
            p.g = data[index + 1];
            p.b = data[index + 2];
            image[y][x] = p;
            index += 3;
        }
    }

    stbi_image_free(data);
    return image;
}

std::vector<std::vector<float>> generateGaussianKernel(int kernelSize, float sigma) {
    std::vector<std::vector<float>> kernel(kernelSize, std::vector<float>(kernelSize));
    float sum = 0.0f;

    int halfSize = kernelSize / 2;
    float twoSigmaSquare = 2.0f * sigma * sigma;

    for (int x = -halfSize; x <= halfSize; ++x) {
        for (int y = -halfSize; y <= halfSize; ++y) {
            float exponent = -(x * x + y * y) / twoSigmaSquare;
            kernel[x + halfSize][y + halfSize] = std::exp(exponent) / (M_PI * twoSigmaSquare);
            sum += kernel[x + halfSize][y + halfSize];
        }
    }

    // Normalize
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i][j] /= sum;
        }
    }
    return kernel;
}

std::vector<std::vector<Pixel>> applyGaussianBlur(
    const std::vector<std::vector<Pixel>>& image,
    const std::vector<std::vector<float>>& kernel
) {
    int kernelSize = kernel.size();
    int halfSize   = kernelSize / 2;
    int width      = image[0].size();
    int height     = image.size();

    std::vector<std::vector<Pixel>> output(height, std::vector<Pixel>(width));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float rSum = 0.0f, gSum = 0.0f, bSum = 0.0f;
            for (int ky = -halfSize; ky <= halfSize; ++ky) {
                for (int kx = -halfSize; kx <= halfSize; ++kx) {
                    int ny = std::min(std::max(y + ky, 0), height - 1);
                    int nx = std::min(std::max(x + kx, 0), width - 1);

                    rSum += kernel[ky + halfSize][kx + halfSize] * image[ny][nx].r;
                    gSum += kernel[ky + halfSize][kx + halfSize] * image[ny][nx].g;
                    bSum += kernel[ky + halfSize][kx + halfSize] * image[ny][nx].b;
                }
            }
            output[y][x].r = static_cast<unsigned char>(rSum);
            output[y][x].g = static_cast<unsigned char>(gSum);
            output[y][x].b = static_cast<unsigned char>(bSum);
        }
    }
    return output;
}


int main() {
    try {
        std::string inputPath  = "./ex_pics/doggo.jpg";   //JPEG, PNG, BMP, PSD, GIF (1st frame only), HDR... 
        std::string outputPath = "./ex_pics/doggo_blurred.ppm"; //ppm only

        int width, height;

        std::vector<std::vector<Pixel>> image;
        if (inputPath.size() >= 4 &&
            (inputPath.compare(inputPath.size() - 4, 4, ".ppm") == 0 ||
             inputPath.compare(inputPath.size() - 4, 4, ".PPM") == 0)) 
        {
            image = readPPM(inputPath, width, height);
        }
        else {
            image = readImageStb(inputPath, width, height);
        }

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

        // Generate kernel
        auto kernel = generateGaussianKernel(kernelSize, sigma);

        // Measure time
        std::clock_t start = std::clock();
        auto blurredImage = applyGaussianBlur(image, kernel);
        std::clock_t end = std::clock();
        double elapsedSeconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;

        writePPM(outputPath, blurredImage);

        std::cout << "Blurred image saved to: " << outputPath << std::endl;
        std::cout << "Time taken for Gaussian blur: " << elapsedSeconds << " seconds." << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
