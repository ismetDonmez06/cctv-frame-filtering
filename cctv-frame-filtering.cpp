#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <fstream>
#include <algorithm>

// Image and kernel parameters
const int IMG_SIZE = 300;
const int KERNEL_SIZE = 5;
const int NUM_THREADS = 5;

// Mutex for thread-safe histogram updates
std::mutex histMutex;

// Histogram array [0-255], initialized with zeros
std::vector<int> histogram(256, 0);

// Laplacian-like kernel for edge detection
std::vector<std::vector<int>> kernel = {
    {0,  0, -1,  0, 0},
    {0, -1, -2, -1, 0},
    {-1,-2, 16, -2,-1},
    {0, -1, -2, -1, 0},
    {0,  0, -1,  0, 0}
};

// Convolution function executed by each thread
void convolutionThread(const std::vector<std::vector<int>>& image,
                       std::vector<std::vector<int>>& output,
                       int startRow, int endRow) 
{
    int outSize = IMG_SIZE - KERNEL_SIZE + 1;

    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < outSize; j++) {
            int sum = 0;
            // Apply kernel over the region
            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                    sum += image[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            // Normalize result into [0-255] range
            sum = std::max(0, std::min(255, sum));
            output[i][j] = sum;

            // Thread-safe histogram update
            {
                std::lock_guard<std::mutex> lock(histMutex);
                histogram[sum]++;
            }
        }
    }
}

int main() {
    // 1. Generate a random grayscale "camera frame"
    std::vector<std::vector<int>> image(IMG_SIZE, std::vector<int>(IMG_SIZE));
    std::mt19937 gen(42); // fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, 255);

    for (int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            image[i][j] = dis(gen);
        }
    }

    // 2. Output image after applying the filter
    int outSize = IMG_SIZE - KERNEL_SIZE + 1;
    std::vector<std::vector<int>> output(outSize, std::vector<int>(outSize, 0));

    // 3. Launch threads for convolution
    std::vector<std::thread> threads;
    int rowsPerThread = outSize / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == NUM_THREADS - 1) ? outSize : (t + 1) * rowsPerThread;
        threads.emplace_back(convolutionThread, std::cref(image), std::ref(output), startRow, endRow);
    }

    // Wait for all threads to complete
    for (auto& th : threads) {
        th.join();
    }

    // 4. Save histogram to file
    std::ofstream histFile("histogram.txt");
    if (histFile.is_open()) {
        for (int i = 0; i < 256; i++) {
            histFile << i << " " << histogram[i] << "\n";
        }
        histFile.close();
    }

    std::cout << "✅ Convolution completed. Histogram written to 'histogram.txt'.\n";
    return 0;
}
