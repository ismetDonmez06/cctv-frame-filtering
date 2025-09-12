#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <fstream>
#include <algorithm>

const int IMG_SIZE = 300;
const int KERNEL_SIZE = 5;
const int NUM_THREADS = 5;

std::mutex histMutex;

std::vector<int> histogram(256, 0);

std::vector<std::vector<int>> kernel = {
    {0,  0, -1,  0, 0},
    {0, -1, -2, -1, 0},
    {-1,-2, 16, -2,-1},
    {0, -1, -2, -1, 0},
    {0,  0, -1,  0, 0}
};

void convolutionThread(const std::vector<std::vector<int>>& image,
                       std::vector<std::vector<int>>& output,
                       int startRow, int endRow) 
{
    int outSize = IMG_SIZE - KERNEL_SIZE + 1;

    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < outSize; j++) {
            int sum = 0;
            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                    sum += image[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            sum = std::max(0, std::min(255, sum));
            output[i][j] = sum;
            {
                std::lock_guard<std::mutex> lock(histMutex);
                histogram[sum]++;
            }
        }
    }
}

int main() {
    std::vector<std::vector<int>> image(IMG_SIZE, std::vector<int>(IMG_SIZE));
    std::mt19937 gen(42); 
    std::uniform_int_distribution<> dis(0, 255);

    for (int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            image[i][j] = dis(gen);
        }
    }

    int outSize = IMG_SIZE - KERNEL_SIZE + 1;
    std::vector<std::vector<int>> output(outSize, std::vector<int>(outSize, 0));

    std::vector<std::thread> threads;
    int rowsPerThread = outSize / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == NUM_THREADS - 1) ? outSize : (t + 1) * rowsPerThread;
        threads.emplace_back(convolutionThread, std::cref(image), std::ref(output), startRow, endRow);
    }

    for (auto& th : threads) {
        th.join();
    }
    std::ofstream histFile("histogram.txt");
    if (histFile.is_open()) {
        for (int i = 0; i < 256; i++) {
            histFile << i << " " << histogram[i] << "\n";
        }
        histFile.close();
    }
    return 0;
}
