# Multithreaded Image Filtering & Histogram Analyzer

## 📌 Project Description
This project demonstrates how to perform **image convolution in parallel using multithreading in C++**.  
A **300×300 grayscale image** is processed with a **5×5 Laplacian kernel** (edge detection filter).  
The program then calculates and updates a **[0–255] histogram** of the filtered image values in a **thread-safe manner**.  

The task simulates a **real-world scenario** such as:
- Processing CCTV/security camera frames in real time.
- Detecting edges, sharpness, or brightness distributions.
- Understanding light conditions (day/night, dark/bright areas) using histograms.

## 🚀 Features
- Uses **C++17** with `std::thread` for parallelism.  
- Employs **mutex locks** to ensure **thread-safe histogram updates**.  
- Divides the image across **5 threads** (each processes a unique section).  
- Generates a histogram file (`histogram.txt`) for analysis.  
- Implements a **Laplacian kernel** for edge detection.  

## 🖼️ Example Workflow
1. A synthetic 300×300 grayscale image is generated (pixel values between 0–255).  
2. A 5×5 convolution kernel is applied to the image in **parallel threads**.  
3. Each thread updates the **global histogram safely** using `std::mutex`.  
4. The histogram is saved as `histogram.txt`.  

## 🛠️ Requirements
- C++17 or later  
- POSIX threads support (`-pthread` flag)  

## ⚡ Compilation & Run
```bash
g++ -std=c++17 -pthread main.cpp -o convolution
./convolution
