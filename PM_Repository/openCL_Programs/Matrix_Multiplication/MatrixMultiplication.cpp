// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "omp.h"
#include <string.h>
#include "OpenCL_util.h"
#include<array> 
#include<time.h>
#include <chrono>
#include <CL/cl.h>
#include <CL/cl.hpp>
#include <fstream>
#include <random>

#define MAX_SOURCE_SIZE (0x100000)

// Matrix dimensions
#define ROW_A 1024
#define COL_A 1024
#define COL_B 1024

using namespace std::chrono;

#define SCALAR float

cl::Program CreateProgram(const std::string& file) {
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);

    auto platform = platforms[2];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    auto device = devices.front();

    std::ifstream helloWorldFile(file);
    std::string src(std::istreambuf_iterator<char>(helloWorldFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

    cl::Context context(device);

    cl::Program program(context, sources);

    program.build("-cl-std=CL1.2");

    return program;
}


void displayInfo();


// Repeat all kernels multiple times to get an average timing result
#define NUM_RUNS 2

// Size of the matrices - K, M, N (squared)
#define SIZE 128

// Threadblock sizes (e.g. for kernels myGEMM1 or myGEMM2)
#define TS 32


#define ARRAY_SIZE SIZE*SIZE


void printArray(const char * title,int * A) {
    std::cout << title << std::endl;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}


void matrixMultiplication(int* matrixA, int* matrixB, int* resultMatrix, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            int sum = 0;
            for (int k = 0; k < colsA; k++) {
                sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
            }
            resultMatrix[i * colsB + j] = sum;
        }
    }
}

void verifyResults(int* openCLMatrix, int* standardMatrix, int rowsA) {
    for (int i = 0; i < rowsA; i++) {

        if (openCLMatrix[i] != standardMatrix[i]) {
            std::cout << "Inocrrect Matrix Multiplication Calculations!" << std::endl;
            break;
        }
    }

    std::cout << "Calculation are correct!" << std::endl;
}

void matrixMultiplicationOpenCLProgram1() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 100);  // Range of random values: 1 to 100

    int K = SIZE;
    int M = SIZE;
    int N = SIZE;

    int retval;

    int* A = (int*)malloc(M * K * sizeof(int*));
    int* B = (int*)malloc(K * N * sizeof(int*));
    int* C = (int*)malloc(M * N * sizeof(int*));
    int* C_Standard = (int*)malloc(M * N * sizeof(int*));
    for (int i = 0; i < M * K; i++) { A[i] = i + 4; }
    for (int i = 0; i < K * N; i++) { B[i] = i + 45; }


    //printArray("Array A", A);
    //printArray("Array B", B);

    cl::Program program = CreateProgram("ProcessMultidimensionalArray.cl");
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    auto& device = devices.front();


    cl::Buffer buf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, A);
    cl::Buffer buf2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, B);
    cl::Buffer buf3(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, C);


    cl::Kernel kernel(program, "ProcessMultidimensionalArray");

    retval = kernel.setArg(0, SIZE);
    retval = kernel.setArg(1, SIZE);
    retval = kernel.setArg(2, SIZE);
    retval = kernel.setArg(3, buf);
    retval = kernel.setArg(4, buf2);
    retval = kernel.setArg(5, buf3);

    if (retval != CL_SUCCESS) {
        printf("Failed to Set the kernel arguments.\n");
        exit(-1);
    }

    const int WPT = 1;

    const size_t local[2] = { 4, 4 };
    const size_t global[2] = { M, N};

    cl::NDRange globalSize(M, N);


    cl::CommandQueue queue(context, device);

    std::cout << "retval = " << retval << std::endl;

    cl_int weq = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, cl::NullRange);
    queue.enqueueReadBuffer(buf3, GL_TRUE, 0, sizeof(int) * ARRAY_SIZE, C);


    std::cout << "web = " << weq << std::endl;

    //printArray("Array C", C);

    std::cout << "Stanrard Calculations " << std::endl;

    matrixMultiplication(A, B, C_Standard, SIZE, SIZE, SIZE);

    verifyResults(C, C_Standard, SIZE);

   // printArray("Array C", C);


}


void matrixMultiplicationOpenCLProgram2() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 100);  // Range of random values: 1 to 100

    int K = SIZE;
    int M = SIZE;
    int N = SIZE;

    int retval;

    int* A = (int*)malloc(M * K * sizeof(int*));
    int* B = (int*)malloc(K * N * sizeof(int*));
    int* C = (int*)malloc(M * N * sizeof(int*));
    int* C_Standard = (int*)malloc(M * N * sizeof(int*));

    for (int i = 0; i < M * K; i++) { A[i] = i + 4; }
    for (int i = 0; i < K * N; i++) { B[i] = i + 45; }


    //printArray("Array A", A);
    //printArray("Array B", B);


    cl::Program program = CreateProgram("ProcessMultidimensionalArray2.cl");
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    auto& device = devices.front();


    cl::Buffer buf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, A);
    cl::Buffer buf2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, B);
    cl::Buffer buf3(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, C);

    cl::Kernel kernel(program, "ProcessMultidimensionalArray2");

    retval = kernel.setArg(0, SIZE);
    retval = kernel.setArg(1, SIZE);
    retval = kernel.setArg(2, SIZE);
    retval = kernel.setArg(3, buf);
    retval = kernel.setArg(4, buf2);
    retval = kernel.setArg(5, buf3);

    if (retval != CL_SUCCESS) {
        printf("Failed to Set the kernel arguments.\n");
        exit(-1);
    }

    const int WPT = 1;

    const size_t local[2] = { 4, 4 };
    const size_t global[2] = { M, N };

    cl::NDRange globalSize(M, N);


    cl::CommandQueue queue(context, device);

    std::cout << "retval = " << retval << std::endl;

    cl_int weq = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, cl::NullRange);
    queue.enqueueReadBuffer(buf3, GL_TRUE, 0, sizeof(int) * ARRAY_SIZE, C);


    std::cout << "web = " << weq << std::endl;

    //printArray("Array C", C);

    std::cout << "Stanrard Calculations " << std::endl;

    matrixMultiplication(A, B, C_Standard, SIZE, SIZE, SIZE);

    verifyResults(C, C_Standard, SIZE);


    //printArray("Array C", C);
}


void matrixMultiplicationOpenCLProgram3() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 100);  // Range of random values: 1 to 100

    int K = SIZE;
    int M = SIZE;
    int N = SIZE;

    int retval;

    int* A = (int*)malloc(M * K * sizeof(int*));
    int* B = (int*)malloc(K * N * sizeof(int*));
    int* C = (int*)malloc(M * N * sizeof(int*));
    int* C_Standard = (int*)malloc(M * N * sizeof(int*));
    for (int i = 0; i < M * K; i++) { A[i] = i + 4; }
    for (int i = 0; i < K * N; i++) { B[i] = i + 45; }


    //printArray("Array A", A);
    //printArray("Array B", B);


    cl::Program program = CreateProgram("ProcessMultidimensionalArray3.cl");
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    auto& device = devices.front();



    cl::Buffer buf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, A);
    cl::Buffer buf2(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, B);
    cl::Buffer buf3(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, C);

    cl::Kernel kernel(program, "ProcessMultidimensionalArray3");

    retval = kernel.setArg(0, SIZE);
    retval = kernel.setArg(1, SIZE);
    retval = kernel.setArg(2, SIZE);
    retval = kernel.setArg(3, buf);
    retval = kernel.setArg(4, buf2);
    retval = kernel.setArg(5, buf3);

    if (retval != CL_SUCCESS) {
        printf("Failed to Set the kernel arguments.\n");
        exit(-1);
    }

    const int WPT = 1;

    const size_t local[2] = { 4, 4 / WPT };
    const size_t global[2] = { M, N / WPT };

    cl::NDRange globalSize(M, N);


    cl::CommandQueue queue(context, device);

    std::cout << "retval = " << retval << std::endl;

    cl_int weq = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, cl::NullRange);
    queue.enqueueReadBuffer(buf3, GL_TRUE, 0, sizeof(int) * ARRAY_SIZE, C);


    std::cout << "web = " << weq << std::endl;

    //printArray("Array C", C);

    std::cout << "Stanrard Calculations " << std::endl;

    matrixMultiplication(A, B, C, SIZE, SIZE, SIZE);

    verifyResults(C, C_Standard, SIZE);

    //printArray("Array C", C);
}


int main()
{
    displayInfo();
    auto start1 = high_resolution_clock::now();

    matrixMultiplicationOpenCLProgram1();
   
    auto stop1 = high_resolution_clock::now();

    auto start2 = high_resolution_clock::now();

    matrixMultiplicationOpenCLProgram2();
    auto stop2 = high_resolution_clock::now();
    auto start3 = high_resolution_clock::now();
    matrixMultiplicationOpenCLProgram3();
    auto stop3 = high_resolution_clock::now();


    auto duration1 = duration_cast<nanoseconds>(stop1 - start1);
    auto duration2 = duration_cast<nanoseconds>(stop2 - start2);
    auto duration3 = duration_cast<nanoseconds>(stop3 - start3);

    duration<double> durationInSeconds1 = duration<double>(duration1.count()) / 1'000'000'000;

    double seconds1 = durationInSeconds1.count();

    duration<double> durationInSeconds2 = duration<double>(duration2.count()) / 1'000'000'000;

    double seconds2 = durationInSeconds2.count();

    duration<double> durationInSeconds3 = duration<double>(duration3.count()) / 1'000'000'000;

    double seconds3 = durationInSeconds3.count();

    std::cout << "Czas wykonania programu: " << duration1.count() << " nanosekund\n" << " (" << seconds1 << " seconds " << std::endl;
    std::cout << "Czas wykonania programu: " << duration2.count() << " nanosekund\n" << " (" << seconds2 << " seconds " << std::endl;
    std::cout << "Czas wykonania programu: " << duration3.count() << " nanosekund\n" << " (" << seconds3 << " seconds " << std::endl;

    return 0;
}



void displayInfo()
{
    int i, j;
    cl_int retval;
    char* info; size_t size;

    /* POBRANIE I WYSWIETLENIE LICZBY PLATFORM */
    // First, query the total number of platforms
    cl_uint numPlatforms;
    retval = clGetPlatformIDs(0, (cl_platform_id*)NULL, &numPlatforms);
    printf("\nNumber of platforms: %u\n", numPlatforms);

    /* POBRANIE I WYSWIETLENIE INFORMACJI O PLATFORMACH */
    // Next, allocate memory for the installed plaforms, and qeury 
    // to get the list.
    cl_platform_id* platformIds;
    platformIds = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);

    // Then, query the platform IDs
    retval = clGetPlatformIDs(numPlatforms, platformIds, NULL);

    // Iterate through the list of platforms displaying associated information
    printf("CCCCCCCCCCCCCCCC %d", &numPlatforms);
    for (i = 0; i < numPlatforms; i++) {

        printf("\nPlatform ID - %d\n", i);

        //Nazwa
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 0, NULL, &size);
        info = (char*)malloc(size * sizeof(char));
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, size, info, NULL);
        printf("\nPlatform name: ---------------------------------- %s", info);
        free(info);

        //Nazwa producenta
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
        info = (char*)malloc(size * sizeof(char));
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, size, info, NULL);
        printf("\nVendor name: ------------------------------------ %s", info);
        free(info);

        //Wersja platformy
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, 0, NULL, &size);
        info = (char*)malloc(size * sizeof(char));
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, size, info, NULL);
        printf("\nVersion: ---------------------------------------- %s", info);
        free(info);

        //Informacje o profilu wsparcie opencl
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, 0, NULL, &size);
        info = (char*)malloc(size * sizeof(char));
        retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, size, info, NULL);
        printf("\nPlatform profile: ------------------------------- %s:", info);
        free(info);

        /* POBRANIE I WYSWIETLENIE LISTY URZADZEN */
        cl_uint numDevices;
        retval = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        printf("\n\tNumber of devices: %u\n", numDevices);

        /* POBRANIE INFORMACJI O URZADZENIACH */
        cl_device_id* devicesIds;
        devicesIds = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        retval = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, numDevices, devicesIds, NULL);

        for (j = 0; j < numDevices; j++) {

            //Nazwa
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_NAME, 0, NULL, &size);
            info = (char*)malloc(size * sizeof(char));
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_NAME, size, info, &size);
            printf("\nDevice name: ------------------------------------ %s", info);
            free(info);

            //Dostawca
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_VENDOR, 0, NULL, &size);
            info = (char*)malloc(size * sizeof(char));
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_VENDOR, size, info, &size);
            printf("\nDevice vendor: ---------------------------------- %s", info);
            free(info);

            //Dostepny sterownik OpenCL
            retval = clGetDeviceInfo(devicesIds[j], CL_DRIVER_VERSION, 0, NULL, &size);
            info = (char*)malloc(size * sizeof(char));
            retval = clGetDeviceInfo(devicesIds[j], CL_DRIVER_VERSION, size, info, &size);
            printf("\nDriver version: --------------------------------- %s", info);
            free(info);

            //Wspierany OpenCL
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_VERSION, 0, NULL, &size);
            info = (char*)malloc(size * sizeof(char));
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_VERSION, size, info, &size);
            printf("\nSupported OpenCL by device: --------------------- %s", info);
            free(info);

            //Type
            cl_device_type infoType;
            retval = clGetDeviceInfo(devicesIds[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &infoType, NULL);
            switch (infoType)
            {
            case CL_DEVICE_TYPE_CPU:
                printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_CPU");
                break;

            case CL_DEVICE_TYPE_GPU:
                printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_GPU");
                break;

            case CL_DEVICE_TYPE_ACCELERATOR:
                printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_ACCELERATOR");
                break;

            default:
                printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_DEFAULT");
                break;
            };


            //Profil urzadzenia

            //Maksymalna czestotliwosc zegara MHz

            //Rozmiar pamieci globalnej

            //Rozmiar cache pamieci globalnej

            //Rozmiar linijki cache pamieci globalnej

            //Typ pamieci lokalnej

            //Rozmiar pamieci lokalnej

            //Maksymalny rozmiar pamięci do zaalokowania

            //Maksymalny rozmiar bufora stałych

            //Maksymalna liczba jednostek obliczeniowych

            //Maksymalny rozmiar grupy roboczej

            //Maksymalny wymiar przestrzeni wątków

            //Rozszerzenia


        }
        free(devicesIds);

        printf("\n\n\t ---- ************* ----\n");
    }
    printf("\n\n");
    free(platformIds);
}
