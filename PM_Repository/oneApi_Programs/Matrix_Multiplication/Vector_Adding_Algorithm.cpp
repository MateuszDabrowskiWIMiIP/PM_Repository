%%writefile lab/simple.cpp
#include <malloc.h>
#include <iostream>
#include <sycl/sycl.hpp>
#include <chrono>

typedef unsigned long long UINT64;

using namespace std;

//#define WORK_GROUP_SIZE 256
//#define VECTOR_SIZE WORK_GROUP_SIZE
//#define VECTOR_SIZE 64*1024*1024*WORK_GROUP_SIZE

constexpr int NUM = 214748364;

using namespace sycl;
using namespace std;
using namespace std::chrono;


auto exception_handler = [](sycl::exception_list exceptionList) {
    for (std::exception_ptr const& e : exceptionList) {
        try {
            std::rethrow_exception(e);
        }
        catch (sycl::exception const& e) {
            std::terminate();
        }
    }
};

void vector_add_oneAPI(int *A, int* B, int *C) {
    int i, j, k;

    default_selector device;
    queue q(device, exception_handler);
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    sycl::buffer<int, 1> bufA(A, sycl::range<1>(NUM));
    sycl::buffer<int, 1> bufB(B, sycl::range<1>(NUM));
    sycl::buffer<int, 1> bufC(C, sycl::range<1>(NUM));

    auto start = high_resolution_clock::now();
    q.submit([&](sycl::handler& h) {
        accessor accessorA(bufA, h, read_only);
        accessor accessorB(bufB, h);
        accessor accessorC(bufC, h, write_only);

        h.parallel_for(NUM, [=](sycl::id<1> ind) {
            accessorC[ind] = accessorA[ind] + accessorB[ind];
            });

        }).wait_and_throw();

        auto stop = high_resolution_clock::now();
        auto program_duration = duration_cast<nanoseconds>(stop - start);
        std::cout << "Program execution time in nanoseconds:: " << program_duration.count() << " nnoseconds\n" << std::endl;
        duration<double> durationInSeconds = duration<double>(program_duration.count()) / 1'000'000'000;
        double seconds = durationInSeconds.count();
        std::cout << "Program execution time in seconds:: " << seconds << " seconds\n" << std::endl;

        /*auto host_accC = bufC.get_access<sycl::access::mode::read>();
        std::cout << "Result of Vector Addition:" << std::endl;
        for (int i = 0; i < NUM; ++i) {
            std::cout << host_accC[i] << " ";
        }
        std::cout << std::endl;*/
}



void printArray(const char* title, int* A) {
    std::cout << title << std::endl;
    for (int i = 0; i < NUM; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

void standardVectorAddingAlgorithm(int* A, int* B, int* C) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < NUM; i++) {
        C[i] = A[i] + B[i];
    }
    auto stop = high_resolution_clock::now();

    auto program_duration = duration_cast<nanoseconds>(stop - start);
    std::cout << "Program Standard Vecotr ADDing execution time in nanoseconds:: " << program_duration.count() << " nnoseconds\n" << std::endl;
    duration<double> durationInSeconds = duration<double>(program_duration.count()) / 1'000'000'000;
    double seconds = durationInSeconds.count();
    std::cout << "Program Standard Vecotr ADDing execution time in seconds:: " << seconds << " seconds\n" << std::endl;
}


int main(int argc, char* argv[]) {

    int* A = (int*)malloc(NUM * NUM * sizeof(int*));
    int* B = (int*)malloc(NUM * NUM * sizeof(int*));
    int* C = (int*)malloc(NUM * NUM * sizeof(int*));
    int* D = (int*)malloc(NUM * NUM * sizeof(int*));


    for (int i = 0; i < NUM; i++) { A[i] = i; }
    for (int i = 0; i < NUM; i++) { B[i] = i; }
    for (int i = 0; i < NUM; i++) { C[i] = 0; }
    for (int i = 0; i < NUM; i++) { D[i] = 0; }


    for (int i=0;i<10;i++) {
        vector_add_oneAPI(A, B, C);
        standardVectorAddingAlgorithm(A,B, D);
    }

    return 0;
}