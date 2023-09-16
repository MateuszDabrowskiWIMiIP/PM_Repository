%%writefile lab/simple.cpp
#include <array>
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;
using namespace std;
using namespace std::chrono;

constexpr int NUM = 2048;
typedef float TYPE;
typedef TYPE Array[NUM];

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


void StandardMatrixMultiplication(int* matrixA, int* matrixB, int* resultMatrix, int rowsA, int colsA, int colsB) {
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

void printArray(const char* title, int* A) {
    std::cout << title << std::endl;
    for (int i = 0; i < NUM * NUM; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

void matrixMultiply(sycl::queue& q) {

    int* A = (int*)malloc(NUM * NUM * sizeof(int*));
    int* B = (int*)malloc(NUM * NUM * sizeof(int*));
    int* C = (int*)malloc(NUM * NUM * sizeof(int*));

    for (int i = 0; i < NUM * NUM; i++) { A[i] = i + 4; }
    for (int i = 0; i < NUM * NUM; i++) { B[i] = i + 45; }
    for (int i = 0; i < NUM * NUM; i++) { C[i] = 0; }

    //printArray("Array A", A);
    //printArray("Array B", B);

    sycl::buffer<int, 1> bufA(A, sycl::range<1>(NUM * NUM));
    sycl::buffer<int, 1> bufB(B, sycl::range<1>(NUM * NUM));
    sycl::buffer<int, 1> bufC(C, sycl::range<1>(NUM * NUM));

    auto start = high_resolution_clock::now();

    q.submit([&](sycl::handler& h) {
        auto accA = bufA.get_access<sycl::access::mode::read>(h);
        auto accB = bufB.get_access<sycl::access::mode::read>(h);
        auto accC = bufC.get_access<sycl::access::mode::write>(h);

        start = high_resolution_clock::now();
        h.parallel_for(sycl::range<1>(NUM * NUM), [=](sycl::item<1> item) {
            int row = item.get_id(0) / NUM;
            int col = item.get_id(0) % NUM;
            int sum = 0;
            for (int i = 0; i < NUM; ++i) {
                sum += accA[row * NUM + i] * accB[i * NUM + col];
            }
            accC[item.get_id(0)] = sum;
            });     
        }).wait_and_throw();

    auto stop = high_resolution_clock::now();
    auto program_duration = duration_cast<nanoseconds>(stop - start);
    std::cout << "Program execution time in nanoseconds XNXBNXNXN:: " << program_duration.count() << " nnoseconds\n" << std::endl;
    duration<double> durationInSeconds = duration<double>(program_duration.count()) / 1'000'000'000;
    double seconds = durationInSeconds.count();
    std::cout << "Program execution time in seconds:: WERWERWER" << seconds << " seconds\n" << std::endl;

    // Copy result C back to the host
    auto host_accC = bufC.get_access<sycl::access::mode::read>();
    //printArray("Array C", C);
    std::cout << std::endl;
    //for (int i = 0; i < NUM * NUM; i++) { C[i] = 0; }
    //printArray("Array C  Before", C);

    auto start4 = high_resolution_clock::now();

    StandardMatrixMultiplication(A,B, C, NUM, NUM, NUM);
    auto stop4 = high_resolution_clock::now();

    auto program_duration4 = duration_cast<nanoseconds>(stop4 - start4);
    std::cout << "Standard Multiplication Program execution time in nanoseconds WERWERWER:: " << program_duration4.count() << " nnoseconds\n" << std::endl;
    duration<double> durationInSeconds4 = duration<double>(program_duration4.count()) / 1'000'000'000;
    double seconds4 = durationInSeconds4.count();
    std::cout << "Standard Multiplication  Program execution time in seconds:: WERWERWER" << seconds4 << " seconds\n" << std::endl;

    //printArray("Array C", C);

}

void ParallelMultiply() {
    sycl::queue q{ sycl::default_selector() };
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    matrixMultiply(q);
}

int main(int argc, char* argv[]) {

    for (int i=0; i< 10; i++) {
        ParallelMultiply();
    }

    return 0;
}