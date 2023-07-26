#include <malloc.h>
#include <iostream>
#include <chrono>

#include "dpc_common.hpp"
#include "vector_add.hpp"


typedef unsigned long long UINT64;
#define xstr(s) x_str(s)
#define x_str(s) #s

using namespace std;

typedef std::array<int, NUM> IntVector;
size_t num_repetitions = 1;

using namespace sycl;
using namespace std;
using namespace std::chrono;



void printArray(const char* title, int* A) {
    std::cout << title << std::endl;
    for (int i = 0; i < NUM; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}


int main(int argc, char* argv[]) {

    int* A = (int*)malloc(NUM * NUM * sizeof(int*));
    int* B = (int*)malloc(NUM * NUM * sizeof(int*));
    int* C = (int*)malloc(NUM * NUM * sizeof(int*));


    for (int i = 0; i < NUM; i++) { A[i] = i + 4; }
    for (int i = 0; i < NUM; i++) { B[i] = i + 45; }
    for (int i = 0; i < NUM; i++) { C[i] = 0; }

    //printArray("Array A", A);
    //printArray("Array B", B);

    cout << "Entry Matrix A" << std::endl;
    //PrintArr("Matrix A", a);
    cout << "Entry Matrix B" << std::endl;
    //PrintArr("Matrix B", b);
    cout << std::endl;
    //PrintArr("Matrix C", c);



    dpc_common::TimeInterval matrix_time;
    auto start = high_resolution_clock::now();
    vector_add(0, A, B, C);
    auto stop = high_resolution_clock::now();

    auto program_duration = duration_cast<nanoseconds>(stop - start);

    std::cout << "Program execution time in nanoseconds:: " << program_duration.count() << " nnoseconds\n" << std::endl;

    duration<double> durationInSeconds = duration<double>(program_duration.count()) / 1'000'000'000;

    // Access the converted value in seconds
    double seconds = durationInSeconds.count();

    std::cout << "Program execution time in seconds:: " << seconds << " seconds\n" << std::endl;

    // Output the result
    std::cout << "Duration in seconds: " << seconds << "s" << std::endl;

    double matrix_elapsed = matrix_time.Elapsed();
    cout << "Elapsed Time: " << matrix_elapsed << "s\n";

    cout << std::endl << "Matrix after changes " << std::endl;
    cout << std::endl;
    //PrintArr("Matrix C", c);
    //printArray("Array C", C);
    cout << std::endl;
 
    return 0;
}