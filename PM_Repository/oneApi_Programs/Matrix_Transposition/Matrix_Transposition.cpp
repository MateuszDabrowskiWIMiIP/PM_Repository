#include <malloc.h>
#include <iostream>
#include <chrono>

#include "dpc_common.hpp"
#include "transpozition.hpp"


typedef unsigned long long UINT64;
#define xstr(s) x_str(s)
#define x_str(s) #s

#define rowsA 4
#define rowsB 4
#define rowsC 4
#define colsA 4
#define colsB 4
#define colsC 4

using namespace std;
using namespace std::chrono;


void printArray(const char* title, int* A) {
    std::cout << title << std::endl;
    for (int i = 0; i < rowsA * colsA; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}


int main(int argc, char* argv[]) {

    int* A = (int*)malloc(rows * cols * sizeof(int*));
    int* B = (int*)malloc(rows * cols * sizeof(int*));


    for (int i = 0; i < rows * cols; i++) { A[i] = i + 4; }
    for (int i = 0; i < rows * cols; i++) { B[i] = 0; }

    cout << "Entry Matrix A" << endl;
    //printArray("Matrix A", A);
    cout << "Entry Matrix B" << endl;
    //printArray("Matrix B", B);

    //cout << "Using multiply kernel: " << xstr(MULTIPLY) << "\n";

    dpc_common::TimeInterval matrix_time;
    //auto start = high_resolution_clock::now();
    Matrix_Transpozition(0, A, B);
    //auto stop = high_resolution_clock::now();

    //auto duration2 = duration_cast<nanoseconds>(stop - start);


    //duration<double> durationInSeconds = duration<double>(duration2.count()) / 1'000'000'000;

    //double seconds = durationInSeconds.count();

    //std::cout << "Program execution time in ns: " << duration2.count() << " nanoseconds\n" << seconds << " seconds\n" << std::endl;

    double matrix_elapsed = matrix_time.Elapsed();
    cout << "Elapsed Time: " << matrix_elapsed << "s\n";

    cout << endl << "Matrix after changes " << endl;
    //PrintArr("Matrix A", a);
    //cout << endl;
    //printArray("Matrix B", B);
    cout << endl;

    return 0;
}