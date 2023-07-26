
#include <malloc.h>
#include <iostream>
#include <chrono>

#include "dpc_common.hpp"
#include "multiply.hpp"

typedef unsigned long long UINT64;
#define xstr(s) x_str(s)
#define x_str(s) #s

using namespace std;
using namespace chrono;

void InitArr(TYPE row, TYPE col, TYPE off, TYPE a[][NUM]) {
    int i, j;

    for (i = 0; i < NUM; i++) {
        for (j = 0; j < NUM; j++) {
            a[i][j] = row * i + col * j + off;
        }
    }
}

void PrintArr(char* name, TYPE Array[][NUM]) {
    int i, j;

    cout << "\n" << name << "\n";

    for (i = 0; i < NUM; i++) {
        for (j = 0; j < NUM; j++) {
            cout << Array[i][j] << "\t";

        }
        cout << endl;

    }
}


int main(int argc, char* argv[]) {

    dpc_common::TimeInterval matrix_time;
    auto start2 = high_resolution_clock::now();
    ParallelMultiply(NUM);

    double matrix_elapsed = matrix_time.Elapsed();
    auto stop2 = high_resolution_clock::now();

    auto duration2 = duration_cast<nanoseconds>(stop2 - start2);

    std::cout << "Czas wykonania programu 2: " << duration2.count() << " nanosekund\n" << std::endl;

    duration<double> durationInSeconds = duration<double>(duration2.count()) / 1'000'000'000;

    double seconds = durationInSeconds.count();

    std::cout << "Czas wykonania programu 2 w sekundach: " << seconds << " nanosekund\n" << std::endl;
    cout << "Elapsed Time: " << matrix_elapsed << "s\n";

    return 0;
}