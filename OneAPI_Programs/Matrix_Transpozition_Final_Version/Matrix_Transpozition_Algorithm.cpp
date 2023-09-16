%%writefile lab/simple.cpp
#include <array>
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;
using namespace std;
using namespace std::chrono;

constexpr int NUM = 4096;
typedef float TYPE;
typedef TYPE Array[NUM];

#define TRANSP transpozition1
#define rows 16384
#define cols 16384

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


void Matrix_Transpozition_OneAPI(int numt, unsigned long long *A, unsigned long long *B) {
    int i, j, k;

    default_selector device;
    queue q(device, exception_handler);
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    range<2> matrix_range{ NUM, NUM };

    sycl::buffer<unsigned long long, 1> bufA(A, sycl::range<1>(rows * cols));
    sycl::buffer<unsigned long long, 1> bufB(B, sycl::range<1>(cols * rows));

    auto start = high_resolution_clock::now();
    q.submit([&](sycl::handler& h) {
        auto accA = bufA.get_access<sycl::access::mode::read>(h);
        auto accC = bufB.get_access<sycl::access::mode::write>(h);
        
        start = high_resolution_clock::now();
        h.parallel_for(sycl::range<1>(rows * cols), [=](sycl::item<1> item) {
            unsigned long long row = item.get_id(0) / cols;
            unsigned long long col = item.get_id(0) % cols;
            accC[col * rows + row] = accA[item.get_id(0)];
            });

        }).wait_and_throw();

        auto stop = high_resolution_clock::now();
        auto program_duration = duration_cast<nanoseconds>(stop - start);
        std::cout << "Pro:: 2222      " << program_duration.count() << " nnoseconds\n" << std::endl;
        duration<double> durationInSeconds = duration<double>(program_duration.count()) / 1'000'000'000;
        double seconds = durationInSeconds.count();
        std::cout << "Program execution time in seconds:: 22222      " << seconds << " seconds\n" << std::endl;

    //auto host_accC = bufB.get_access<sycl::access::mode::read>();
    /*std::cout << "Result of Transposition:" << std::endl;
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            std::cout << host_accC[i * rows + j] << " ";
        }
        std::cout << std::endl;
    }*/
}

void transposeMatrixStandardAlgorithm(unsigned long long* arr, unsigned long long* B) {
    auto start = high_resolution_clock::now();

    unsigned long long** transposed;

    transposed = new unsigned long long* [rows];

    for (unsigned long long iw = 0; iw < rows; iw++) transposed[iw] = new unsigned long long[cols];

    // Fill the transposed matrix
    for (unsigned long long i = 0; i < rows; ++i) {
        for (unsigned long j = 0; j < cols; ++j) {
            transposed[j][i] = arr[i];
        }
    }

    // Copy the transposed matrix back to the original array
    for (unsigned long long i = 0; i < cols; ++i) {
        for (unsigned long long j = 0; j < rows; ++j) {
            B[i * rows + j] = transposed[i][j];
        }
    }
    auto stop = high_resolution_clock::now();
    auto program_duration = duration_cast<nanoseconds>(stop - start);
    std::cout << "Matrix Transpozition Standard:: " << program_duration.count() << " nnoseconds\n" << std::endl;
    duration<double> durationInSeconds = duration<double>(program_duration.count()) / 1'000'000'000;
    double seconds = durationInSeconds.count();
    std::cout << "Program execution Normal Algorithm time in seconds:: 33333333      " << seconds << " seconds\n" << std::endl;

    for (unsigned long long iw = 0; iw < rows; iw++) delete[] transposed[iw];

    delete[] transposed;
}

void TestProgram() {

    unsigned long long* A = (unsigned long long*)malloc(rows * cols * sizeof(unsigned long long*));
    unsigned long long* B = (unsigned long long*)malloc(rows * cols * sizeof(unsigned long long*));


    for (unsigned long i = 0; i < rows * cols; i++) { A[i] = i + 4; }
    for (unsigned long i = 0; i < rows * cols; i++) { B[i] = 0; }

    std::cout << "Entry Matrix A" << std::endl;
    //printArray("Matrix A", A);
    std::cout << "Entry Matrix B" << std::endl;
    //printArray("Matrix B", B);

    Matrix_Transpozition_OneAPI(0, A, B);
    transposeMatrixStandardAlgorithm(A, B);

}


int main(int argc, char* argv[]) {

    for (int i = 0; i < 10; i++) {
        TestProgram();
    }

    return 0;
}