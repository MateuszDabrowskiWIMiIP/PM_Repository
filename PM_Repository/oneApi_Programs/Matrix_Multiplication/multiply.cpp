#include <array>
#include <sycl/sycl.hpp>
#include "multiply.hpp"

using namespace sycl;
using namespace std;

template <typename T>
class First_Matrix;

template <typename T>
class Second_Matrix;

template <typename T>
class Third_Matrix;

// I function of matrix multiplication
void Matrix_Multiplication1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM],
    TYPE c[][NUM], TYPE t[][NUM]) {
    int i, j, k;

    default_selector device;
    queue q(device, exception_handler);
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    range<2> matrix_range{ NUM, NUM };

    buffer bufferA((TYPE*)a, range(matrix_range));
    buffer bufferB((TYPE*)b, range(matrix_range));
    buffer bufferC((TYPE*)c, range(matrix_range));

    q.submit([&](sycl::handler& h) {
        accessor accessorA(bufferA, h, read_only);
        accessor accessorB(bufferB, h, read_only);
        accessor accessorC(bufferC, h);

        h.parallel_for<class First_Matrix<TYPE> >(matrix_range, [=](sycl::id<2> ind) {
            int k;
            for (k = 0; k < NUM; k++) {
                accessorC[ind[0]][ind[1]] += accessorA[ind[0]][k] * accessorB[k][ind[1]];
            }
            });
        }).wait_and_throw();
}

void printArray(const char* title, int* A) {
    std::cout << title << std::endl;
    for (int i = 0; i < NUM * NUM; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

// Function to perform matrix multiplication
void matrixMultiply(sycl::queue& q) {
    // Input matrices A and B


    int* A = (int*)malloc(NUM * NUM * sizeof(int*));
    int* B = (int*)malloc(NUM * NUM * sizeof(int*));
    int* C = (int*)malloc(NUM * NUM * sizeof(int*));


    for (int i = 0; i < NUM * NUM; i++) { A[i] = i + 4; }
    for (int i = 0; i < NUM * NUM; i++) { B[i] = i + 45; }
    for (int i = 0; i < NUM * NUM; i++) { C[i] = 0; }

    //printArray("Array A", A);
    //printArray("Array B", B);

    // Create buffers for A, B, and C
    sycl::buffer<int, 1> bufA(A, sycl::range<1>(NUM * NUM));
    sycl::buffer<int, 1> bufB(B, sycl::range<1>(NUM * NUM));
    sycl::buffer<int, 1> bufC(C, sycl::range<1>(NUM * NUM));

    // Submit command group for matrix multiplication
    q.submit([&](sycl::handler& h) {
        auto accA = bufA.get_access<sycl::access::mode::read>(h);
        auto accB = bufB.get_access<sycl::access::mode::read>(h);
        auto accC = bufC.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(NUM * NUM), [=](sycl::item<1> item) {
            int row = item.get_id(0) / NUM;
            int col = item.get_id(0) % NUM;
            int sum = 0;
            for (int i = 0; i < NUM; ++i) {
                sum += accA[row * NUM + i] * accB[i * NUM + col];
            }
            accC[item.get_id(0)] = sum;
            });
        });

    // Copy result C back to the host
    auto host_accC = bufC.get_access<sycl::access::mode::read>();
    //printArray("Array C", C);
    //std::cout << "Array C" << std::endl;
    /*for (int i = 0; i < NUM * NUM; ++i) {

        std::cout << host_accC[i] << " ";
    }*/

    std::cout << std::endl;

}

// II function of matrix multiplication
void Matrix_Multiplication2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]) {

    int i, j, k;

    default_selector device;
    queue q(device, exception_handler);
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    range<2> matrix_range{ NUM, NUM };

    buffer bufferA((TYPE*)a, range(matrix_range));
    buffer bufferB((TYPE*)b, range(matrix_range));
    buffer bufferC((TYPE*)c, range(matrix_range));

    q.submit([&](sycl::handler& h) {
        accessor accessorA(bufferA, h, read_only);
        accessor accessorB(bufferB, h, read_only);
        accessor accessorC(bufferC, h);

        h.parallel_for<class Second_Matrix<TYPE>>(matrix_range, [=](sycl::id<2> ind) {
            int k;
            TYPE acc = 0.0;
            for (k = 0; k < NUM; k++) {
                acc += accessorA[ind[0]][k] * accessorB[k][ind[1]];
            }
            accessorC[ind[0]][ind[1]] = acc;
            });
        }).wait_and_throw();
}

// III function of matrix multiplication
void Matrix_Multiplication3(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM],
    TYPE c[][NUM], TYPE t[][NUM]) {
    int i, j, k;

    default_selector device;
    queue q(device, exception_handler);
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    range<2> matrix_range{ NUM, NUM };
    range<2> tile_range{ MATRIXTILESIZE, MATRIXTILESIZE };

    buffer bufferA((TYPE*)a, range(matrix_range));
    buffer bufferB((TYPE*)b, range(matrix_range));
    buffer bufferC((TYPE*)c, range(matrix_range));

    q.submit([&](sycl::handler& h) {
        accessor accessorA(bufferA, h, read_only);
        accessor accessorB(bufferB, h, read_only);
        accessor accessorC(bufferC, h);

        accessor<TYPE, 2, sycl::access::mode::read_write, sycl::access::target::local> aTile(sycl::range<2>(MATRIXTILESIZE, MATRIXTILESIZE), h);
        accessor<TYPE, 2, sycl::access::mode::read_write, sycl::access::target::local> bTile(sycl::range<2>(MATRIXTILESIZE, MATRIXTILESIZE), h);
        
        h.parallel_for<class Third_Matrix<TYPE>>(sycl::nd_range<2>(matrix_range, tile_range), [=](sycl::nd_item<2> it) {
            int k;
            const int numTiles = NUM / MATRIXTILESIZE;
            const int row = it.get_local_id(0);
            const int col = it.get_local_id(1);
            const int globalRow = MATRIXTILESIZE * it.get_group(0) + row;
            const int globalCol = MATRIXTILESIZE * it.get_group(1) + col;
            TYPE acc = 0.0;
            for (int t = 0; t < numTiles; t++) {
                const int tiledRow = MATRIXTILESIZE * t + row;
                const int tiledCol = MATRIXTILESIZE * t + col;
                aTile[row][col] = accessorA[globalRow][tiledCol];
                bTile[row][col] = accessorB[tiledRow][globalCol];
                it.barrier(sycl::access::fence_space::local_space);
                for (k = 0; k < MATRIXTILESIZE; k++) {
                    acc += aTile[row][k] * bTile[k][col];
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            accessorC[globalRow][globalCol] = acc;
            });
        }).wait_and_throw();
}


void ParallelMultiply(int msize) {
    int NTHREADS = MAXTHREADS;
    int MSIZE = NUM;

    sycl::queue q{ sycl::default_selector() };
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    //Matrix_Multiplication1(MSIZE, NTHREADS, 0, a, b, c, t);
    matrixMultiply(q);
    //Matrix_Multiplication2(MSIZE, NTHREADS, 0, a, b, c, t);
    //Matrix_Multiplication3(MSIZE, NTHREADS, 0, a, b, c, t);
}