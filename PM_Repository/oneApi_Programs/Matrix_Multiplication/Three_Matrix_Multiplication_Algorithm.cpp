%%writefile lab/simple.cpp
#include <array>
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;
using namespace std;
using namespace std::chrono;

constexpr int NUM = 2048;
constexpr int MATRIXTILESIZE = 16;
typedef float TYPE;

template <typename T>
class First_Matrix;

template <typename T>
class Second_Matrix;

template <typename T>
class Third_Matrix;

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

void InitArr(TYPE **a) {
  int i, j;

  for (i = 0; i < NUM; i++) {
    for (j = 0; j < NUM; j++) {
      a[i][j] = 3 * i + 4 * j + 5;
    }
  }
}

void First_Matrix_Multiplication_OneAPI(TYPE **a, TYPE **b,
               TYPE **c, TYPE **t) {
  int i, j, k;

  default_selector device;
  queue q(device, exception_handler);
  cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n"; 
  // Declare a 2 dimensional range
  range<2> matrix_range{NUM, NUM};

  buffer bufferA((TYPE*)a, range(matrix_range));
  buffer bufferB((TYPE*)b, range(matrix_range));
  buffer bufferC((TYPE*)c, range(matrix_range));
  
  auto start = high_resolution_clock::now();
  q.submit([&](sycl::handler& h) { 
    accessor accessorA(bufferA, h, read_only);
    accessor accessorB(bufferB, h, read_only);
    accessor accessorC(bufferC, h);
    
    start = high_resolution_clock::now();
    // Execute matrix multiply in parallel over our matrix_range
    // ind is an index into this range
	h.parallel_for<class First_Matrix<TYPE> >(matrix_range,[=](sycl::id<2> ind) {
		int k;
		for (k = 0; k < NUM; k++) {
		// Perform computation ind[0] is row, ind[1] is col
		accessorC[ind[0]][ind[1]] += accessorA[ind[0]][k] * accessorB[k][ind[1]];
		}
		});
  }).wait_and_throw();
    
  auto stop = high_resolution_clock::now();
  auto program_duration = duration_cast<nanoseconds>(stop - start);
  std::cout << "Program execution time in nanoseconds XNXBNXNXN:: " << program_duration.count() << " nnoseconds\n" << std::endl;
  duration<double> durationInSeconds = duration<double>(program_duration.count()) / 1'000'000'000;
  double seconds = durationInSeconds.count();
  std::cout << "Program execution time in seconds:: WERWERWER" << seconds << " seconds\n" << std::endl;
}

void Second_Matrix_Multiplication_OneAPI(TYPE **a, TYPE **b,TYPE **c, TYPE **t) {
  int i, j, k;

  default_selector device;
  queue q(device, exception_handler);
  cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n"; 

  range<2> matrix_range{NUM, NUM};

  buffer bufferA((TYPE*)a, range(matrix_range));
  buffer bufferB((TYPE*)b, range(matrix_range));
  buffer bufferC((TYPE*)c, range(matrix_range));

  auto start = high_resolution_clock::now();
  q.submit([&](sycl::handler& h) {
    accessor accessorA(bufferA, h, read_only);
    accessor accessorB(bufferB, h, read_only);
    accessor accessorC(bufferC, h);
    start = high_resolution_clock::now();
    // Execute matrix multiply in parallel over our matrix_range
    // ind is an index into this range
    h.parallel_for<class Second_Matrix<TYPE>>(matrix_range,[=](sycl::id<2> ind) {
      int k;
      TYPE acc = 0.0;
      for (k = 0; k < NUM; k++) {
        // Perform computation ind[0] is row, ind[1] is col
        acc += accessorA[ind[0]][k] * accessorB[k][ind[1]];
      }
      accessorC[ind[0]][ind[1]] = acc;
     });
 }).wait_and_throw();
    
 auto stop = high_resolution_clock::now();
 auto program_duration = duration_cast<nanoseconds>(stop - start);
 std::cout << "Program execution time in nanoseconds XNXBNXNXN:: " << program_duration.count() << " nnoseconds\n" << std::endl;
 duration<double> durationInSeconds = duration<double>(program_duration.count()) / 1'000'000'000;
 double seconds = durationInSeconds.count();
 std::cout << "Program execution time in seconds:: WERWERWER" << seconds << " seconds\n" << std::endl;
}

void Third_Matrix_Multiplication_OneAPI(TYPE **a, TYPE **b,
                 TYPE **c, TYPE **t) {
  int i, j, k;

  default_selector device;
  queue q(device, exception_handler);
  cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n"; 

  range<2> matrix_range{NUM, NUM};
  range<2> tile_range{MATRIXTILESIZE, MATRIXTILESIZE};

  buffer bufferA((TYPE*)a, range(matrix_range));
  buffer bufferB((TYPE*)b, range(matrix_range));
  buffer bufferC((TYPE*)c, range(matrix_range));
  auto start = high_resolution_clock::now();

  q.submit([&](sycl::handler& h) { 
    accessor accessorA(bufferA, h, read_only);
    accessor accessorB(bufferB, h, read_only);
    accessor accessorC(bufferC, h);

    accessor<TYPE, 2, sycl::access::mode::read_write, sycl::access::target::local> aTile(sycl::range<2>(MATRIXTILESIZE, MATRIXTILESIZE), h);
    accessor<TYPE, 2, sycl::access::mode::read_write, sycl::access::target::local> bTile(sycl::range<2>(MATRIXTILESIZE, MATRIXTILESIZE), h);
      
    start = high_resolution_clock::now();  
    h.parallel_for<class Third_Matrix<TYPE>>(sycl::nd_range<2>(matrix_range,tile_range),[=](sycl::nd_item<2> it) {
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
          // Perform computation ind[0] is row, ind[1] is col
          acc += aTile[row][k] * bTile[k][col];
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      accessorC[globalRow][globalCol] = acc;
    });
  }).wait_and_throw();
    
  auto stop = high_resolution_clock::now();
  auto program_duration = duration_cast<nanoseconds>(stop - start);
  std::cout << "Program execution time in nanoseconds XNXBNXNXN:: " << program_duration.count() << " nnoseconds\n" << std::endl;
  duration<double> durationInSeconds = duration<double>(program_duration.count()) / 1'000'000'000;
  double seconds = durationInSeconds.count();
  std::cout << "Program execution time in seconds:: WERWERWER" << seconds << " seconds\n" << std::endl;
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

void printArray(const char* title, int* A) {
    std::cout << title << std::endl;
    for (int i = 0; i < NUM * NUM; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

void ParallelMultiply(TYPE **a, TYPE **b, TYPE **c, TYPE **t) {
    sycl::queue q{ sycl::default_selector() };
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

     First_Matrix_Multiplication_OneAPI(a, b, c, t);
     Second_Matrix_Multiplication_OneAPI(a, b, c, t);
     Third_Matrix_Multiplication_OneAPI(a, b, c, t);
}

int main(int argc, char* argv[]) {
  
  TYPE ** a,**b,**c,**t;
    
  a = new TYPE * [NUM];
  b = new TYPE * [NUM];
  c = new TYPE * [NUM];
  t = new TYPE * [NUM];
    
    for(int i = 0; i < NUM; i++ )
  {
    a [ i ] = new TYPE [ NUM ];
    b [ i ] = new TYPE [ NUM ];
    c [ i ] = new TYPE [ NUM ];
    t [ i ] = new TYPE [ NUM ];
  } 

  InitArr(a);
  InitArr(b);

  for (int i=0; i<10; i++) {
      cout <<" Iteracja " << i << "\n";
      ParallelMultiply(a, b, c, t);
  }  
    
  for(int i = 0; i < NUM; i++ )
  {
    delete [ ] a [ i ];
    delete [ ] b [ i ];
    delete [ ] c [ i ];
    delete [ ] t [ i ];
  }
    

  for(int i = 0; i < NUM; i++ ) {
       delete [ ] a;
  delete [ ] b;
  delete [ ] c;
  delete [ ] t; 
  }
 

  return 0;
}