
constexpr int MAXTHREADS = 16;
#define WORK_GROUP_SIZE 256
  //#define VECTOR_SIZE WORK_GROUP_SIZE
#define VECTOR_SIZE 64*1024*1024*WORK_GROUP_SIZE

constexpr int NUM = 1024;
constexpr int MATRIXTILESIZE = 16;
constexpr int WPT = 8;

#include <sycl/sycl.hpp>

typedef int TYPE;
typedef TYPE Array[NUM];

#define VECTOR_ADD vector_add

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

extern void vector_add(int numt, int* A, int* B, int* C);




