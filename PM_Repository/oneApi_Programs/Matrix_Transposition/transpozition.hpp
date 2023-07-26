
constexpr int MAXTHREADS = 16;
constexpr int NUM = 4;
constexpr int MATRIXTILESIZE = 16;
constexpr int WPT = 8;

#include <sycl/sycl.hpp>

typedef int TYPE;
typedef TYPE Array[NUM];

#define TRANSP transpozition1
#define rows 1024
#define cols 1024

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

extern void Matrix_Transpozition(int numt, int* A, int* B);



