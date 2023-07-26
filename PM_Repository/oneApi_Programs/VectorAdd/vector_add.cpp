#include <array>
#include <sycl/sycl.hpp>
#include "vector_add.hpp"

using namespace sycl;
using namespace std;

void vector_add(int numt, int *A, int* B, int *C) {
    int i, j, k;
    int NTHREADS = MAXTHREADS;
    int MSIZE = NUM;

    default_selector device;
    queue q(device, exception_handler);
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    sycl::buffer<int, 1> bufA(A, sycl::range<1>(NUM));
    sycl::buffer<int, 1> bufB(B, sycl::range<1>(NUM));
    sycl::buffer<int, 1> bufC(B, sycl::range<1>(NUM));


    q.submit([&](sycl::handler& h) {
        accessor accessorA(bufA, h, read_only);
        accessor accessorB(bufB, h);
        accessor accessorC(bufC, h, write_only);

        h.parallel_for(NUM, [=](sycl::id<1> ind) {
            accessorC[ind] = accessorA[ind] + accessorB[ind];
            });
        }).wait_and_throw();

        /*auto host_accC = bufC.get_access<sycl::access::mode::read>();
        std::cout << "Result of Vector Addition:" << std::endl;
        for (int i = 0; i < NUM; ++i) {
            std::cout << host_accC[i] << " ";
        }
        std::cout << std::endl;*/
}
