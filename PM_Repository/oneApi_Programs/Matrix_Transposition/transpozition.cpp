#include <array>
#include <sycl/sycl.hpp>
#include "transpozition.hpp"
#include <chrono>

using namespace sycl;
using namespace std;
using namespace std::chrono;

void Matrix_Transpozition(int numt, int *A, int *B) {
    int i, j, k;

    default_selector device;
    queue q(device, exception_handler);
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    range<2> matrix_range{ NUM, NUM };

    sycl::buffer<int, 1> bufA(A, sycl::range<1>(rows * cols));
    sycl::buffer<int, 1> bufB(B, sycl::range<1>(cols * rows));

    auto start = high_resolution_clock::now();

    q.submit([&](sycl::handler& h) {
        auto accA = bufA.get_access<sycl::access::mode::read>(h);
        auto accC = bufB.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(rows * cols), [=](sycl::item<1> item) {
            int row = item.get_id(0) / cols;
            int col = item.get_id(0) % cols;
            accC[col * rows + row] = accA[item.get_id(0)];
            });
        });

    auto stop = high_resolution_clock::now();

    auto host_accC = bufB.get_access<sycl::access::mode::read>();
    /*std::cout << "Result of Transposition:" << std::endl;
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            std::cout << host_accC[i * rows + j] << " ";
        }
        std::cout << std::endl;
    }*/

    auto duration2 = duration_cast<nanoseconds>(stop - start);


    duration<double> durationInSeconds = duration<double>(duration2.count()) / 1'000'000'000;

    double seconds = durationInSeconds.count();

    std::cout << "Program execution time in ns: " << duration2.count() << " nanoseconds\n" << seconds << " seconds\n" << std::endl;

}