//==============================================================
// Copyright  2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

constexpr int MAXTHREADS = 16;
constexpr int NUM = 128;
constexpr int MATRIXTILESIZE = 16;
constexpr int WPT = 8;

#include <sycl/sycl.hpp>


typedef float TYPE;
typedef TYPE Array[NUM];

#define MULTIPLY multiply1_2

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

extern void multiply1(int msize, int tidx, int numt, TYPE a[][NUM],
    TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1_1(int msize, int tidx, int numt, TYPE a[][NUM],
    TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1_2(int msize, int tidx, int numt, TYPE a[][NUM],
    TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);

extern void ParallelMultiply(int msize);



