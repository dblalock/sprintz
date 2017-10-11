//
//  test_sprintz2.cpp
//  Compress
//
//  Created by DB on 9/28/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include "catch.hpp"
#include "eigen/Eigen"

#include "compress_testing.hpp"

// #include "array_utils.hpp"
#include "sprintz2.h"
// #include "bitpack.h"
#include "test_utils.hpp"

#include "debug_utils.hpp" // TODO rm

//TEST_CASE("how_many_times_are_these_run?", "[rowmajor][dbg]") {
//    printf("running nop test\n");
//    auto ndims_list = ar::range(1, 33 + 1);
//    for (auto ndims : ndims_list) {
//        printf("------ ndims = %d\n", ndims);
//    }
//
//    REQUIRE(true);
//}

// template<typename SizeT, typename CompT, typename DecompT>
// void _test_simple_inputs(SizeT SZ, CompT f_comp, DecompT f_decomp) {
//     {
//         Vec_u8 raw(SZ);
//         for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }
//         TEST_COMPRESSOR(SZ, f_comp, f_decomp);
//     }
//     {
//         Vec_u8 raw(SZ);
//         for (int i = 0; i < SZ; i++) { raw(i) = ((i + 64) / 1) % 128; }
//         TEST_COMPRESSOR(SZ, f_comp, f_decomp);
//     }
// }

//*
 #undef TEST_SIMPLE_INPUTS
 #define TEST_SIMPLE_INPUTS(SZ, COMP_FUNC, DECOMP_FUNC)                 \
    {                                                                   \
        Vec_u8 raw(SZ);                                                 \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 0;                \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \

/*
     {                                                   \
         Vec_u8 raw(SZ);                                                 \
         for (int i = 0; i < SZ; i++) { raw(i) = (i + 64) % 128; } \
         TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
     }                                                                  \
{                                                    \
    Vec_u8 raw(SZ);                                                 \
    for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }               \
    TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
}                                                                   \
//*/

TEST_CASE("compress8b_rowmajor", "[rowmajor][dbg]") {
//     TEST_SIMPLE_INPUTS(1, compress8b_rowmajor, decompress8b_rowmajor);
//     TEST_SIMPLE_INPUTS(63, compress8b_rowmajor, decompress8b_rowmajor);
//     TEST_SIMPLE_INPUTS(64, compress8b_rowmajor, decompress8b_rowmajor);
//     TEST_SIMPLE_INPUTS(127, compress8b_rowmajor, decompress8b_rowmajor);
//     TEST_SIMPLE_INPUTS(128, compress8b_rowmajor, decompress8b_rowmajor);
//     TEST_SIMPLE_INPUTS(129, compress8b_rowmajor, decompress8b_rowmajor);
//     TEST_SIMPLE_INPUTS(4096 + 17, compress8b_rowmajor, decompress8b_rowmajor);

    printf("executing rowmajor test\n");

//    uint16_t ndims = 8;
//     auto ndims_list = ar::range(1, 33 + 1);
    auto ndims_list = ar::range(1, 129 + 1);
//    auto ndims_list = ar::range(33, 33 + 1);
//    auto ndims_list = ar::range(65, 65 + 1);
//    auto ndims_list = ar::range(1, 2);
//    ar::print(ndims_list, "ndims_list");
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
            return compress8b_rowmajor(src, len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return decompress8b_rowmajor(src, dest);
        };

        //    size_t SZ = ndims * 8;
//            size_t SZ = 64;
//            Vec_u8 raw(SZ);
//        //    for (int i = 0; i < SZ; i++) { raw(i) = 64 + i % 64; }
//        //    for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? 64 + i % 64 : 72; }
//            for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? (i + 64) % 128 : 0;}
//            TEST_COMPRESSOR(SZ, comp, decomp);

//        TEST_SQUARES_INPUT(1, comp, decomp);
//        _test_simple_inputs(1, comp, decomp);
//       TEST_SIMPLE_INPUTS(1, comp, decomp);
//       TEST_SIMPLE_INPUTS(63, comp, decomp);
//       TEST_SIMPLE_INPUTS(64, comp, decomp);
//       TEST_SIMPLE_INPUTS(65, comp, decomp);
//       TEST_SIMPLE_INPUTS(ndims * 8 - 1, comp, decomp);
       TEST_SIMPLE_INPUTS(ndims * 8, comp, decomp);
//       TEST_SIMPLE_INPUTS(ndims * 8 + 1, comp, decomp);
//       TEST_SIMPLE_INPUTS(ndims * 16 - 1, comp, decomp);
//       TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
//       TEST_SIMPLE_INPUTS(ndims * 16 + 1, comp, decomp);
//       TEST_SIMPLE_INPUTS(127, comp, decomp);
//       TEST_SIMPLE_INPUTS(128, comp, decomp);
//       TEST_SIMPLE_INPUTS(129, comp, decomp);
//       TEST_SIMPLE_INPUTS(4096 + 17, comp, decomp);
    }
//    REQUIRE(true);
}
