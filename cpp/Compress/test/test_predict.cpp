//
//  test_predict.cpp
//  Compress
//
//  Created by DB on 11/4/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include "catch.hpp"
//#include "eigen/Eigen"

#include "compress_testing.hpp"

// #include "array_utils.hpp"
//#include "sprintz2.h"
// #include "bitpack.h"
#include "predict.h"
#include "test_utils.hpp"

#include "debug_utils.hpp" // TODO rm


// ============================================================ sprintz predict

TEST_CASE("xff_rowmajor_8b (no compression)", "[rowmajor][xff][preproc]") {
    printf("executing rowmajor xff 8b test (no compression)\n");
    TEST_CODEC_MANY_NDIMS_8b(encode_xff_rowmajor_8b, decode_xff_rowmajor_8b);
}
//TEST_CASE("xff_rowmajor_16b (no compression)", "[rowmajor][xff][preproc]") {
//    printf("executing rowmajor xff 16b test (no compression)\n");
//    TEST_CODEC_MANY_NDIMS_16b(encode_xff_rowmajor_16b, decode_xff_rowmajor_16b);
//}

//    // int ndims = 9;
//    // auto ndims_list = ar::range(ndims, ndims + 1);
//    auto ndims_list = ar::range(1, 129 + 1);
//    for (auto _ndims : ndims_list) {
//        auto ndims = (uint16_t)_ndims;
//        printf("---- ndims = %d\n", ndims);
//        CAPTURE(ndims);
//        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
//            return encode_xff_rowmajor_8b(src, (uint32_t)len, dest, ndims);
//        };
//        auto decomp = [](int8_t* src, uint8_t* dest) {
//            return decode_xff_rowmajor(src, dest);
//        };
//
//        // TEST_SQUARES_INPUT(7, comp, decomp);
//        // TEST_SQUARES_INPUT(16, comp, decomp);
//       // TEST_SQUARES_INPUT(16 * ndims, comp, decomp);
//       // TEST_SQUARES_INPUT(24 * ndims, comp, decomp);
////                 TEST_SQUARES_INPUT(ndims * 8, comp, decomp);
//        //         TEST_SIMPLE_INPUTS(ndims * 2, comp, decomp);
//        //         TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
//        // TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
//        // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
//        TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
//        // TEST_KNOWN_INPUT(ndims * 19, comp, decomp);
//        // TEST_SIMPLE_INPUTS(ndims * 19, comp, decomp);
//
//        // #define COMP_FUNC comp
//        // #define DECOMP_FUNC decomp
//
//        // auto SZ = ndims * 19;
//        // Vec_u8 raw(SZ);
//        // {
//        //     for (int i = 0; i < SZ; i++) {
//        //         raw(i) = (i % 2) ? (i + 64) % 128 : 0;
//        //     }
//        //     TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);
//        // }
//
//        // vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,
//        //     66, 71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17};
//        // {
//        //     for (auto sz : sizes) {
//        //         CAPTURE(sz);
//        //         TEST_KNOWN_INPUT(sz, COMP_FUNC, DECOMP_FUNC);
//        //     }
//        // }
//        // {
//        //     for (auto sz : sizes) {
//        //         TEST_ZEROS(sz, COMP_FUNC, DECOMP_FUNC);
        //     }
        // }
        // {
        //     for (auto sz : sizes) {
        //         TEST_FUZZ(sz, COMP_FUNC, DECOMP_FUNC);
        //     }
        // }
        // {
        //     TEST_FUZZ(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);
        // }
        // {
        //     TEST_SPARSE(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);
        // }
//}
//
