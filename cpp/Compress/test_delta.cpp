//
//  test_delta.cpp
//  Compress
//
//  Created by DB on 11/1/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include "catch.hpp"
//#include "eigen/Eigen"

#include "compress_testing.hpp"

// #include "array_utils.hpp"
//#include "sprintz2.h"
// #include "bitpack.h"
#include "delta.h"
#include "test_utils.hpp"

#include "debug_utils.hpp" // TODO rm


// ============================================================ 8b

TEST_CASE("delta_rowmajor_8b (no compression)", "[rowmajor][delta][preproc][8b]") {
    printf("executing rowmajor delta test (no compression) \n");
    // test_codec_many_ndims<1>(encode_delta_rowmajor, decode_delta_rowmajor);
    // using CompF = decltype(encode_delta_rowmajor);
    // using DecompF = decltype(decode_delta_rowmajor);
    // test_codec_many_ndims<1, CompF, DecompF>(encode_delta_rowmajor, decode_delta_rowmajor);

    // auto comp = [](const uint8_t* src, size_t len, int8_t* dest, uint16_t ndims) {
    //     return encode_delta_rowmajor(src, (uint32_t)len, dest, ndims);
    // };
    // auto decomp = [](int8_t* src, uint8_t* dest) {
    //     return decode_delta_rowmajor(src, dest);
    // };
    // test_codec_many_ndims<1>(comp, decomp);

    TEST_CODEC_MANY_NDIMS_8b(encode_delta_rowmajor, decode_delta_rowmajor);

//     int ndims = 33;
//     auto ndims_list = ar::range(ndims, ndims + 1);
//     auto ndims_list = ar::range(1, 129 + 1);
//     for (auto _ndims : ndims_list) {
//         auto ndims = (uint16_t)_ndims;
//         printf("---- ndims = %d\n", ndims);
//         CAPTURE(ndims);
//         auto comp = [ndims](const uint8_t* src, size_t len, int8_t* dest) {
//             return encode_delta_rowmajor(src, (uint32_t)len, dest, ndims);
//         };
//         auto decomp = [](int8_t* src, uint8_t* dest) {
//             return decode_delta_rowmajor(src, dest);
//         };

// //        TEST_SQUARES_INPUT(7, comp, decomp);
// //        TEST_SQUARES_INPUT(256, comp, decomp);
// //         TEST_SQUARES_INPUT(ndims * 16, comp, decomp);
// //         TEST_SIMPLE_INPUTS(ndims * 2, comp, decomp);
// //         TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
// //         TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
//         // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
//         // TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
//         test_codec<1>(comp, decomp);
//     }
}

TEST_CASE("doubledelta_rowmajor_8b (no compression)", "[rowmajor][delta][preproc][8b]") {
    printf("executing rowmajor double delta test (no compression)\n");
    // test_codec_many_ndims<1>(
    //     encode_doubledelta_rowmajor, decode_doubledelta_rowmajor);
    TEST_CODEC_MANY_NDIMS_8b(
        encode_doubledelta_rowmajor, decode_doubledelta_rowmajor);

    // //     int ndims = 33;
    // //     auto ndims_list = ar::range(ndims, ndims + 1);
    // auto ndims_list = ar::range(1, 129 + 1);
    // for (auto _ndims : ndims_list) {
    //     auto ndims = (uint16_t)_ndims;
    //     printf("---- ndims = %d\n", ndims);
    //     CAPTURE(ndims);
    //     auto comp = [ndims](const uint8_t* src, size_t len, int8_t* dest) {
    //         return encode_doubledelta_rowmajor(src, (uint32_t)len, dest, ndims);
    //     };
    //     auto decomp = [](int8_t* src, uint8_t* dest) {
    //         return decode_doubledelta_rowmajor(src, dest);
    //     };
    //     test_codec<1>(comp, decomp);

    //     //                TEST_SQUARES_INPUT(7, comp, decomp);
    //     //        TEST_SQUARES_INPUT(8 * ndims, comp, decomp);
    //     //                TEST_SQUARES_INPUT(256, comp, decomp);
    //     //         TEST_SQUARES_INPUT(ndims * 16, comp, decomp);
    //     //         TEST_SIMPLE_INPUTS(ndims * 2, comp, decomp);
    //     //         TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
    //     //         TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
    //     // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
    //     // TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
    // }
}

// ============================================================ 16b

TEST_CASE("delta_rowmajor_16b (no compression)", "[rowmajor][delta][preproc][16b]") {
    printf("executing 16b rowmajor delta test\n");

    //     int ndims = 33;
    //     auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
//        auto comp = [ndims](uint16_t* src, size_t len, int16_t* dest) {
//            return encode_delta_rowmajor(src, (uint32_t)len, dest, ndims);
//        };
//        auto decomp = [](int16_t* src, uint16_t* dest) {
//            return decode_delta_rowmajor(src, dest);
//        };
//        test_codec<2>(comp, decomp);
        //        TEST_SQUARES_INPUT(7, comp, decomp);
        //        TEST_SQUARES_INPUT(256, comp, decomp);
        //         TEST_SQUARES_INPUT(ndims * 16, comp, decomp);
        //         TEST_SIMPLE_INPUTS(ndims * 2, comp, decomp);
        //         TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
        //         TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
        // TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
    }
}
