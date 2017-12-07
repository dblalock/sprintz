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
    TEST_CODEC_MANY_NDIMS_8b(encode_delta_rowmajor_8b, decode_delta_rowmajor_8b);
}

TEST_CASE("doubledelta_rowmajor_8b (no compression)", "[rowmajor][doubledelta][preproc][8b]") {
    printf("executing rowmajor double delta test (no compression)\n");
    TEST_CODEC_MANY_NDIMS_8b(
        encode_doubledelta_rowmajor_8b, decode_doubledelta_rowmajor_8b);
}

// ============================================================ 16b

TEST_CASE("delta_rowmajor_16b (no compression)", "[rowmajor][delta][preproc][16b]") {
    printf("executing 16b rowmajor delta test\n");
    TEST_CODEC_MANY_NDIMS_16b(encode_delta_rowmajor_16b, decode_delta_rowmajor_16b);
    // //     int ndims = 33;
    // //     auto ndims_list = ar::range(ndims, ndims + 1);
    // static const int elem_sz = 2;
    // using uint_t = typename elemsize_traits<elem_sz>::uint_t;
    // using int_t = typename elemsize_traits<elem_sz>::int_t;

    // auto ndims_list = ar::range(1, 129 + 1);
    // for (auto _ndims : ndims_list) {
    //     auto ndims = (uint16_t)_ndims;
    //     printf("---- ndims = %d\n", ndims);
    //     CAPTURE(ndims);
    //    auto comp = [ndims](const uint_t* src, size_t len, int_t* dest) {
    //        return encode_delta_rowmajor_16b(src, (uint32_t)len, dest, ndims);
    //    };
    //    auto decomp = [](int_t* src, uint_t* dest) {
    //        return decode_delta_rowmajor_16b(src, dest);
    //    };

    //    test_codec<2>(comp, decomp);

    //     //        TEST_SQUARES_INPUT(7, comp, decomp);
    //     //        TEST_SQUARES_INPUT(256, comp, decomp);
    //     //         TEST_SQUARES_INPUT(ndims * 16, comp, decomp);
    //     //         TEST_SIMPLE_INPUTS(ndims * 2, comp, decomp);
    //     //         TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
    //     //         TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
    //     // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
    //     // TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
    // }
}

TEST_CASE("doubledelta_rowmajor_16b (no compression)", "[rowmajor][doubledelta][preproc][16b]") {
    printf("executing 16b rowmajor double delta test\n");
    TEST_CODEC_MANY_NDIMS_16b(encode_doubledelta_rowmajor_16b, decode_doubledelta_rowmajor_16b);
}

