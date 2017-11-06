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

TEST_CASE("xff_rowmajor_8b (no compression)", "[rowmajor][delta]") {
    printf("executing rowmajor xff test\n");
    
    //     int ndims = 33;
    //     auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
            return encode_xff_rowmajor(src, (uint32_t)len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return decode_xff_rowmajor(src, dest);
        };
        
        //        TEST_SQUARES_INPUT(7, comp, decomp);
        //        TEST_SQUARES_INPUT(256, comp, decomp);
        //         TEST_SQUARES_INPUT(ndims * 16, comp, decomp);
        //         TEST_SIMPLE_INPUTS(ndims * 2, comp, decomp);
        //         TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
        //         TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
        TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
    }
}

