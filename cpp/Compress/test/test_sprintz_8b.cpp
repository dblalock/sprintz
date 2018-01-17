//
//  test_sprintz_8b.cpp
//  Compress
//
//  Created by DB on 12/5/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include "catch.hpp"
#include "eigen/Eigen"

#include "compress_testing.hpp"

// #include "array_utils.hpp"
//#include "sprintz_delta.h"
// #include "bitpack.h"
#include "sprintz.h"
#include "testing_utils.hpp"

#include "debug_utils.hpp" // TODO rm

TEST_CASE("sprintz 8b delta", "[sprintz][delta]") {
    printf("executing sprintz 8b delta\n");
    
    // int ndims = 1;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
            return sprintz_compress_delta_8b(src, (uint32_t)len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return sprintz_decompress_delta_8b(src, dest);
        };
        
        TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
    }
}

TEST_CASE("sprintz 8b xff", "[sprintz][xff]") {
    printf("executing sprintz 8b xff\n");
    
    // int ndims = 1;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
            return sprintz_compress_xff_8b(src, (uint32_t)len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return sprintz_decompress_xff_8b(src, dest);
        };
        
        TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
    }
}
