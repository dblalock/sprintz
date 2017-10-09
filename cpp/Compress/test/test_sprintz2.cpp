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

#define TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC)                     \
    Vec_i8 compressed((SZ) * 2 + 16);                                   \
    Vec_u8 decompressed((SZ));                                          \
    compressed.setZero();                                               \
    decompressed.setOnes();                                             \
    auto len = COMP_FUNC(raw.data(), (SZ), compressed.data());          \
    len = DECOMP_FUNC(compressed.data(), decompressed.data());          \
    CAPTURE(SZ);                                                        \
    REQUIRE(decompressed.size() == SZ);                                 \
    REQUIRE(ar::all_eq(raw, decompressed));


TEST_CASE("compress8b_rowmajor", "[rowmajor][dbg]") {
    TEST_SIMPLE_INPUTS(64, compress8b_rowmajor, decompress8b_rowmajor);
}
