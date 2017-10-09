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

#include "array_utils.hpp"
#include "sprintz2.h"
#include "bitpack.h"
#include "test_utils.hpp"

#include "debug_utils.hpp" // TODO rm

TEST_CASE("compress8b_rowmajor", "[rowmajor][dbg]") {
    size_t SZ = 64;
    srand(123);

    Vec_i8 compressed((SZ) * 2 + 16);
    Vec_u8 decompressed((SZ));
    compressed.setZero();
    decompressed.setOnes();

    Vec_u8 raw((SZ));
//    raw.setZero();
//    raw.setOnes();
//    for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }
//    for (int i = 0; i < SZ; i++) { raw(i) = ((i + 64) / 1) % 128; }
//    for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? ((i + 64) / 1) % 128 : 0; }
    for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? (i + 64) % 128 : 62 + (i+1) % 4; }
//    for (int i = 0; i < SZ; i++) { raw(i) = (i + 96) % 256; }
//    for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? (i+64) % 128 : 126 + (i+1) % 4; }
//  for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? (i+64) % 128 : 72; }
    
    auto len = compress8b_rowmajor(raw.data(), (SZ), compressed.data());
    printf("compressed length: %lld\n", len);
    
    len = decompress8b_rowmajor(compressed.data(), decompressed.data());
    printf("decompressed length: %lld\n", len);
    REQUIRE(true);

//    CAPTURE(SZ);
//    REQUIRE(decompressed.size() == SZ);
    REQUIRE(ar::all_eq(raw, decompressed));
}

