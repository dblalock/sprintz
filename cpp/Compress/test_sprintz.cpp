//
//  test_sprintz.cpp
//  Compress
//
//  Created by DB on 7/3/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include "catch.hpp"
#include "eigen/Eigen"

#include "array_utils.hpp"
#include "sprintz.h"

template<class T> using Vec = Eigen::Array<T, Eigen::Dynamic, 1>;
using Vec_i8 = Vec<int8_t>;
using Vec_u8 = Vec<uint8_t>;
using Vec_i16 = Vec<int16_t>;
using Vec_u16 = Vec<uint16_t>;
//using Vec_u8 = Eigen::Array<uint8_t, Eigen::Dynamic, 1>;
//using Vec_i16 = Eigen::Array<int16_t, Eigen::Dynamic, 1>;
//using Vec_u16 = Eigen::Array<uint16_t, Eigen::Dynamic, 1>;


TEST_CASE("smoke test", "[sanity]") {
    int x = 0;
    REQUIRE(x == 0);
}

TEST_CASE("naiveDelta", "[sanity]") {
    uint16_t sz = 256;
    Vec_u8 raw(sz);
    raw.setRandom();
    raw *= 127;
    Vec_i8 compressed(sz);
    Vec_u8 decompressed(sz);

    auto len = compress8b_naiveDelta(raw.data(), sz, compressed.data());
    REQUIRE(len == sz);
    auto len2 = decompress8b_naiveDelta(compressed.data(), sz, decompressed.data());
    REQUIRE(len2 == sz);

    
    REQUIRE(ar::all_eq(raw, decompressed));
}
