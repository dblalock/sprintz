//
//  test_sprintz.cpp
//  Compress
//
//  Created by DB on 7/3/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include "compress_testing.hpp"
#include "univariate_8b.h"
#include "bitpack.h"
#include "testing_utils.hpp"

#include "debug_utils.hpp" // TODO rm


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

//vector<int64_t> sizes {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 66,
//    71, 72, 73, 127, 128, 129, 135, 136};


#define DEBUG_COMP_DECOMP_PAIR(COMP_FUNC, DECOMP_FUNC)          \
    vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 66, \
    71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17}; \
    for (auto sz : sizes) {                                     \
        TEST_ZEROS(sz, COMP_FUNC, DECOMP_FUNC);                 \
        TEST_KNOWN_INPUT(sz, COMP_FUNC, DECOMP_FUNC);           \
        srand(123); TEST_FUZZ(sz, COMP_FUNC, DECOMP_FUNC); \
    }


TEST_CASE("just_bitpack_8b_online", "[delta]") {
    // this codec is not designed to handle bytes with MSB of 1, so just
    // test it on input meeting this condition
    vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 66,
        71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17};
    for (auto sz : sizes) {
        Vec_u8 raw(sz);
        raw.setRandom();
        raw /= 2;
        TEST_COMPRESSOR(sz, compress8b_online, decompress8b_online);
    }
}
TEST_CASE("delta_8b_simple", "[delta]") {
    TEST_COMP_DECOMP_PAIR(compress8b_delta_simple, decompress8b_delta_simple);
}
TEST_CASE("delta_8b", "[delta]") {
    TEST_COMP_DECOMP_PAIR(compress8b_delta, decompress8b_delta);
}
TEST_CASE("delta_8b_online", "[delta]") {
    TEST_COMP_DECOMP_PAIR(compress8b_delta_online, decompress8b_delta_online);
}
TEST_CASE("delta2_8b_online", "[delta]") {
    TEST_COMP_DECOMP_PAIR(compress8b_delta2_online, decompress8b_delta2_online);
}
TEST_CASE("delta_8b_rle", "[delta]") {
    TEST_COMP_DECOMP_PAIR(compress8b_delta_rle, decompress8b_delta_rle);
}
TEST_CASE("delta_8b_rle2", "[delta]") {
    TEST_COMP_DECOMP_PAIR(compress8b_delta_rle2, decompress8b_delta_rle2);
}
TEST_CASE("doubledelta", "[delta]") {
    TEST_COMP_DECOMP_PAIR(compress8b_doubledelta, decompress8b_doubledelta);
}
TEST_CASE("dyndelta", "[delta]") {
    TEST_COMP_DECOMP_PAIR(compress8b_dyndelta, decompress8b_dyndelta);
}
