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
#include "bitpack.h"
#include "test_utils.hpp"

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


#define TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC)                     \
    Vec_i8 compressed((SZ) * 2 + 16);                                   \
    Vec_u8 decompressed((SZ));                                          \
    compressed.setZero();                                               \
    decompressed.setZero();                                             \
    auto len = COMP_FUNC(raw.data(), (SZ), compressed.data());          \
    len = DECOMP_FUNC(compressed.data(), decompressed.data());          \
    CAPTURE(SZ);                                                        \
    REQUIRE(ar::all_eq(raw, decompressed));

//    std::cout << "decompressed size: " << decompressed.size() << "\n";  \
//    std::cout << decompressed.cast<int>() << "\n";  \
//    REQUIRE(ar::all_eq(raw, decompressed));


#define TEST_KNOWN_INPUT(SZ, COMP_FUNC, DECOMP_FUNC)                    \
    do {                                                                \
        Vec_u8 raw((SZ));                                               \
        for (int i = 0; i < (SZ); i++) {                                \
            raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);             \
        }                                                               \
        TEST_COMPRESSOR((SZ), compress8b_delta, decompress8b_delta);    \
    } while(0);


#define TEST_FUZZ(SZ, COMP_FUNC, DECOMP_FUNC)                               \
    do {                                                                    \
    srand(123);                                                             \
    Vec_u8 raw((SZ));                                                       \
    raw.setRandom();                                                        \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 2;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 2;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 2;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 2;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 8;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    } while(0);


#define TEST_ZEROS(SZ, COMP_FUNC, DECOMP_FUNC)                              \
    do {                                                                    \
        Vec_u8 raw(sz);                                                     \
        raw.setZero();                                                      \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                        \
    } while(0);





#define TEST_COMP_DECOMP_PAIR(COMP_FUNC, DECOMP_FUNC)                       \
    do {                                                                    \
        vector<int64_t> sizes {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 66,    \
            72, 127, 128, 129, 4096, 4096 + 17};                            \
        SECTION("known input") {                                            \
            for (auto sz : sizes) {                                         \
                TEST_KNOWN_INPUT(sz, COMP_FUNC, DECOMP_FUNC);               \
            }                                                               \
        }                                                                   \
        SECTION("zeros") {                                                  \
            for (auto sz : sizes) {                                         \
                TEST_ZEROS(sz, COMP_FUNC, DECOMP_FUNC);                     \
            }                                                               \
        }                                                                   \
        SECTION("fuzz_multiple_sizes") {                                    \
            for (auto sz : sizes) {                                         \
                TEST_FUZZ(sz, COMP_FUNC, DECOMP_FUNC);                      \
            }                                                               \
        }                                                                   \
        SECTION("long fuzz") {                                              \
            TEST_FUZZ(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);             \
        }                                                                   \
    } while (0);


#define DEBUG_COMP_DECOMP_PAIR(COMP_FUNC, DECOMP_FUNC)          \
    vector<int64_t> sizes {72};                                 \
    for (auto sz : sizes) {                                     \
        TEST_FUZZ(sz, COMP_FUNC, DECOMP_FUNC);                  \
    }



TEST_CASE("delta_8b_simple", "[delta][bitpack]") {
    TEST_COMP_DECOMP_PAIR(compress8b_delta_simple, decompress8b_delta_simple);
}
TEST_CASE("delta_8b", "[delta][bitpack]") {
    TEST_COMP_DECOMP_PAIR(compress8b_delta, decompress8b_delta);
}
TEST_CASE("delta_8b_online", "[delta][bitpack][dbg]") {
    TEST_COMP_DECOMP_PAIR(compress8b_delta_online, decompress8b_delta_online);
}
TEST_CASE("doubledelta", "[delta][bitpack]") {
    TEST_COMP_DECOMP_PAIR(compress8b_doubledelta, decompress8b_doubledelta);
}
TEST_CASE("dyndelta", "[delta][bitpack]") {
    TEST_COMP_DECOMP_PAIR(compress8b_dyndelta, decompress8b_dyndelta);
}
