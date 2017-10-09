//
//  compress_testing.hpp
//  Compress
//
//  Created by DB on 10/9/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef compress_testing_h
#define compress_testing_h

#include "catch.hpp"
#include "eigen/Eigen"

#include "array_utils.hpp"

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

//    std::cout << "decompressed size: " << decompressed.size() << "\n";  \
//    std::cout << decompressed.cast<int>() << "\n";  \
//    REQUIRE(ar::all_eq(raw, decompressed));

#define TEST_SIMPLE_INPUTS(_SZ, COMP_FUNC, DECOMP_FUNC)                 \
    do {                                                                \
    auto SZ = (_SZ);                                                    \
    Vec_u8 raw(SZ);                                                     \
    SECTION("<64") {                                                    \
        for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    SECTION("<128") {                                                   \
        for (int i = 0; i < SZ; i++) { raw(i) = ((i + 64) / 1) % 128; } \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    SECTION("<256") {                                                   \
        for (int i = 0; i < SZ; i++) { raw(i) = (i + 96) % 256; }       \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    SECTION("<128, alternating") {                                      \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? ((i + 64) / 1) % 128 : 0;                \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    SECTION("<128, alternating with 62-65") {                           \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 62 + (i+1) % 4;         \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    SECTION("<128, alternating with 126-129") {                         \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i+64) % 128 : 126 + (i+1) % 4;          \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    SECTION("<128, alternating with 72 (regression test)") {            \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 72;                     \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    } while(0);


#define TEST_SQUARES_INPUT(SZ, COMP_FUNC, DECOMP_FUNC) \
    do {                                                                \
        Vec_u8 raw((SZ));                                               \
        for (int i = 0; i < (SZ); i++) {                                \
            raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);             \
        }                                                               \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                  \
    } while(0);


#define TEST_KNOWN_INPUT(SZ, COMP_FUNC, DECOMP_FUNC)                    \
    TEST_SQUARES_INPUT(SZ, COMP_FUNC, DECOMP_FUNC)                      \
    TEST_SIMPLE_INPUTS(SZ, COMP_FUNC, DECOMP_FUNC)


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
        vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,  \
            66, 71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17}; \
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


#endif /* compress_testing_h */
