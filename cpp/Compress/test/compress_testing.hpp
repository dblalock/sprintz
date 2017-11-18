//
//  compress_testing.hpp
//  Compress
//
//  Created by DB on 10/9/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

// TODO nothing in this file really has to be a macro anymore...

#ifndef compress_testing_h
#define compress_testing_h

#include "catch.hpp"
#include "eigen/Eigen"

#include "array_utils.hpp"
/*
    compressed.setZero();                                               \
    decompressed.setOnes();                                             \
    */

#define TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC)                     \
    Vec_i8 compressed((SZ) * 3/2 + 64);                                 \
    Vec_u8 decompressed((SZ)+ 64);                                      \
    compressed += 0x55; /* poison memory */                             \
    decompressed += 0xaa;                                               \
    auto len = COMP_FUNC(raw.data(), (SZ), compressed.data());          \
    len = DECOMP_FUNC(compressed.data(), decompressed.data());          \
    CAPTURE(SZ);                                                        \
    REQUIRE(len == (SZ));                                               \
    auto arrays_eq = ar::all_eq(raw.data(), decompressed.data(), (SZ)); \
    if ((SZ) < 100 && !arrays_eq) {                                     \
        printf("**** Test Failed! ****\n");                             \
        auto input_as_str = ar::to_string(raw.data(), (SZ));            \
        auto output_as_str = ar::to_string(decompressed.data(), (SZ));  \
        printf("input:\t%s\n", input_as_str.c_str());                   \
        printf("output:\t%s\n", output_as_str.c_str());                 \
    }                                                                   \
    REQUIRE(arrays_eq);

/*
auto input_bytes_as_str = ar::to_string(raw.data(), (SZ));            \
auto output_bytes_as_str = ar::to_string(decompressed.data(), (SZ));  \
CAPTURE(input_bytes_as_str);                                          \
CAPTURE(output_bytes_as_str);                                         \
*/


    // }
// printf("%s\n", input_as_str.c_str());
// printf("%s\n", output_as_str.c_str());

//    std::cout << "decompressed size: " << decompressed.size() << "\n";
//    std::cout << decompressed.cast<int>() << "\n";
//    REQUIRE(ar::all_eq(raw, decompressed));

// would be nice to separate tests into Catch sections, but Catch runs the
// whole enclosing test_case a number of times equal to the number of sections
// for mysterious reasons
#define TEST_SIMPLE_INPUTS(_SZ, COMP_FUNC, DECOMP_FUNC)                 \
    do {                                                                \
    size_t SZ = (size_t)(_SZ);                                          \
    Vec_u8 raw(SZ);                                                     \
    {                                                                   \
        for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) { raw(i) = (i + 64) % 128; }       \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) { raw(i) = (i + 96) % 256; }       \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 0;                      \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 62 + (i + 1) % 4;       \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 126 + (i + 1) % 4;      \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 72;                     \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    } while(0);


#define TEST_SQUARES_INPUT(SZ, COMP_FUNC, DECOMP_FUNC)                  \
    do {                                                                \
        Vec_u8 raw((SZ));                                               \
        for (int i = 0; i < (SZ); i++) {                                \
            raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);             \
        }                                                               \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                  \
    } while(0);


#define TEST_KNOWN_INPUT(SZ, COMP_FUNC, DECOMP_FUNC)                    \
    TEST_SQUARES_INPUT(SZ, COMP_FUNC, DECOMP_FUNC);                     \
    TEST_SIMPLE_INPUTS(SZ, COMP_FUNC, DECOMP_FUNC);


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
        Vec_u8 raw(SZ);                                                     \
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


#define TEST_COMP_DECOMP_PAIR_NO_SECTIONS(COMP_FUNC, DECOMP_FUNC)           \
    do {                                                                    \
        vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,  \
            66, 71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17}; \
        {                                                                   \
            for (auto sz : sizes) {                                         \
                TEST_KNOWN_INPUT(sz, COMP_FUNC, DECOMP_FUNC);               \
            }                                                               \
        }                                                                   \
        {                                                                   \
            for (auto sz : sizes) {                                         \
                TEST_ZEROS(sz, COMP_FUNC, DECOMP_FUNC);                     \
            }                                                               \
        }                                                                   \
        {                                                                   \
            for (auto sz : sizes) {                                         \
                TEST_FUZZ(sz, COMP_FUNC, DECOMP_FUNC);                      \
            }                                                               \
        }                                                                   \
        {                                                                   \
            TEST_FUZZ(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);             \
        }                                                                   \
    } while (0);


#endif /* compress_testing_h */
