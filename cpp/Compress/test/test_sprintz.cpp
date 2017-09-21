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
#include "timing_utils.hpp"

#include "debug_utils.hpp" // TODO rm

template<class T> using Vec = Eigen::Array<T, Eigen::Dynamic, 1>;
using Vec_i8 = Vec<int8_t>;
using Vec_u8 = Vec<uint8_t>;
using Vec_i16 = Vec<int16_t>;
using Vec_u16 = Vec<uint16_t>;


template<class T>
void _set_random_bits(T* dest, size_t size, int max_val) {
    T val = static_cast<T>(max_val);
    for (int i = 0; i < size; i += 8) {
        int highest_idx = (i / 8) % 8;
        for (int j = i; j < i + 8; j++) {
            if (j == highest_idx || val == 0) {
                dest[j] = val;
            } else {
                if (val > 0) {
                    dest[j] = rand() % val;
                } else {
                    dest[j] = -(rand() % abs(val));
                }
            }
        }
    }
}


TEST_CASE("smoke test", "[sanity]") {
    int x = 0;
    REQUIRE(x == 0);
}

TEST_CASE("building blocks", "[sanity]") {
    uint16_t sz = 16;
    Vec_u8 raw(sz);
    raw.setRandom(); // random vals in [0, 255]

    Vec_i8 compressed(sz);
    Vec_u8 decompressed(sz);

    SECTION("naiveDelta") {
//        std::cout << raw.cast<uint16_t>();
        auto len = compress8b_naiveDelta(raw.data(), sz, compressed.data());
        REQUIRE(len == sz);
        auto len2 = decompress8b_naiveDelta(compressed.data(), len, decompressed.data());
        REQUIRE(len2 == sz);

        REQUIRE(ar::all_eq(raw, decompressed));
//        std::cout << raw.cast<int16_t>();
//        std::cout << "------\n";
//        std::cout << decompressed.cast<int16_t>();
    }

    SECTION("bitpack") {
        Vec_u8 compressed(sz);
//        for (uint8_t nbits = 1; nbits <= 1; nbits++) {
        for (uint8_t nbits = 1; nbits <= 8; nbits++) {
            CAPTURE(nbits);

            raw.setRandom();
            raw /= (1 << (8 - nbits));

            auto len = compress8b_bitpack(raw.data(), sz, compressed.data(), nbits);
            REQUIRE(len == (sz / 8) * nbits);

//            std::cout << "raw: " << raw.cast<uint16_t>();
//            std::cout << "comp: " << compressed.cast<uint16_t>();

            auto len2 = decompress8b_bitpack(compressed.data(), len, decompressed.data(), nbits);
            REQUIRE(len2 == sz);
//
//            std::cout << "raw: " << raw.cast<uint16_t>();
//            std::cout << "decomp: " << decompressed.cast<uint16_t>();

            REQUIRE(ar::all_eq(raw, decompressed));
        }
    }
}

 TEST_CASE("max_nbits_i16", "[bitpack]") {
     const uint16_t SIZE = 8;
     int16_t data[SIZE];

     SECTION("8bit values") {
         SECTION("nbits_simple") {
         for (int val = -128; val <= 127; val++) {
             // for (int val = -1; val <= -1; val++) {
                 CAPTURE(val);
//                 for (int i = 0; i < SIZE; i++) { data[i] = val; }
                 _set_random_bits(data, SIZE, val);
                 for (int i = 0; i < SIZE; i += 8) {
                     uint8_t nbits_simple = needed_nbits_i16x8_simple(data + i);
                     CAPTURE((int)nbits_simple);
                     REQUIRE(nbits_simple == NBITS_COST_I8[val]);
                 }
             }
         }
         SECTION("nbits simd") {
             for (int val = -128; val <= 127; val++) {
             // for (int val = -1; val <= -1; val++) {
                 CAPTURE(val);
//                 for (int i = 0; i < SIZE; i++) { data[i] = val; }
                 _set_random_bits(data, SIZE, val);
                 for (int i = 0; i < SIZE; i += 8) {
                     uint8_t nbits = needed_nbits_i16x8(data + i);
                     CAPTURE((int)nbits);
                     REQUIRE(nbits == NBITS_COST_I8[val]);
                 }
             }
         }
     }
     SECTION("all 16bit values") {
        // for (int val = -1; val <= -1; val++) {
        for (int val = -32768; val <= 32767; val++) {
//            for (int i = 0; i < SIZE; i++) { data[i] = val; }
            _set_random_bits(data, SIZE, val);
            for (int i = 0; i < SIZE; i += 8) {
                CAPTURE(val);
                uint8_t nbits = needed_nbits_i16x8(data + i);
                uint8_t nbits_simple = needed_nbits_i16x8_simple(data + i);
                CAPTURE((int)nbits);
                CAPTURE((int)nbits_simple);
                if (nbits_simple != nbits) {
                    printf("val=%d) nbits, nbits_simple: %d, %d\n", val, nbits, nbits_simple);
                }
                REQUIRE(nbits_simple == nbits);
            }
        }
    }
 }

TEST_CASE("max_nbits_i8", "[bitpack]") {
   const uint16_t SIZE = 8 * 8;
   int8_t data[SIZE];
   // Vec_i8 raw(sz);
   // raw.setRandom(); // random vals in [0, 255]

    SECTION("nbits_simple") {
        srand(123);
        for (int val = -128; val <= 127; val++) {
            CAPTURE(val);
            _set_random_bits(data, SIZE, val);
            for (int i = 0; i < SIZE / 8; i += 8) {
                uint8_t nbits_simple = needed_nbits_i8x8_simple(data + i);
                REQUIRE(nbits_simple == NBITS_COST_I8[val]);
            }
        }
    }
    SECTION("nbits_simd") {
       for (int val = -128; val <= 127; val++) {
           CAPTURE(val);
//           for (int i = 0; i < SIZE; i++) { data[i] = val; }
           _set_random_bits(data, SIZE, val);
           for (int i = 0; i < SIZE / 8; i += 8) {
               uint8_t nbits = needed_nbits_i8x8(data + i);
               CAPTURE((int)nbits);
               CAPTURE((int)NBITS_COST_I8[val]);
               REQUIRE(nbits == NBITS_COST_I8[val]);
           }
       }
   }
}

TEST_CASE("profile_bitpack_u8", "[profile][bitpack]") {
    uint64_t sz = 256 * 1024 * 1024;
    Vec_u8 raw_orig(sz);
    Vec_u8 raw(sz);
    raw.setRandom(); // random vals in [0, 255]

    Vec_u8 compressed(sz);
    Vec_u8 decompressed(sz);
    for (uint8_t nbits = 1; nbits <= 8; nbits++) {
        std::cout << "---- nbits: " << (uint16_t)nbits << "\n";
//        raw.setRandom();
        raw = raw_orig / (1 << (8 - nbits));

        uint64_t len = 0, len2 = 0;
        {
//            cputime_t _tstart(timeNow());
            volatile PrintTimer t("compress");
            len = compress8b_bitpack(raw.data(), sz, compressed.data(), nbits);
//            auto elapsed = durationMs(_tstart, timeNow());
//            std::cout << "compress time:\t" << elapsed << "\tms\n";
        }
        if (len != (sz / 8) * nbits) { std::cout << "compression error!\n"; }
//        REQUIRE(len == (sz / 8) * nbits);

        {
            volatile PrintTimer t("decompress");
            len2 = decompress8b_bitpack(compressed.data(), len, decompressed.data(), nbits);
        }
        if (len2 != sz) { std::cout << "decompresion error!\n"; }
//        REQUIRE(len2 == sz);
    }
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


#define TEST_COMPRESSOR(COMP_FUNC, DECOMP_FUNC)                         \
    Vec_i8 compressed(sz * 2 + 16);                                     \
    Vec_u8 decompressed(sz);                                            \
    compressed.setZero();                                               \
    decompressed.setZero();                                             \
    auto len = COMP_FUNC(raw.data(), sz, compressed.data());            \
    len = DECOMP_FUNC(compressed.data(), decompressed.data());          \
    CAPTURE(sz);                                                        \
    REQUIRE(ar::all_eq(raw, decompressed));

//    std::cout << "decompressed size: " << decompressed.size() << "\n";  \
//    std::cout << decompressed.cast<int>() << "\n";  \
//    REQUIRE(ar::all_eq(raw, decompressed));


void _test_delta_8_simple_known_input(int64_t sz) {
    Vec_u8 raw(sz);
    for (int i = 0; i < sz; i++) {
        raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);
    }
    
    TEST_COMPRESSOR(compress8b_delta_simple, decompress8b_delta_simple);
}

TEST_CASE("delta_8b_simple_known_input", "[delta][bitpack]") {
    vector<int64_t> sizes {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 66, 72,
        127, 128, 129, 4096, 4096 + 17};
    for (auto sz : sizes) {
        _test_delta_8_simple_known_input(sz);
    }
}

void _test_delta_8_known_input(int64_t sz) {
    Vec_u8 raw(sz);
    for (int i = 0; i < sz; i++) {
        raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);
    }
    TEST_COMPRESSOR(compress8b_delta, decompress8b_delta);
//    
//    Vec_i8 compressed(sz * 2);
//    Vec_u8 decompressed(sz);
//    compressed.setZero();
//    decompressed.setZero();
//    
//    //    dump_16B_aligned(raw.data());
//    //    dump_16B_aligned(raw.data() + 16);
//    
//    auto len = compress8b_delta(raw.data(), sz, compressed.data());
//    
//    printf("compressed data (ignoring initial 8B) (length=%lld):\n", len);
//    for (int i = 8; i <= len - 16; i += 16) {
//        dump_16B_aligned(compressed.data() + i);
//    }
//
//    len = decompress8b_delta(compressed.data(), sz, decompressed.data());
////    
//    printf("decompressed data (length=%lld):\n", len);
//    for (int i = 64; i <= sz - 16; i += 16) {
//        dump_16B_aligned(decompressed.data() + i);
//    }
//
//    CAPTURE(sz);
//    //    REQUIRE(ar::all_eq(raw.data(), decompressed.data(), 64));
//    REQUIRE(ar::all_eq(raw, decompressed));
}

void _test_delta_8_fuzz(int64_t sz) {
    srand(123);
    Vec_u8 raw(sz);
    raw.setRandom();
    {
        TEST_COMPRESSOR(compress8b_delta, decompress8b_delta);
    }
    raw /= 2;
    {
        TEST_COMPRESSOR(compress8b_delta, decompress8b_delta);
    }
    raw /= 2;
    {
        TEST_COMPRESSOR(compress8b_delta, decompress8b_delta);
    }
    raw /= 2;
    {
        TEST_COMPRESSOR(compress8b_delta, decompress8b_delta);
    }
    raw /= 2;
    {
        TEST_COMPRESSOR(compress8b_delta, decompress8b_delta);
    }
    raw /= 8;
    {
        TEST_COMPRESSOR(compress8b_delta, decompress8b_delta);
    }
}

void _test_delta_zeros(int64_t sz) {
    Vec_u8 raw(sz);
    raw.setZero();
    TEST_COMPRESSOR(compress8b_delta, decompress8b_delta);
}


TEST_CASE("delta_8b", "[delta][bitpack]") {
//    vector<int64_t> sizes {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 66, 72,
//        127, 128, 129, 4096, 4096 + 17};
    vector<int64_t> sizes {72};
    SECTION("known input") {
        for (auto sz : sizes) { _test_delta_8_known_input(sz); }
    }
    SECTION("fuzz_multiple_sizes") {
        for (auto sz : sizes) { _test_delta_8_fuzz(sz); }
    }
    SECTION("zeros") {
        for (auto sz : sizes) { _test_delta_zeros(sz); }
    }
    SECTION("long fuzz") {
        _test_delta_8_fuzz(1024 * 1024 + 7);
    }
}

// TODO replace near-duplicate funcs with one templated func
void _test_doubledelta_8_known_input(int64_t sz) {
    Vec_u8 raw(sz);
    for (int i = 0; i < sz; i++) {
        raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);
    }
    
    Vec_i8 compressed(sz * 2);
    Vec_u8 decompressed(sz);
    compressed.setZero();
    decompressed.setZero();
    
    auto len = compress8b_doubledelta(raw.data(), sz, compressed.data());
    
//    printf("compressed data (ignoring initial 8B) (length=%lld):\n", len);
//    for (int i = 8; i <= len - 16; i += 16) {
//        dump_16B_aligned(compressed.data() + i);
//    }
    
    len = decompress8b_doubledelta(compressed.data(), decompressed.data());
    REQUIRE(ar::all_eq(raw, decompressed));
}
TEST_CASE("doubledelta_8b_known_input", "[delta][bitpack]") {
    vector<int64_t> sizes {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 66, 72,
        127, 128, 129, 4096, 4096 + 17};
//    vector<int64_t> sizes {64};
    for (auto sz : sizes) {
        _test_doubledelta_8_known_input(sz);
    }
}

void _test_dyndelta_8_known_input(int64_t sz) {
    Vec_u8 raw(sz);
    for (int i = 0; i < sz; i++) {
        raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);
    }
    
    Vec_i8 compressed(sz * 2);
    Vec_u8 decompressed(sz);
    compressed.setZero();
    decompressed.setZero();
    
    auto len = compress8b_dyndelta(raw.data(), sz, compressed.data());
    
//    printf("compressed data (ignoring initial 8B) (length=%lld):\n", len);
//    for (int i = 8; i <= len - 16; i += 16) {
//        dump_16B_aligned(compressed.data() + i);
//    }
    
    len = decompress8b_dyndelta(compressed.data(), decompressed.data());
    
//    printf("decompressed data (length=%lld):\n", len);
//    for (int i = 0; i <= sz - 16; i += 16) {
//        dump_16B_aligned(decompressed.data() + i);
//    }

//    CAPTURE(sz);
    REQUIRE(ar::all_eq(raw, decompressed));
}
TEST_CASE("dyndelta_8b_known_input", "[delta][bitpack]") {
    vector<int64_t> sizes {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 66, 72,
        127, 128, 129, 4096, 4096 + 17};
//    vector<int64_t> sizes {64};
    for (auto sz : sizes) {
        _test_dyndelta_8_known_input(sz);
    }
}
