//
//  test_bitpack.cpp
//  Compress
//
//  Created by DB on 9/21/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include "bitpack.h"

#include <stdio.h>
#include "catch.hpp"
#include "eigen/Eigen"

#include "array_utils.hpp"
#include "debug_utils.hpp"
#include "testing_utils.hpp"
#include "timing_utils.hpp"


TEST_CASE("building blocks", "[bitpack]") {
    uint16_t sz = 16;
    Vec_u8 raw(sz);
    raw.setRandom(); // random vals in [0, 255]

    Vec_i8 compressed(sz);
    Vec_u8 decompressed(sz);

    SECTION("bitpack") {
        Vec_u8 compressed(sz);
//        for (uint8_t nbits = 1; nbits <= 1; nbits++) {
        for (uint8_t nbits = 1; nbits <= 8; nbits++) {
            CAPTURE(nbits);

            raw.setRandom();
            raw /= (1 << (8 - nbits));

            auto len = compress8b_bitpack(raw.data(), sz, compressed.data(), nbits);
            REQUIRE(len == (sz * nbits)/ 8);

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

TEST_CASE("zigzag_8b", "[bitpack][zigzag]") {
    SECTION("scalar") {
        for (int val = -128; val <= 127; val++) {
            REQUIRE(val == zigzag_decode_i8(zigzag_encode_i8((int8_t)val)));
        }
    }
    SECTION("scalar macro") {
        for (int val = -128; val <= 127; val++) {
            REQUIRE(val == ZIGZAG_DECODE_SCALAR(ZIGZAG_ENCODE_SCALAR((int8_t)val)));
        }
    }
    SECTION("mm256_epi8 constant vectors") {
        static const int n = 32; // number of elements in a SIMD vector
        uint8_t buff[n];
        __m256i* store_ptr = (__m256i*)buff;
         for (int val = -128; val <= 127; val++) {
            __m256i v = _mm256_set1_epi8((int8_t)val);
            auto encoded = mm256_zigzag_encode_epi8(v);
//            printf("encoded: "); dump_m256i(encoded);
            auto decoded = mm256_zigzag_decode_epi8(encoded);
//            printf("decoded: "); dump_m256i(decoded);
            auto same = _mm256_cmpeq_epi8(v, decoded);
            _mm256_storeu_si256(store_ptr, same);
            CAPTURE(val);
            for (int i = 0; i < n; i++) {
                REQUIRE(buff[i] == 0xff);
            }
        }
    }
    SECTION("mm256_epi8 random vectors") {
        static const int n = 32; // number of elements in a SIMD vector
        static const size_t sz = 256 * 1024;

        using ByteMat = Eigen::Matrix<uint8_t, Eigen::Dynamic,
            Eigen::Dynamic, Eigen::RowMajor>;
        ByteMat X(sz, n);
        ByteMat X_out(sz, n);
        X.setRandom(); // random vals in [0, 255]

        // NOTE: 8x faster to use pointers directly than to use X.row(i).data()
        for (size_t i = 0; i < sz; i++) {
            auto inptr = X.data() + n * i;
            auto v = _mm256_loadu_si256((const __m256i*)inptr);
            auto encoded = mm256_zigzag_encode_epi8(v);
            auto decoded = mm256_zigzag_decode_epi8(encoded);

            auto same = _mm256_cmpeq_epi8(v, decoded);
            __m256i* store_ptr = (__m256i*)(X_out.data() + n * i);
            _mm256_storeu_si256(store_ptr, same);
        }

        REQUIRE(X_out.minCoeff() == 255); // all comparisons true (ones byte)
    }
}

TEST_CASE("zigzag_16b", "[bitpack][zigzag]") {
    const int16_t minval =  -128 * 256;
    const int16_t maxval =  127 * 256;
    SECTION("scalar macro") {
        for (int val = -minval; val <= maxval; val++) {
            REQUIRE(val == ZIGZAG_DECODE_SCALAR(ZIGZAG_ENCODE_SCALAR((int16_t)val)));
        }
    }
    SECTION("mm256_epi16 constant vectors") {
        static const int n = 16; // number of elements in a SIMD vector
        uint16_t buff[n];
        __m256i* store_ptr = (__m256i*)buff;
         for (int val = -128; val <= 127; val++) {
            __m256i v = _mm256_set1_epi16((int8_t)val);
            auto encoded = mm256_zigzag_encode_epi16(v);
//            printf("encoded: "); dump_m256i(encoded);
            auto decoded = mm256_zigzag_decode_epi16(encoded);
//            printf("decoded: "); dump_m256i(decoded);
            auto same = _mm256_cmpeq_epi16(v, decoded);
            _mm256_storeu_si256(store_ptr, same);
            CAPTURE(val);
            for (int i = 0; i < n; i++) {
                REQUIRE(buff[i] == 0xffff);
            }
        }
    }
    SECTION("mm256_epi16 random vectors") {
        static const int n = 16; // number of elements in a SIMD vector
        static const size_t sz = 256 * 1024;

        using DataMat = Eigen::Matrix<uint16_t, Eigen::Dynamic,
            Eigen::Dynamic, Eigen::RowMajor>;
        DataMat X(sz, n);
        DataMat X_out(sz, n);
        X.setRandom(); // random vals in [0, 255]

        // NOTE: 8x faster to use pointers directly than to use X.row(i).data()
        for (size_t i = 0; i < sz; i++) {
            auto inptr = X.data() + n * i;
            auto v = _mm256_loadu_si256((const __m256i*)inptr);
            auto encoded = mm256_zigzag_encode_epi16(v);
            auto decoded = mm256_zigzag_decode_epi16(encoded);

            auto same = _mm256_cmpeq_epi16(v, decoded);
            __m256i* store_ptr = (__m256i*)(X_out.data() + n * i);
            _mm256_storeu_si256(store_ptr, same);
        }

        REQUIRE(X_out.minCoeff() == 0xffff); // all comparisons true (ones byte)
    }
}

TEST_CASE("profile zigzag_8b", "[bitpack][zigzag][profile]") {
    using ByteMat = Eigen::Matrix<uint8_t, Eigen::Dynamic,
        Eigen::Dynamic, Eigen::RowMajor>;
    static const int n = 32; // number of elements in a SIMD vector
    static const size_t sz = 256 * 1024;

    SECTION("mm256_epi8 random vectors, not explicitly inlined") {
        ByteMat X(sz, n);
        ByteMat X_out(sz, n);
        X.setRandom(); // random vals in [0, 255]
        {
            volatile PrintTimer t("zigzag");
            for (size_t i = 0; i < sz; i++) {
                // auto inptr = X.row(i).data();
                auto inptr = X.data() + n * i; // 10x faster than prev line...
                auto v = _mm256_loadu_si256((const __m256i*)inptr);
                auto encoded = mm256_zigzag_encode_epi8(v);
                auto decoded = mm256_zigzag_decode_epi8(encoded);

                auto same = _mm256_cmpeq_epi8(v, decoded);
                // __m256i* store_ptr = (__m256i*)X_out.row(i).data();
                __m256i* store_ptr = (__m256i*)(X_out.data() + n * i);
                _mm256_storeu_si256(store_ptr, same);
            }
        }
        REQUIRE(X_out.minCoeff() == 255); // all comparisons true (ones byte)
    }

    SECTION("mm256_epi8 random vectors, explicitly inlined") {
        ByteMat X(sz, n);
        ByteMat X_out(sz, n);
        X.setRandom(); // random vals in [0, 255]
        {
            volatile PrintTimer t("zigzag inlined");
            static __m256i zeros = _mm256_setzero_si256();
            static __m256i ones = _mm256_set1_epi8(1);
            static __m256i high_bits_one = _mm256_set1_epi8(-128);

            for (size_t i = 0; i < sz; i++) {
                // auto inptr = X.row(i).data();
                auto inptr = X.data() + n * i;
                auto x = _mm256_loadu_si256((const __m256i*)inptr);

                // encode
                __m256i invert_mask = _mm256_cmpgt_epi8(zeros, x);
                __m256i shifted = _mm256_andnot_si256(ones, _mm256_slli_epi64(x, 1));
                __m256i encoded = _mm256_xor_si256(invert_mask, shifted);

                // decode
                __m256i shifted2 = _mm256_andnot_si256(
                    high_bits_one, _mm256_srli_epi64(encoded, 1));
                __m256i invert_mask2 = _mm256_cmpgt_epi8(
                    zeros, _mm256_slli_epi64(encoded, 7));
                __m256i decoded = _mm256_xor_si256(invert_mask2, shifted2);

                auto same = _mm256_cmpeq_epi8(x, decoded);
                // __m256i* store_ptr = (__m256i*)X_out.row(i).data();
                __m256i* store_ptr = (__m256i*)(X_out.data() + n * i);
                _mm256_storeu_si256(store_ptr, same);
            }
        }
        REQUIRE(X_out.minCoeff() == 255); // all comparisons true (ones byte)
    }
}

TEST_CASE("profile_bitpack_u8", "[profile][bitpack]") {
    uint64_t sz = 16 * 1024 * 1024;
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
       REQUIRE(len == (sz / 8) * nbits);

        {
            volatile PrintTimer t("decompress");
            len2 = decompress8b_bitpack(compressed.data(), len, decompressed.data(), nbits);
        }
        if (len2 != sz) { std::cout << "decompresion error!\n"; }
       REQUIRE(len2 == sz);
    }
}
