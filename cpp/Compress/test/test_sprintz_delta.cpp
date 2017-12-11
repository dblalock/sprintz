//
//  test_sprintz2.cpp
//  Compress
//
//  Created by DB on 9/28/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>
#include <vector>

#include "catch.hpp"
#include "eigen/Eigen"

#include "sprintz_delta.h"
#include "util.h" // TODO new test file for these

#include "array_utils.hpp"
#include "compress_testing.hpp"
#include "debug_utils.hpp"
#include "test_utils.hpp"


TEST_CASE("mm256_shuffle_epi8_to_epi16", "[util]") {
    using dtype = uint8_t;

    static const __m256i nbits_to_mask_16b_low = _mm256_setr_epi8(
        0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff,   // 0-8
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,               // 9-15
        0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff,   // 0-8
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);              // 9-15
    static const __m256i nbits_to_mask_16b_high = _mm256_setr_epi8(
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,         // 0-7
        0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0xff,         // 8-15
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,         // 0-7
        0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0xff);        // 8-15

    static const __m256i idxs = _mm256_setr_epi8(
        15, 14, 13, 12, 11, 10, 9,  8,
        7,  6,  5,  4,  3,  2,  1,  0,
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15);

    std::vector<uint16_t> ans = {
        0xffff, 0x3fff, 0x1fff, 0x0fff, 0x07ff, 0x03ff, 0x01ff, 0x00ff,
        0x7f,   0x3f,   0x1f,   0xf,    0x7,    0x3,    0x1,    0x0,
        0x0,    0x1,    0x3,    0x7,    0xf,    0x1f,   0x3f,   0x7f,
        0xff,   0x1ff,  0x3ff,  0x7ff,  0xfff,  0x1fff, 0x3fff, 0xffff
    };
    std::vector<uint16_t> ret(ans.size());

    __m256i masks0 = _mm256_undefined_si256();
    __m256i masks1 = _mm256_undefined_si256();
    mm256_shuffle_epi8_to_epi16(nbits_to_mask_16b_low, nbits_to_mask_16b_high,
                                idxs, masks0, masks1);

    _mm256_storeu_si256((__m256i*)ret.data(), masks0);
    _mm256_storeu_si256((__m256i*)(ret.data() + 16), masks1);

    bool eq = ar::all_eq(ans, ret);
    if (!eq) {
        printf("answer:\n");
        dump_elements(ans.data(), 32);
        printf("returned data:\n");
        dump_elements(ret.data(), 32);
    }
    REQUIRE(eq);
}

TEST_CASE("compress rowmajor bitpack 8b", "[rowmajor][bitpack][8b]") {
    printf("executing rowmajor bitpack-only test\n");

    TEST_CODEC_MANY_NDIMS_8b(compress_rowmajor_8b, decompress_rowmajor_8b);
    // TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_8b, decompress_rowmajor_8b, 1, 9);
    // TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_8b, decompress_rowmajor_8b, 1, 5);
}

TEST_CASE("compress rowmajor bitpack 16b", "[rowmajor][bitpack][16b]") {
    printf("executing rowmajor bitpack-only 16b test\n");
    // TEST_CODEC_MANY_NDIMS_16b(compress_rowmajor_16b, decompress_rowmajor_16b);

    // uint16_t ndims = 4;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](const uint16_t* src, size_t len, int16_t* dest) {
            return compress_rowmajor_16b(src, (uint32_t)len, dest, ndims);
        };
        auto decomp = [](int16_t* src, uint16_t* dest) {
            return decompress_rowmajor_16b(src, dest);
        };

        test_codec<2>(comp, decomp);
        // auto SZ = 128;
        // srand(123);
        // Vec_u16 raw(SZ);
        // {
        //     for (int i = 0; i < SZ; i++) {
        //         // raw(i) = (i % 2) ? (i + 64) % 128 : 0;
        //     //     raw(i) = (i % 64);
        //     //     // raw(i) = 128;
        //         // raw(i) = (i % 2) ? (i + 64) % 128 : 72;
        //         // raw(i) = (i % 2) ? (i * 1024) % 65536 : 64;
        //         raw(i) = (i % 2) ? 32768 : 64;
        //     }
        //     // raw.setRandom();

        //     // TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);
        //     test_compressor<2>(raw, comp, decomp, "debug test");
        // }
    }
}

void test_dset(DatasetName name, uint16_t ndims, int64_t limit_bytes=-1) {
    Dataset raw = read_dataset(name, limit_bytes);
    // ar::print(data.get(), 100, "msrc");
    auto comp = [ndims](uint8_t* src, uint32_t len, int8_t* dest) {
        return compress_rowmajor_8b(src, len, dest, ndims);
    };
    auto decomp = [](int8_t* src, uint8_t* dest) {
        return decompress_rowmajor_8b(src, dest);
    };
    printf("compressing %lld bytes of %d-dimensional data\n", raw.size(), ndims);
    TEST_COMPRESSOR((uint32_t)raw.size(), comp, decomp);
}

TEST_CASE("real datasets", "[rowmajor][dsets]") {

    SECTION("chlorine") { test_dset(DatasetName::CHLORINE, 1); }
    SECTION("msrc") { test_dset(DatasetName::MSRC, 80); }
    // SECTION("pamap") { test_dset(DatasetName::PAMAP, 31); }
    // SECTION("uci_gas") { test_dset(DatasetName::UCI_GAS, 18); }
    // SECTION("rand_0-63") {
    //     // auto ndims_list = ar::range(4, 5);
    //     auto ndims_list = ar::range(1, 65);
    //     for (auto _ndims : ndims_list) {
    //         auto ndims = (uint16_t)_ndims;
    //         printf("---- using ndims = %d\n", ndims);
    //         CAPTURE(ndims);
    //         test_dset(DatasetName::RAND_1M_0_63, 1);
    //     }
    // }

    // auto pamap = read_dataset(DatasetName::PAMAP, 1000);
    // ar::print(pamap.data(), 100, "pamap 1k");
    // auto uci_gas = read_dataset(DatasetName::PAMAP, 100);
    // ar::print(uci_gas.data(), 100, "uci_gas 100B");
}


// ============================================================ rowmajor delta

TEST_CASE("compress_rowmajor_delta 8b", "[rowmajor][delta][8b][dbg]") {
    printf("executing rowmajor delta 8b test\n");
    TEST_CODEC_MANY_NDIMS_8b(compress_rowmajor_delta_8b, decompress_rowmajor_delta_8b);
    // TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_delta_8b, decompress_rowmajor_delta_8b, 1, 41);
    // TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_delta_8b, decompress_rowmajor_delta_8b, 1, 3);
}

TEST_CASE("compress rowmajor delta 16b", "[rowmajor][delta][16b][dbg]") {
    printf("executing rowmajor delta 16b test\n");
    // TEST_CODEC_MANY_NDIMS_16b(compress_rowmajor_16b, decompress_rowmajor_16b);

    // uint16_t ndims = 17;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](const uint16_t* src, size_t len, int16_t* dest) {
            return compress_rowmajor_delta_16b(src, (uint32_t)len, dest, ndims);
        };
        auto decomp = [](int16_t* src, uint16_t* dest) {
            return decompress_rowmajor_delta_16b(src, dest);
        };

        test_codec<2>(comp, decomp);
        // auto SZ = ndims * 16;
        // // auto SZ = 400;
        // srand(123);
        // Vec_u16 raw(SZ);
        // {
        //     for (int i = 0; i < SZ; i++) {
        //         // raw(i) = i % 64;
        //         // raw(i) = (i % 2) ? (i + 64) % 128 : 0;
        //         raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);
        //     //     raw(i) = (i % 64);
        //     //     // raw(i) = 128;
        //         // raw(i) = (i % 2) ? (i + 64) % 128 : 72;
        //         // raw(i) = (i % 2) ? (i * 1024) % 65536 : 64;
        //         // raw(i) = (i % 2) ? 32768 : 64;
        //     }
        //     // raw.setRandom();

        //     // TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);
        //     // test_compressor<2>(raw, comp, decomp, "debug test", true);
        //     test_compressor<2>(raw, comp, decomp, "debug test");
        // }
    }
}

// ============================================================ rowmajor delta rle

TEST_CASE("compress8b_rowmajor_delta_rle", "[rowmajor][delta][rle]") {
    printf("executing rowmajor delta rle test\n");

    // int ndims = 64;
    // int ndims = 17;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    // auto ndims_list = ar::range(64, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](const uint8_t* src, size_t len, int8_t* dest) {
            return compress8b_rowmajor_delta_rle(src, len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return decompress8b_rowmajor_delta_rle(src, dest);
        };
        // test_codec<1>(comp, decomp);
        TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
    }
}

TEST_CASE("compress8b_rowmajor_delta_rle_lowdims", "[rowmajor][delta][rle]") {
    printf("executing rowmajor delta rle lowdim test\n");

    // int ndims = 1;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 4 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
            return compress8b_rowmajor_delta_rle_lowdim(src, len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return decompress8b_rowmajor_delta_rle_lowdim(src, dest);
        };

        // #define COMP_FUNC comp
        // #define DECOMP_FUNC decomp

        // size_t SZ = 128;
        // Vec_u8 raw(SZ);
        // {
        //     // for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }
        //     // for (int i = 0; i < SZ; i++) { raw(i) = (i % 8) * (i % 8); }
        //     for (int i = 0; i < SZ; i++) { raw(i) = (i % 6) * (i % 6); }
        //     // for (int i = 0; i < SZ; i++) { raw(i) = 6; }
        //     TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);
        // }

    // auto SZ = 128;
    // srand(1234);
    // Vec_u8 orig((SZ));
    // Vec_u8 raw((SZ));
    // orig.setRandom();
    // {
    //     raw = orig / 200;
    //     TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);
    // }
    // // {
    // //     raw = orig / 250;
    // //     TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);
    // // }
    // // {
    // //     raw = orig / 254;
    // //     TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);
    // // }

        // TEST_SQUARES_INPUT(128, comp, decomp);
        // TEST_SQUARES_INPUT(ndims * 16, comp, decomp);
        // TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
        // TEST_ZEROS(400 * 1024, comp, decomp);
        // TEST_ZEROS(128, comp, decomp);
        // TEST_ZEROS(4096, comp, decomp);
        // TEST_SPARSE(1 << 10, comp, decomp);
        TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
        // TEST_COMP_DECOMP_PAIR(comp, decomp);
    }
}
