//
//  test_predict.cpp
//  Compress
//
//  Created by DB on 11/4/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include "catch.hpp"
//#include "eigen/Eigen"

#include "compress_testing.hpp"

// #include "array_utils.hpp"
//#include "sprintz2.h"
// #include "bitpack.h"
#include "sprintz_xff.h"
#include "testing_utils.hpp"
#include "debug_utils.hpp" // TODO rm


// ============================================================ sprintz predict

TEST_CASE("xff_rowmajor_8b (with compression)", "[rowmajor][xff]") {
    printf("executing rowmajor compress xff test\n");

//     int ndims = 8;
//     auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
            // return encode_xff_rowmajor(src, (uint32_t)len, dest, ndims);
            return compress8b_rowmajor_xff(src, (uint32_t)len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return decompress8b_rowmajor_xff(src, dest);
        };

        // TEST_SQUARES_INPUT(7, comp, decomp);
        // TEST_SQUARES_INPUT(16, comp, decomp);
//        TEST_SQUARES_INPUT(16 * ndims, comp, decomp);
       // TEST_SQUARES_INPUT(24 * ndims, comp, decomp);
//                 TEST_SQUARES_INPUT(ndims * 8, comp, decomp);
        //         TEST_SIMPLE_INPUTS(ndims * 2, comp, decomp);
        //         TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
//         TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
        TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
       // TEST_COMP_DECOMP_PAIR(comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 19, comp, decomp);
        // TEST_SIMPLE_INPUTS(ndims * 19, comp, decomp);
//        TEST_FUZZ(ndims * 16, comp, decomp);

//         #define COMP_FUNC comp
//         #define DECOMP_FUNC decomp

//         auto SZ = ndims * 16;
//         Vec_u8 raw(SZ);
//         {
//             for (int i = 0; i < SZ; i++) {
// //                 raw(i) = (i % 2) ? (i + 64) % 128 : 0;
//                 raw(i) = (i % ndims) + (i / ndims)*(i / ndims);
//                 // for (int i = 0; i < SZ; i++) { raw(i) = (i % 6) * (i % 6); }
//             }
//             TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);
//         }

        // vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,
        //     66, 71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17};
        // {
        //     for (auto sz : sizes) {
        //         CAPTURE(sz);
        //         TEST_KNOWN_INPUT(sz, COMP_FUNC, DECOMP_FUNC);
        //     }
        // }
        // {
        //     for (auto sz : sizes) {
        //         TEST_ZEROS(sz, COMP_FUNC, DECOMP_FUNC);
        //     }
        // }
        // {
        //     for (auto sz : sizes) {
        //         TEST_FUZZ(sz, COMP_FUNC, DECOMP_FUNC);
        //     }
        // }
        // {
        //     TEST_FUZZ(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);
        // }
        // {
        //     TEST_SPARSE(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);
        // }
    }
}

// #undef TEST_SPARSE
// #define TEST_SPARSE(SZ, COMP_FUNC, DECOMP_FUNC)                             \
//     do {                                                                    \
//     srand(123);                                                             \
//     Vec_u8 orig((SZ));                                                      \
//     Vec_u8 raw((SZ));                                                       \
//     orig.setRandom();                                                       \
//     {                                                                       \
//         raw = orig / 250;                                                   \
//         TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
//     }                                                                       \
//     } while(0);


TEST_CASE("xff_rle_rowmajor_lowdim_8b (with compression)",
    "[rowmajor][xff][rle][lowdim][8b]")
{
    printf("executing rowmajor compress xff + rle lowdim 8b test\n");
    TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_xff_rle_lowdim_8b, decompress_rowmajor_xff_rle_lowdim_8b, 1, 4);
    // TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_xff_rle_lowdim_8b, decompress_rowmajor_xff_rle_lowdim_8b, 1, 2);
}

TEST_CASE("xff_rle_rowmajor_lowdim_16b (with compression)",
    "[rowmajor][xff][rle][lowdim][16b]")
{
    printf("executing rowmajor compress xff + rle lowdim 16b test\n");
    TEST_CODEC_NDIMS_RANGE(2, compress_rowmajor_xff_rle_lowdim_16b, decompress_rowmajor_xff_rle_lowdim_16b, 1, 2);
    // // int ndims = 1;
    // // auto ndims_list = ar::range(ndims, ndims + 1);
    // auto ndims_list = ar::range(1, 2 + 1);
    // for (auto _ndims : ndims_list) {
    //     auto ndims = (uint16_t)_ndims;
    //     printf("---- ndims = %d\n", ndims);
    //     CAPTURE(ndims);
    //     auto comp = [ndims](const uint16_t* src, size_t len, int16_t* dest) {
    //         return compress_rowmajor_xff_rle_lowdim_16b(src, (uint32_t)len, dest, ndims);
    //     };
    //     auto decomp = [](int16_t* src, uint16_t* dest) {
    //         return decompress_rowmajor_xff_rle_lowdim_16b(src, dest);
    //     };
    //     test_codec<2>(comp, decomp);

    //     // static const int ElemSz = 2;
    //     // uint32_t sz = 128;
    //     // using UVec = typename elemsize_traits<ElemSz>::uvec_t;
    //     // uint32_t denominator_shift = 8 * (ElemSz - 1);
    //     // UVec orig(sz);
    //     // UVec raw(sz);
    //     // srand(12345);
    //     // orig.setRandom();
    //     // // raw = orig / (193 << denominator_shift);
    //     // // test_compressor<ElemSz>(raw, comp, decomp, "sparse 56/256");
    //     // raw = orig / (250 << denominator_shift);
    //     // test_compressor<ElemSz>(raw, comp, decomp, "sparse 6/256");
    //     // // raw = orig / (254 << denominator_shift);
    //     // // test_compressor<ElemSz>(raw, comp, decomp, "sparse 2/256");
    // }
}

TEST_CASE("xff_rle_rowmajor_8b (with compression)",
    "[rowmajor][xff][rle][8b][dbg]")
{
    printf("executing rowmajor compress xff + rle test\n");
     TEST_CODEC_MANY_NDIMS_8b(compress_rowmajor_xff_rle_8b, decompress_rowmajor_xff_rle_8b);
//    TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_xff_rle_8b, decompress_rowmajor_xff_rle_8b, 1, 5);
    // TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_xff_rle_8b, decompress_rowmajor_xff_rle_8b, 1, 1);
}

TEST_CASE("xff_rle_rowmajor_16b (with compression)",
    "[rowmajor][xff][rle][16b][dbg]")
{
    printf("executing rowmajor compress xff + rle 16b test\n");
    // int ndims = 1;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](const uint16_t* src, size_t len, int16_t* dest) {
            return compress_rowmajor_xff_rle_16b(src, (uint32_t)len, dest, ndims);
//            return compress8b_rowmajor_xff(src, (uint32_t)len, dest, ndims);
        };
        auto decomp = [](int16_t* src, uint16_t* dest) {
            return decompress_rowmajor_xff_rle_16b(src, dest);
//            return decompress8b_rowmajor_xff(src, dest);
        };
        test_codec<2>(comp, decomp);

        // uint32_t sz = 128;
        // static const int ElemSz = 2;
        // using UVec = typename elemsize_traits<ElemSz>::uvec_t;
        // uint32_t denominator_shift = 8 * (ElemSz - 1);
        // UVec orig(sz);
        // UVec raw(sz);
        // srand(12345);
        // orig.setRandom();
        // // raw = orig / (193 << denominator_shift);
        // // test_compressor<ElemSz>(raw, comp, decomp, "sparse 56/256");
        // // raw = orig / (250 << denominator_shift);
        // // test_compressor<ElemSz>(raw, comp, decomp, "sparse 6/256");
        // raw = orig / (254 << denominator_shift);
        // test_compressor<ElemSz>(raw, comp, decomp, "sparse 2/256");

        // auto SZ = 1024;
        // srand(123);
        // Vec_u16 raw(SZ);
        // {
        //     for (int i = 0; i < SZ; i++) {
        //         // raw(i) = i % 64;
        //         // raw(i) = (i % 2) ? (i + 64) % 128 : 0;
        //         // raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);
        //     //     // raw(i) = 128;
        //         // raw(i) = (i % 2) ? (i + 64) % 128 : 72;
        //         // raw(i) = (i % 2) ? (i * 512) % 65536 : 64;
        //         // raw(i) = (i % 2) ? (i * 4096) % 65536 : 64;
        //         // raw(i) = (i % 2) ? 32768 : 64;
        //     }
        //     raw.setRandom();
        //     raw /= 4;

        //     int start_idx = 0;
        //     // int end_idx = start_idx + SZ;
        //     // int start_idx = 20;
        //     int end_idx = start_idx + 296;
        //     int new_len = end_idx - start_idx;
        //     Vec_u16 raw2(new_len);
        //     for (int i = 0; i < new_len; i++) {
        //         raw2(i) = raw(i + start_idx);
        //     }

        //     // test_compressor<2>(raw, comp, decomp, "debug test", true);
        //     test_compressor<2>(raw2, comp, decomp, "debug test");
        //     // test_compressor<2>(raw, comp, decomp, "debug test");
        // }
    }
}
