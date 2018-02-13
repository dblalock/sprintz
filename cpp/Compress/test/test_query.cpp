//
//  test_query.cpp
//  Compress
//
//  Created by DB on 2/12/18.
//  Copyright © 2018 D Blalock. All rights reserved.
//

#include "sprintz_delta.h"
#include "sprintz_xff.h"

#include "catch.hpp"
#include "compress_testing.hpp"
#include "testing_utils.hpp"

TEST_CASE("query rowmajor delta rle 8b", "[rowmajor][delta][rle][8b][query]") {
    printf("executing rowmajor delta rle 8b query test\n");
    auto query_func = [](int8_t* src, uint8_t* dest) {
        QueryParams qp;
        return query_rowmajor_delta_rle_8b(src, dest, qp);
    };
    uint16_t ndims = 3;
    auto ndims_list = ar::range(1, ndims + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](const uint8_t* src, size_t len, int8_t* dest) {
            return compress_rowmajor_delta_rle_8b(src, (uint32_t)len, dest, ndims);
        };
//        auto decomp = [](int16_t* src, uint16_t* dest) {
//            return query_rowmajor_delta_rle_16b(src, dest);
//        };
        
        test_codec<1>(comp, query_func);
    }
//    TEST_CODEC_MANY_NDIMS_8b(compress_rowmajor_delta_rle_8b, query_rowmajor_delta_rle_8b);
//     TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_delta_rle_8b, query_func, 1, 3);
//         TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_delta_rle_8b, decompress_rowmajor_delta_rle_8b, 1, 3);
}

// TODO uncomment
//
TEST_CASE("query rowmajor delta rle 16b", "[rowmajor][delta][rle][16b][query]") {
    printf("executing rowmajor delta rle 16b query test\n");
    // TEST_CODEC_MANY_NDIMS_16b(compress_rowmajor_16b, decompress_rowmajor_16b);

     uint16_t ndims = 3;
    auto ndims_list = ar::range(1, ndims + 1);
//     auto ndims_list = ar::range(ndims, ndims + 1);
//    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](const uint16_t* src, size_t len, int16_t* dest) {
            return compress_rowmajor_delta_rle_16b(src, (uint32_t)len, dest, ndims);
        };
        auto decomp = [](int16_t* src, uint16_t* dest) {
            QueryParams qp;
            return query_rowmajor_delta_rle_16b(src, dest, qp);
        };

        test_codec<2>(comp, decomp);
    }
}

TEST_CASE("xff rle rowmajor 8b query (with compression)",
          "[rowmajor][xff][rle][8b][query]")
{
    printf("executing rowmajor compress xff + rle query test\n");
     TEST_CODEC_MANY_NDIMS_8b(compress_rowmajor_xff_rle_8b, query_rowmajor_xff_rle_8b);
//    TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_xff_rle_8b, decompress_rowmajor_xff_rle_8b, 1, 5);
    // TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_xff_rle_8b, decompress_rowmajor_xff_rle_8b, 1, 1);
}

TEST_CASE("xff rle rowmajor 16b (with compression)",
          "[rowmajor][xff][rle][16b][query]")
{
    printf("executing rowmajor compress xff + rle 16b query test\n");
    // int ndims = 1;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](const uint16_t* src, size_t len, int16_t* dest) {
            return compress_rowmajor_xff_rle_16b(src, (uint32_t)len, dest, ndims);
        };
        auto decomp = [](int16_t* src, uint16_t* dest) {
            return query_rowmajor_xff_rle_16b(src, dest);
        };
        test_codec<2>(comp, decomp);
    }
}
