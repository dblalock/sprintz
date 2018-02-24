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

// ================================================================ Helpers

// template<int ElemSz, class RawT, class CompF, class DecompF>
// void test_query(QueryParams qp, const RawT& raw, CompF&& f_comp,
template<int ElemSz, class CompF, class DecompF>
void test_query(QueryParams qp, CompF&& f_comp, DecompF&& f_decomp)
{
    vector<uint32_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,
        66, 71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17};
    uint16_t ndims = 80;
    auto ndims_list = ar::range(1, ndims + 1);

    using traits = elemsize_traits<ElemSz>;
    using uint_t = typename traits::uint_t;
    using UVec = typename traits::uvec_t;
    using IVec = typename traits::ivec_t;
    auto sz = sizes[sizes.size() - 1]; // assumes last size is largest
    IVec compressed(sz * 3/2 + 64);
    uint16_t decomp_padding = 64;
    UVec decompressed(sz + decomp_padding);
    compressed.setZero();
    decompressed.setZero();

    // auto f_comp2 = compress_rowmajor_delta_rle_8b;

    for (auto sz : sizes) {
        CAPTURE(ndims);
        UVec raw(sz);
        srand(123);
        raw.setRandom();
        for (auto ndims : ndims_list) {
            CAPTURE(ndims);
//            printf("---- ndims = %d\n", ndims);
            f_comp(raw.data(), sz, compressed.data(), ndims);
//            f_comp2(raw.data(), sz, compressed.data(), ndims);
            // compress_rowmajor_delta_rle_8b(raw.data(), sz, compressed.data(), ndims);
            f_decomp(compressed.data(), decompressed.data(), qp);
        }
    }
}

// ================================================================ Delta

TEST_CASE("query rowmajor delta rle 8b", "[rowmajor][delta][rle][8b][query]") {
    printf("executing rowmajor delta rle 8b query test\n");
    auto query_func = [](int8_t* src, uint8_t* dest) {
        QueryParams qp;
        qp.materialize = true;
//        qp.materialize = false; // this should, and does, fail
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

TEST_CASE("query delta noop 8b", "[delta][8b][query]") {
    printf("executing delta noop query 8b test\n");
    QueryParams qp;
    // why the heck does this work in a lambda, but not when passed as a
    // function pointer directly?
    auto f_comp = [](const uint8_t* src, uint32_t len, int8_t* dest, uint16_t ndims) {
        return compress_rowmajor_delta_rle_8b(src, len, dest, ndims);
    };
    test_query<1>(qp, f_comp, query_rowmajor_delta_rle_8b);
}
TEST_CASE("query delta reduce max 8b", "[delta][8b][query]") {
    printf("executing delta reduce max query 8b test\n");
    QueryParams qp;
    qp.op = REDUCE_MAX;
    // why the heck does this work in a lambda, but not when passed as a
    // function pointer directly?
    auto f_comp = [](const uint8_t* src, uint32_t len, int8_t* dest, uint16_t ndims) {
        return compress_rowmajor_delta_rle_8b(src, len, dest, ndims);
    };
    test_query<1>(qp, f_comp, query_rowmajor_delta_rle_8b);
}
TEST_CASE("query delta reduce sum 16b", "[delta][8b][query]") {
    printf("executing delta reduce sum query 16b test\n");
    QueryParams qp;
    qp.op = REDUCE_SUM;
    // why the heck does this work in a lambda, but not when passed as a
    // function pointer directly?
    auto f_comp = [](const uint16_t* src, uint32_t len, int16_t* dest, uint16_t ndims) {
        return compress_rowmajor_delta_rle_16b(src, len, dest, ndims);
    };
    test_query<2>(qp, f_comp, query_rowmajor_delta_rle_16b);
}

// ================================================================ XFF

// TEST_CASE("xff rle rowmajor 8b query (with compression)",
//           "[rowmajor][xff][rle][8b][query]")
// {
//     printf("executing rowmajor compress xff + rle query test\n");
// //     TEST_CODEC_MANY_NDIMS_8b(compress_rowmajor_xff_rle_8b, query_rowmajor_xff_rle_8b);
//     TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_xff_rle_8b, query_rowmajor_xff_rle_8b, 1, 4);
//     // TEST_CODEC_NDIMS_RANGE(1, compress_rowmajor_xff_rle_8b, decompress_rowmajor_xff_rle_8b, 1, 1);
// }

// TEST_CASE("xff rle rowmajor 16b (with compression)",
//           "[rowmajor][xff][rle][16b][query]")
// {
//     printf("executing rowmajor compress xff + rle 16b query test\n");
//      int ndims = 3;
//      auto ndims_list = ar::range(1, ndims + 1);
// //    auto ndims_list = ar::range(1, 129 + 1);
//     for (auto _ndims : ndims_list) {
//         auto ndims = (uint16_t)_ndims;
//         printf("---- ndims = %d\n", ndims);
//         CAPTURE(ndims);
//         auto comp = [ndims](const uint16_t* src, size_t len, int16_t* dest) {
//             return compress_rowmajor_xff_rle_16b(src, (uint32_t)len, dest, ndims);
//         };
//         auto decomp = [](int16_t* src, uint16_t* dest) {
//             return query_rowmajor_xff_rle_16b(src, dest);
//         };
//         test_codec<2>(comp, decomp);
//     }
// }


TEST_CASE("query rowmajor xff rle 8b", "[rowmajor][xff][rle][8b][query]") {
    printf("executing rowmajor xff rle 8b query test\n");
    auto query_func = [](int8_t* src, uint8_t* dest) {
        QueryParams qp;
        qp.materialize = true;
//        qp.materialize = false; // this should, and does, fail
        return query_rowmajor_xff_rle_8b(src, dest, qp);
    };
    uint16_t ndims = 3;
    auto ndims_list = ar::range(1, ndims + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](const uint8_t* src, size_t len, int8_t* dest) {
            return compress_rowmajor_xff_rle_8b(src, (uint32_t)len, dest, ndims);
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

// XXX this started failing; unsure if it's supposed to
//
//TEST_CASE("query rowmajor xff rle 16b", "[rowmajor][xff][rle][16b][query]") {
//    printf("executing rowmajor xff rle 16b query test\n");
//    // TEST_CODEC_MANY_NDIMS_16b(compress_rowmajor_16b, decompress_rowmajor_16b);
//
//     uint16_t ndims = 3;
//    auto ndims_list = ar::range(1, ndims + 1);
////     auto ndims_list = ar::range(ndims, ndims + 1);
////    auto ndims_list = ar::range(1, 129 + 1);
//    for (auto _ndims : ndims_list) {
//        auto ndims = (uint16_t)_ndims;
//        printf("---- ndims = %d\n", ndims);
//        CAPTURE(ndims);
//        auto comp = [ndims](const uint16_t* src, size_t len, int16_t* dest) {
//            return compress_rowmajor_xff_rle_16b(src, (uint32_t)len, dest, ndims);
//        };
//        auto decomp = [](int16_t* src, uint16_t* dest) {
//            QueryParams qp;
//            return query_rowmajor_xff_rle_16b(src, dest, qp);
//        };
//
//        test_codec<2>(comp, decomp);
//    }
//}


TEST_CASE("query xff noop 8b", "[xff][8b][query]") {
    printf("executing xff noop query 8b test\n");
    QueryParams qp;
    auto f_comp = [](const uint8_t* src, uint32_t len, int8_t* dest, uint16_t ndims) {
        return compress_rowmajor_xff_rle_8b(src, len, dest, ndims);
    };
    test_query<1>(qp, f_comp, query_rowmajor_xff_rle_8b);
}
TEST_CASE("query xff reduce max 8b", "[xff][8b][query]") {
    printf("executing xff reduce max query 8b test\n");
    QueryParams qp;
    qp.op = REDUCE_MAX;
    auto f_comp = [](const uint8_t* src, uint32_t len, int8_t* dest, uint16_t ndims) {
        return compress_rowmajor_xff_rle_8b(src, len, dest, ndims);
    };
    test_query<1>(qp, f_comp, query_rowmajor_xff_rle_8b);
}
TEST_CASE("query xff reduce sum 8b", "[xff][8b][query]") {
    printf("executing xff reduce sum query 8b test\n");
    QueryParams qp;
    qp.op = REDUCE_SUM;
    auto f_comp = [](const uint8_t* src, uint32_t len, int8_t* dest, uint16_t ndims) {
        return compress_rowmajor_xff_rle_8b(src, len, dest, ndims);
    };
    test_query<1>(qp, f_comp, query_rowmajor_xff_rle_8b);
}
TEST_CASE("query xff reduce max 16b", "[xff][8b][query]") {
    printf("executing xff reduce max query 16b test\n");
    QueryParams qp;
    qp.op = REDUCE_MAX;
    auto f_comp = [](const uint16_t* src, uint32_t len, int16_t* dest, uint16_t ndims) {
        return compress_rowmajor_xff_rle_16b(src, len, dest, ndims);
    };
    test_query<2>(qp, f_comp, query_rowmajor_xff_rle_16b);
}

// TODO this one segfaults for unclear reasons
//
 TEST_CASE("query xff reduce sum 16b", "[xff][16b][query][sum][dbg]") {
     printf("executing xff reduce sum query 16b test\n");
     QueryParams qp;
     qp.op = REDUCE_SUM;
     auto f_comp = [](const uint16_t* src, uint32_t len, int16_t* dest, uint16_t ndims) {
         return compress_rowmajor_xff_rle_16b(src, len, dest, ndims);
     };
     test_query<2>(qp, f_comp, query_rowmajor_xff_rle_16b);
 }

