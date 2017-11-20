//
//  test_sprintz2.cpp
//  Compress
//
//  Created by DB on 9/28/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include "catch.hpp"
#include "eigen/Eigen"

#include "compress_testing.hpp"

// #include "array_utils.hpp"
#include "sprintz2.h"
// #include "bitpack.h"
#include "test_utils.hpp"

#include "debug_utils.hpp" // TODO rm

//TEST_CASE("how_many_times_are_these_run?", "[rowmajor][dbg]") {
//    printf("running nop test\n");
//    auto ndims_list = ar::range(1, 33 + 1);
//    for (auto ndims : ndims_list) {
//        printf("------ ndims = %d\n", ndims);
//    }
//
//    REQUIRE(true);
//}

// template<typename SizeT, typename CompT, typename DecompT>
// void _test_simple_inputs(SizeT SZ, CompT f_comp, DecompT f_decomp) {
//     {
//         Vec_u8 raw(SZ);
//         for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }
//         TEST_COMPRESSOR(SZ, f_comp, f_decomp);
//     }
//     {
//         Vec_u8 raw(SZ);
//         for (int i = 0; i < SZ; i++) { raw(i) = ((i + 64) / 1) % 128; }
//         TEST_COMPRESSOR(SZ, f_comp, f_decomp);
//     }
// }

/*
 #undef TEST_KNOWN_INPUT
 #define TEST_KNOWN_INPUT(SZ, COMP_FUNC, DECOMP_FUNC)                 \
    {                                                                   \
        Vec_u8 raw(SZ);                                                 \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 0;                \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
//*/
/*
     {                                                   \
         Vec_u8 raw(SZ);                                                 \
         for (int i = 0; i < SZ; i++) { raw(i) = (i + 64) % 128; } \
         TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
     }                                                                  \
{                                                    \
    Vec_u8 raw(SZ);                                                 \
    for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }               \
    TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
}                                                                   \
//*/

TEST_CASE("compress8b_rowmajor", "[rowmajor]") {
    printf("executing rowmajor test\n");

   // uint16_t ndims = 8;
    // auto ndims_list = ar::range(1, 33 + 1);
   auto ndims_list = ar::range(1, 129 + 1);
//    auto ndims_list = ar::range(33, 33 + 1);
//    auto ndims_list = ar::range(65, 65 + 1);
   // auto ndims_list = ar::range(1, 2);
   // auto ndims_list = ar::range(4, 5);
    // auto ndims_list = ar::range(8, 9);
    // auto ndims_list = ar::range(10, 11);
//    ar::print(ndims_list, "ndims_list");
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
            return compress8b_rowmajor(src, len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return decompress8b_rowmajor(src, dest);
        };

        // size_t SZ = ndims * 16;
        // Vec_u8 raw(SZ);
        // for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }
        // for (int i = 0; i < SZ; i++) { raw(i) = (i + 64) % 128; }
        // for (int i = 0; i < SZ; i++) { raw(i) = 64 + i % 64; }
        // for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? 64 + i % 64 : 72; }
        // for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? (i + 64) % 128 : 0;}
        // for (int i = 0; i < SZ; i++) { raw(i) = (i + 96) % 256; }
        // for (int i = 0; i < SZ; i++) { raw(i) = i % 256; }
        // for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? (i + 64) % 128 : 62 + (i + 1) % 4;}
        // for (int i = 0; i < SZ; i++) { raw(i) = (i % 2) ? (i + 64) % 128 : 30 + (i + 1) % 4;}
        // TEST_COMPRESSOR(SZ, comp, decomp);

       // TEST_SQUARES_INPUT(1, comp, decomp);
       // TEST_SQUARES_INPUT(ndims * 16, comp, decomp);
       // TEST_SQUARES_INPUT(128, comp, decomp);
//        _test_simple_inputs(1, comp, decomp);
       // TEST_SIMPLE_INPUTS(1, comp, decomp);
       // TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
       // TEST_KNOWN_INPUT(1, comp, decomp);
       // TEST_KNOWN_INPUT(64, comp, decomp);
       // TEST_KNOWN_INPUT(63, comp, decomp);
       // TEST_KNOWN_INPUT(64, comp, decomp);
       // TEST_KNOWN_INPUT(65, comp, decomp);
       // TEST_KNOWN_INPUT(ndims * 8 - 1, comp, decomp);
       // TEST_KNOWN_INPUT(ndims * 8, comp, decomp);
       // TEST_KNOWN_INPUT(ndims * 8 + 1, comp, decomp);
       // TEST_KNOWN_INPUT(ndims * 16 - 1, comp, decomp);
       // TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
       // TEST_KNOWN_INPUT(ndims * 16 + 1, comp, decomp);
       // TEST_KNOWN_INPUT(127, comp, decomp);
       // TEST_KNOWN_INPUT(128, comp, decomp);
       // TEST_KNOWN_INPUT(129, comp, decomp);
       // TEST_KNOWN_INPUT(4096 + 17, comp, decomp);

       TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
    }
//    REQUIRE(true);
}


void test_dset(DatasetName name, uint16_t ndims, int64_t limit_bytes=-1) {
    Dataset raw = read_dataset(name, limit_bytes);
    // ar::print(data.get(), 100, "msrc");
    auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
        return compress8b_rowmajor(src, len, dest, ndims);
    };
    auto decomp = [](int8_t* src, uint8_t* dest) {
        return decompress8b_rowmajor(src, dest);
    };
    printf("compressing %lld bytes of %d-dimensional data\n", raw.size(), ndims);
    TEST_COMPRESSOR(raw.size(), comp, decomp);
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

TEST_CASE("compress8b_rowmajor_delta", "[rowmajor][delta]") {
    printf("executing rowmajor delta test\n");

    // int ndims = 1;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
            return compress8b_rowmajor_delta(src, len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return decompress8b_rowmajor_delta(src, dest);
        };

        // TEST_SQUARES_INPUT(128, comp, decomp);
        // TEST_SQUARES_INPUT(ndims * 16, comp, decomp);
        // TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
       TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
    }
}

// ============================================================ rowmajor delta rle

TEST_CASE("compress8b_rowmajor_delta_rle", "[rowmajor][delta][dbg]") {
    printf("executing rowmajor delta rle test\n");

    // int ndims = 2;
    // auto ndims_list = ar::range(ndims, ndims + 1);
    auto ndims_list = ar::range(1, 129 + 1);
    for (auto _ndims : ndims_list) {
        auto ndims = (uint16_t)_ndims;
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims](uint8_t* src, size_t len, int8_t* dest) {
            return compress8b_rowmajor_delta_rle(src, len, dest, ndims);
        };
        auto decomp = [](int8_t* src, uint8_t* dest) {
            return decompress8b_rowmajor_delta_rle(src, dest);
        };

        // TEST_SQUARES_INPUT(128, comp, decomp);
        // TEST_SQUARES_INPUT(ndims * 16, comp, decomp);
        // TEST_SIMPLE_INPUTS(ndims * 16, comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 16, comp, decomp);
        // TEST_KNOWN_INPUT(ndims * 32, comp, decomp);
        TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
        // TEST_ZEROS(129, comp, decomp);
        // TEST_FUZZ((ndims * 8000), comp, decomp);
        // TEST_FUZZ((40 * 1024 * 1024), comp, decomp);
        TEST_SPARSE(1024 * 1024 + 13, comp, decomp);
        // TEST_SPARSE(128, comp, decomp);
        // TEST_SPARSE(, comp, decomp);

        // auto SZ = 1024 * 1024 + 7;
        // srand(123);
        // // Vec_u8 orig(SZ);
        // Vec_u8 raw(SZ);
        // raw.setRandom();
        // raw /= 254;
        // // orig.setRandom();
        // // raw = orig / 254;
        // TEST_COMPRESSOR(SZ, comp, decomp);

        // // Vec_u8 raw(SZ);
        // // for (int i = 0; i < SZ; i++) {
        // //     if ((i / 16) % 2 || (i / 32) % 2) {
        // //         raw(i) = 0;
        // //     } else {
        // //         raw(i) = 1;
        // //     }
        // //  }
        // TEST_COMPRESSOR(SZ, comp, decomp);


        // #define COMP_FUNC comp
        // #define DECOMP_FUNC decomp

        // vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,
        //     66, 71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17};
        // // {
        // //     for (auto sz : sizes) {
        // //         TEST_KNOWN_INPUT(sz, COMP_FUNC, DECOMP_FUNC);
        // //     }
        // // }
        // // {
        // //     for (auto sz : sizes) {
        // //         TEST_ZEROS(sz, COMP_FUNC, DECOMP_FUNC);
        // //     }
        // // }
        // // {
        // //     for (auto sz : sizes) {
        // //         TEST_FUZZ(sz, COMP_FUNC, DECOMP_FUNC);
        // //     }
        // // }
        // // {
        // //     TEST_FUZZ(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);
        // // }
        // {
        //     for (auto sz : sizes) {
        //         TEST_SPARSE(sz, COMP_FUNC, DECOMP_FUNC);
        //     }
        // }

    }
}
