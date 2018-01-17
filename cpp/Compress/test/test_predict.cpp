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
#include "predict.h"
#include "testing_utils.hpp"

#include "debug_utils.hpp" // TODO rm


// ============================================================ sprintz predict

TEST_CASE("xff_rowmajor_8b (no compression)", "[rowmajor][xff][preproc]") {
   printf("executing rowmajor xff 8b test (no compression)\n");
   TEST_CODEC_MANY_NDIMS_8b(encode_xff_rowmajor_8b, decode_xff_rowmajor_8b);
}
TEST_CASE("xff_rowmajor_16b (no compression)", "[rowmajor][xff][preproc]") {
    printf("executing rowmajor xff 16b test (no compression)\n");
   TEST_CODEC_MANY_NDIMS_16b(encode_xff_rowmajor_16b, decode_xff_rowmajor_16b);
}
//     int ndims = 17;
//     auto ndims_list = ar::range(ndims, ndims + 1);
//     // auto ndims_list = ar::range(1, 129 + 1);
//     for (auto _ndims : ndims_list) {
//         auto ndims = (uint16_t)_ndims;
//         printf("---- ndims = %d\n", ndims);
//         CAPTURE(ndims);
//         auto comp = [ndims](const uint16_t* src, size_t len, int16_t* dest) {
//             return encode_xff_rowmajor_16b(src, (uint32_t)len, dest, ndims);
//         };
//         auto decomp = [](const int16_t* src, uint16_t* dest) {
//             return decode_xff_rowmajor_16b(src, dest);
//         };

//         auto SZ = 256;
//         Vec_u16 raw(SZ);
//         {
//             for (int i = 0; i < SZ; i++) {
//                 raw(i) = (i % 2) ? (i + 64) % 128 : 0;
//             }
//             // TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);
//             test_compressor<2>(raw, comp, decomp, "debug test");
//         }
//     }

//     // TEST_COMP_DECOMP_PAIR_NO_SECTIONS(comp, decomp);
// }
