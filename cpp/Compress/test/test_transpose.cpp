//
//  test_transpose.cpp
//  Compress
//
//  Created by DB on 10/14/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>
#include <vector>

#include "transpose.h"

#include "array_utils.hpp"
#include "compress_testing.hpp"
#include "test_utils.hpp"


TEST_CASE("transpose 8b", "[transpose]") {
    std::vector<uint8_t> _in = {
        0,   1,  2,  3,  4,  5,  6,  7,
        10, 11, 12, 13, 14, 15, 16, 17,
        20, 21, 22, 23, 24, 25, 26, 27,
        30, 31, 32, 33, 34, 35, 36, 37
    };
    auto in = _in.data();
    uint8_t _out[32];
    auto out = (uint8_t*)_out;

    SECTION("2x8") {
        std::vector<uint8_t> _ans = {
            0, 10,
            1, 11,
            2, 12,
            3, 13,
            4, 14,
            5, 15,
            6, 16,
            7, 17,
            20, 30,
            21, 31,
            22, 32,
            23, 33,
            24, 34,
            25, 35,
            26, 36,
            27, 37
        };
        auto ans = _ans.data();
        transpose_2x8_8b(in, out);
        REQUIRE(ar::all_eq(ans, out, 32));
    }
    SECTION("3x8") {
        std::vector<uint8_t> _ans = {
            0, 10, 20,
            1, 11, 21,
            2, 12, 22,
            3, 13, 23,
            4, 14, 24,
            5, 15, 25,
            6, 16, 26,
            7, 17, 27,
            0,0,0,0, 0,0,0,0
        };
        auto ans = _ans.data();
        transpose_3x8_8b(in, out);
        REQUIRE(ar::all_eq(ans, out, 32));
    }
    SECTION("4x8") {
        std::vector<uint8_t> _ans = {
            0, 10, 20, 30,
            1, 11, 21, 31,
            2, 12, 22, 32,
            3, 13, 23, 33,
            4, 14, 24, 34,
            5, 15, 25, 35,
            6, 16, 26, 36,
            7, 17, 27, 37
        };
        auto ans = _ans.data();
        transpose_4x8_8b(in, out);
        ar::print(out, 32, "out");
        ar::print(ans, 32, "ans");
        REQUIRE(ar::all_eq(ans, out, 32));
    }
}
