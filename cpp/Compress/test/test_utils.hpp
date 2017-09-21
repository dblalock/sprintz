//
//  test_utils.hpp
//  Compress
//
//  Created by DB on 9/21/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include "eigen/Eigen"

// #include <stdio.h>

// #include "catch.hpp"

// #include "array_utils.hpp"
// #include "sprintz.h"
// #include "bitpack.h"
// #include "timing_utils.hpp"
// #include "test_utils.hpp"

// #include "debug_utils.hpp" // TODO rm


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

#endif // TEST_UTILS_HPP
