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

TEST_CASE("compress8b_rowmajor", "[rowmajor][dbg]") {
     TEST_SIMPLE_INPUTS(1, compress8b_rowmajor, decompress8b_rowmajor);
     TEST_SIMPLE_INPUTS(63, compress8b_rowmajor, decompress8b_rowmajor);
     TEST_SIMPLE_INPUTS(64, compress8b_rowmajor, decompress8b_rowmajor);
     TEST_SIMPLE_INPUTS(127, compress8b_rowmajor, decompress8b_rowmajor);
     TEST_SIMPLE_INPUTS(128, compress8b_rowmajor, decompress8b_rowmajor);
     TEST_SIMPLE_INPUTS(129, compress8b_rowmajor, decompress8b_rowmajor);
     TEST_SIMPLE_INPUTS(4096 + 17, compress8b_rowmajor, decompress8b_rowmajor);
}
