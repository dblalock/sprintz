//
//  sprintz2.h
//  Compress
//
//  Created by DB on 7/3/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef sprintz2_h
#define sprintz2_h

#include <stdint.h>

#define USE_X86_INTRINSICS
#define USE_AVX2

#ifdef USE_AVX2
    static_assert(__AVX2__, "AVX 2 is required! Try --march=native or -mavx2");
    #define USE_X86_INTRINSICS
#endif


int64_t compress8b_rowmajor(const uint8_t* src, size_t len, int8_t* dest,
                            uint16_t ndims=8, bool write_size=true);
int64_t decompress8b_rowmajor(const int8_t* src, uint8_t* dest);

int64_t compress8b_rowmajor_delta(const uint8_t* src, size_t len, int8_t* dest,
                            uint16_t ndims=8, bool write_size=true);
int64_t decompress8b_rowmajor_delta(const int8_t* src, uint8_t* dest);

#endif
