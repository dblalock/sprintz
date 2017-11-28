//
//  sprintz_xff.h
//  Compress
//
//  Created by DB on 7/3/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef sprintz_xff_h
#define sprintz_xff_h

#include <stdint.h>

#define USE_X86_INTRINSICS
#define USE_AVX2

#ifdef USE_AVX2
    static_assert(__AVX2__, "AVX 2 is required! Try --march=native or -mavx2");
    #define USE_X86_INTRINSICS
#endif

int64_t compress8b_rowmajor_xff(const uint8_t* src, size_t len,
    int8_t* dest, uint16_t ndims, bool write_size=true);
int64_t decompress8b_rowmajor_xff(const int8_t* src, uint8_t* dest);

int64_t compress8b_rowmajor_xff_rle(const uint8_t* src, size_t len,
    int8_t* dest, uint16_t ndims, bool write_size=true);
int64_t decompress8b_rowmajor_xff_rle(const int8_t* src, uint8_t* dest);

#endif
