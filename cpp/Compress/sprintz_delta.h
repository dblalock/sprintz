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

#include "macros.h"

#define USE_X86_INTRINSICS
#define USE_AVX2

#ifdef USE_AVX2
    static_assert(__AVX2__, "AVX 2 is required! Try --march=native or -mavx2");
    #define USE_X86_INTRINSICS
#endif

// ------------------------ no preprocessing (just bitpacking)

int64_t compress8b_rowmajor(const uint8_t* src, uint64_t len, int8_t* dest,
                            uint16_t ndims, bool write_size=true);
int64_t decompress8b_rowmajor(const int8_t* src, uint8_t* dest);

// ------------------------ delta coding only

int64_t compress8b_rowmajor_delta(const uint8_t* src, uint64_t len, int8_t* dest,
                                  uint16_t ndims, bool write_size=true);
int64_t decompress8b_rowmajor_delta(const int8_t* src, uint8_t* dest);

// ------------------------ delta + run length encoding

int64_t compress8b_rowmajor_delta_rle(const uint8_t* src, uint64_t len,
    int8_t* dest, uint16_t ndims, bool write_size=true);

SPRINTZ_FORCE_INLINE int64_t decompress8b_rowmajor_delta_rle(
    const int8_t* src, uint8_t* dest, uint16_t ndims, uint64_t ngroups,
    uint16_t remaining_len);

int64_t decompress8b_rowmajor_delta_rle(const int8_t* src, uint8_t* dest);

// ------------------------ delta + rle low dimensional

int64_t compress8b_rowmajor_delta_rle_lowdim(const uint8_t* src, uint64_t len,
    int8_t* dest, uint16_t ndims, bool write_size=true);

SPRINTZ_FORCE_INLINE int64_t decompress8b_rowmajor_delta_rle_lowdim(
    const int8_t* src, uint8_t* dest, uint16_t ndims, uint64_t ngroups,
    uint16_t remaining_len);

int64_t decompress8b_rowmajor_delta_rle_lowdim(
    const int8_t* src, uint8_t* dest);

// ------------------------ misc

uint16_t write_metadata_rle(int8_t* dest, uint16_t ndims, uint32_t ngroups,
    uint16_t remaining_len);

uint16_t read_metadata_rle(const int8_t* src, uint16_t* p_ndims,
    uint64_t* p_ngroups, uint16_t* p_remaining_len);

#endif
