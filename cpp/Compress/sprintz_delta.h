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

int64_t compress_rowmajor_8b(const uint8_t* src, uint32_t len, int8_t* dest,
                            uint16_t ndims, bool write_size=true);
int64_t decompress_rowmajor_8b(const int8_t* src, uint8_t* dest);

int64_t compress_rowmajor_16b(const uint16_t* src, uint32_t len, int16_t* dest,
                            uint16_t ndims, bool write_size=true);
int64_t decompress_rowmajor_16b(const int16_t* src, uint16_t* dest);


// ------------------------ delta coding only

int64_t compress_rowmajor_delta_8b(const uint8_t* src, uint32_t len, int8_t* dest,
                                  uint16_t ndims, bool write_size=true);
int64_t decompress_rowmajor_delta_8b(const int8_t* src, uint8_t* dest);

int64_t compress_rowmajor_delta_16b(const uint16_t* src, uint32_t len,
    int16_t* dest, uint16_t ndims, bool write_size=true);
int64_t decompress_rowmajor_delta_16b(const int16_t* src, uint16_t* dest);

// ------------------------ delta + run length encoding

// 8b
int64_t compress_rowmajor_delta_rle_8b(const uint8_t* src, uint32_t len,
    int8_t* dest, uint16_t ndims, bool write_size=true);

SPRINTZ_FORCE_INLINE int64_t decompress_rowmajor_delta_rle_8b(
    const int8_t* src, uint8_t* dest, uint16_t ndims, uint32_t ngroups,
    uint16_t remaining_len);

int64_t decompress_rowmajor_delta_rle_8b(const int8_t* src, uint8_t* dest);

// 16b
int64_t compress_rowmajor_delta_rle_16b(const uint16_t* src, uint32_t len,
    int16_t* dest, uint16_t ndims, bool write_size=true);

SPRINTZ_FORCE_INLINE int64_t decompress_rowmajor_delta_rle_16b(
    const int16_t* src, uint16_t* dest, uint16_t ndims, uint32_t ngroups,
    uint16_t remaining_len);

int64_t decompress_rowmajor_delta_rle_16b(const int16_t* src, uint16_t* dest);

// ------------------------ delta + rle low dimensional

int64_t compress_rowmajor_delta_rle_lowdim_8b(const uint8_t* src, uint32_t len,
    int8_t* dest, uint16_t ndims, bool write_size=true);

SPRINTZ_FORCE_INLINE int64_t decompress_rowmajor_delta_rle_lowdim_8b(
    const int8_t* src, uint8_t* dest, uint16_t ndims, uint64_t ngroups,
    uint16_t remaining_len);

int64_t decompress_rowmajor_delta_rle_lowdim_8b(
    const int8_t* src, uint8_t* dest);

// ------------------------ misc

// TODO this isn't a great place for this since not part of the API
template<int elem_sz> struct ElemSzTraits {};
template<> struct ElemSzTraits<1> { typedef uint64_t bitwidth_t; };
template<> struct ElemSzTraits<2> { typedef uint32_t bitwidth_t; };

// TODO also not a good place for this
#define CHECK_INT_UINT_TYPES_VALID(int_t, uint_t)               \
    static_assert(sizeof(uint_t) == sizeof(int_t),              \
        "uint type and int type sizes must be the same!");      \
    static_assert(sizeof(uint_t) == 1 || sizeof(uint_t) == 2,   \
        "Only element sizes of 1 and 2 bytes are supported!");  \


#endif
