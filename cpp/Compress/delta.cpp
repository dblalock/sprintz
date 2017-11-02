//
//  delta.cpp
//  Compress
//
//  Created by DB on 11/1/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include "delta.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "immintrin.h"

#include "debug_utils.hpp" // TODO rm

// TODO eliminate dup code
#ifndef MAX
    #define MAX(x, y) ( ((x) > (y)) ? (x) : (y) )
#endif
#ifndef MIN
    #define MIN(x, y) ( ((x) < (y)) ? (x) : (y) )
#endif
#if __cpp_constexpr >= 201304
    #define CONSTEXPR constexpr
#else
    #define CONSTEXPR
#endif
template<typename T, typename T2>
CONSTEXPR inline T round_up_to_multiple(T x, T2 multipleof) {
    T remainder = x % multipleof;
    return remainder ? (x + multipleof - remainder) : x;
}

uint32_t encode_delta_rowmajor(const uint8_t* src, uint32_t len, int8_t* dest,
                               uint16_t ndims, bool write_size)
{
    static const uint8_t vector_sz = 32;
    static const uint8_t block_sz = 8;

    int8_t* orig_dest = dest;

    uint32_t block_sz_elems = block_sz * ndims;
    uint32_t nrows = len / ndims;
    uint32_t nblocks = nrows / block_sz;
    uint32_t padded_ndims = round_up_to_multiple(ndims, vector_sz);
    uint16_t nvectors = padded_ndims / vector_sz + ((padded_ndims % vector_sz) > 0);

    uint8_t* prev_vals_ar = (uint8_t*)calloc(vector_sz, nvectors);

    if (write_size) {
        *(uint32_t*)dest = len;
        dest += 4;
        *(uint16_t*)dest = ndims;
        dest += 2;
        orig_dest = dest; // NOTE: we don't do this in any other function
    }

    printf("-------- compression (len = %lld, ndims = %d)\n", (int64_t)len, ndims);
    // printf("saw original data:\n"); dump_bytes(src, len, ndims);
    printf("saw original data:\n"); dump_bytes(src, ndims * 8, ndims);

    // nblocks = 0; // TODO rm


    // uint16_t overrun_ndims = ndims % vector_sz;
    // uint32_t npad_bytes = overrun_ndims * block_sz;

    // uint32_t npad_blocks = (npad_bytes / block_sz_elems) + ((npad_bytes % block_sz_elems) > 0);
    // nblocks -= MIN(nblocks, npad_blocks);
    // // if (len < vector_sz * block_sz) { nblocks = 0; }

    // printf("overrun ndims: %d\n", overrun_ndims);

    // ensure we don't write past the end of the output
    uint16_t overrun_ndims = ndims % vector_sz;
    uint32_t trailing_nelements = len % block_sz_elems;
    if (nblocks > 1 && overrun_ndims > trailing_nelements) { nblocks -= 1; }
    // if (nblocks > 1) { nblocks -= 1; }

    // printf("using nblocks, nvectors: %d, %d\n", nblocks, nvectors);

    // nblocks = 0;

    for (int32_t b = 0; b < nblocks; b++) { // for each block
        for (int32_t v = nvectors - 1; v >= 0; v--) { // for each stripe
            __m256i* prev_vals_ptr = (__m256i*)(prev_vals_ar + v * vector_sz);
            __m256i prev_vals = _mm256_loadu_si256(prev_vals_ptr);
            for (uint8_t i = 0; i < block_sz; i++) {
                const uint8_t* in_ptr = src + i * ndims + v * vector_sz;
                int8_t* out_ptr = dest + i * ndims + v * vector_sz;
                __m256i vals = _mm256_loadu_si256((__m256i*)in_ptr);
                __m256i vdeltas = _mm256_sub_epi8(vals, prev_vals);
                _mm256_storeu_si256((__m256i*)out_ptr, vdeltas);
                prev_vals = vals;
            }
            _mm256_storeu_si256((__m256i*)(prev_vals_ptr), prev_vals);
        } // for each vector
        src += block_sz_elems;
        dest += block_sz_elems;
    } // for each block

    // delta code trailing elements serially; note that if we jump straight
    // to this section, we need to not read past the beginning of the input
    if (nblocks == 0) {
        uint32_t cpy_len = MIN(ndims, len);
        memcpy(dest, src, cpy_len);
        dest += ndims;
        src += ndims;
    }
    // printf("copying trailing %d bytes\n", (int)((orig_dest + len) - dest));
    for (; dest < (orig_dest + len); ++dest) {
        *dest = *(src) - *(src - ndims);
        src++;
    }

    return len;
}

uint32_t decode_delta_rowmajor(const int8_t* src, uint32_t len, uint8_t* dest,
                               uint16_t ndims)
{
    static const uint8_t vector_sz = 32;
    static const uint8_t block_sz = 8;
    uint8_t* orig_dest = dest;

    uint32_t block_sz_elems = block_sz * ndims;
    uint32_t nrows = len / ndims;
    uint32_t nblocks = nrows / block_sz;
    uint32_t padded_ndims = round_up_to_multiple(ndims, vector_sz);
    uint16_t nvectors = padded_ndims / vector_sz + ((padded_ndims % vector_sz) > 0);

    uint8_t* prev_vals_ar = (uint8_t*)calloc(vector_sz, nvectors);

    printf("-------- decompression (len = %lld, ndims = %d)\n", (int64_t)len, ndims);
    printf("saw compressed data:\n");
    // dump_bytes(src, len, ndims);
    dump_bytes(src, ndims * 8, ndims);

    // nblocks = 0; // TODO rm


    // uint16_t overrun_ndims = ndims % vector_sz;
    // uint32_t npad_bytes = overrun_ndims * block_sz;
    // uint32_t block_sz_elems = block_sz * ndims;
    // uint32_t npad_blocks = (npad_bytes / block_sz_elems) + ((npad_bytes % block_sz_elems) > 0);
    // nblocks -= MIN(nblocks, npad_blocks);

    // printf("overrun ndims: %d\n", overrun_ndims);
    // if (len < vector_sz * block_sz) { nblocks = 0; }

    // ensure we don't write past the end of the output

    uint16_t overrun_ndims = ndims % vector_sz;
    uint32_t trailing_nelements = len % block_sz_elems;
    if (nblocks > 1 && overrun_ndims > trailing_nelements) { nblocks -= 1; }

    // printf("using nblocks, nvectors: %d, %d\n", nblocks, nvectors);

    // nblocks = 0;

    for (uint32_t b = 0; b < nblocks; b++) { // for each block
        for (int32_t v = nvectors - 1; v >= 0; v--) { // for each stripe
            __m256i* prev_vals_ptr = (__m256i*)(prev_vals_ar + v * vector_sz);
            __m256i prev_vals = _mm256_loadu_si256(prev_vals_ptr);
            for (uint8_t i = 0; i < block_sz; i++) {
                const int8_t* in_ptr = src + i * ndims + v * vector_sz;
                uint8_t* out_ptr = dest + i * ndims + v * vector_sz;
                __m256i errs = _mm256_loadu_si256((__m256i*)in_ptr);
                __m256i vals = _mm256_add_epi8(errs, prev_vals);
                _mm256_storeu_si256((__m256i*)out_ptr, vals);
                prev_vals = vals;
            }
            _mm256_storeu_si256((__m256i*)(prev_vals_ptr), prev_vals);
        } // for each vector
        src += block_sz_elems;
        dest += block_sz_elems;
    } // for each block

    // undo delta coding for trailing elements serially
    if (nblocks == 0) {
        uint32_t cpy_len = MIN(ndims, len);
        memcpy(dest, src, cpy_len);
        dest += ndims;
        src += ndims;
    }
    // printf("copying trailing %d bytes\n", (int)((orig_dest + len) - dest));
    for (; dest < (orig_dest + len); ++dest) {
        // printf("copying element %d\n", (int)(dest - orig_dest));
        *dest = *src + *(dest - ndims);
        src++;
    }

    printf("reconstructed input:\n");
    dump_bytes(orig_dest, ndims * 8, ndims);

    return len;
}


uint32_t decode_delta_rowmajor(const int8_t* src, uint8_t* dest) {
    uint32_t len = *(uint32_t*)src;
    src += 4;
    uint16_t ndims = *(uint16_t*)src;
    src += 2;
    return decode_delta_rowmajor(src, len, dest, ndims);
}

#undef MIN
