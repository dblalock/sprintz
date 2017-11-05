//
//  predict.cpp
//  Compress
//
//  Created by DB on 11/4/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include "predict.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "immintrin.h"

#include "debug_utils.hpp" // TODO rm
#include "util.h"

uint32_t encode_xff_rowmajor(const uint8_t* src, uint32_t len, int8_t* dest,
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
        orig_dest = dest;
    }

    // ensure we don't write past the end of the output
    uint16_t overrun_ndims = vector_sz - (ndims % vector_sz);
    uint32_t trailing_nelements = len % block_sz_elems;
    if (nblocks > 1 && overrun_ndims > trailing_nelements) { nblocks -= 1; }

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

    if (write_size) { return len + 6; }
    return len;
}
uint32_t decode_xff_rowmajor(const int8_t* src, uint32_t len, uint8_t* dest,
                             uint16_t ndims)
{
    static const uint8_t vector_sz = 32;
    static const uint8_t block_sz = 8;
    uint8_t* orig_dest = dest;

    if (ndims == 0) { return 0; }

    uint32_t block_sz_elems = block_sz * ndims;
    uint32_t nrows = len / ndims;
    uint32_t nblocks = nrows / block_sz;
    uint32_t padded_ndims = round_up_to_multiple(ndims, vector_sz);
    uint16_t nvectors = padded_ndims / vector_sz + ((padded_ndims % vector_sz) > 0);

    uint8_t* prev_vals_ar = (uint8_t*)calloc(vector_sz, nvectors);

    // ensure we don't write past the end of the output
    uint16_t overrun_ndims = vector_sz - (ndims % vector_sz);
    uint32_t trailing_nelements = len % block_sz_elems;
    if (nblocks > 1 && overrun_ndims > trailing_nelements) { nblocks -= 1; }

    for (uint32_t b = 0; b < nblocks; b++) { // for each block
        const int8_t* block_in_ptr = src + (nvectors - 1) * vector_sz;
        uint8_t* block_out_ptr = dest + (nvectors - 1) * vector_sz;
        for (int32_t v = nvectors - 1; v >= 0; v--) { // for each stripe
            __m256i* prev_vals_ptr = (__m256i*)(&prev_vals_ar[0] + v * vector_sz);
            __m256i prev_vals = _mm256_loadu_si256(prev_vals_ptr);
            const int8_t* in_ptr = block_in_ptr;
            uint8_t* out_ptr = block_out_ptr;
            for (uint8_t i = 0; i < block_sz; i++) {
                __m256i errs = _mm256_loadu_si256((__m256i*)in_ptr);
                __m256i vals = _mm256_add_epi8(errs, prev_vals);
                _mm256_storeu_si256((__m256i*)out_ptr, vals);
                prev_vals = vals;
                in_ptr += ndims;
                out_ptr += ndims;
            }
            _mm256_storeu_si256((__m256i*)(prev_vals_ptr), prev_vals);
            block_in_ptr -= vector_sz;
            block_out_ptr -= vector_sz;
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
    for (; dest < (orig_dest + len); ++dest) {
        *dest = *src + *(dest - ndims);
        src++;
    }
    return len;
}

uint32_t decode_xff_rowmajor_inplace(uint8_t* buff, uint32_t len,
                                      uint16_t ndims)
{
    uint8_t* tmp = (uint8_t*)malloc(len);
    uint32_t sz = decode_xff_rowmajor((int8_t*)buff, len, tmp, ndims);
    memcpy(buff, tmp, sz);
    return sz;
}

uint32_t decode_xff_rowmajor(const int8_t* src, uint8_t* dest) {
    uint32_t len = *(uint32_t*)src;
    src += 4;
    uint16_t ndims = *(uint16_t*)src;
    src += 2;
    return decode_xff_rowmajor(src, len, dest, ndims);
}
