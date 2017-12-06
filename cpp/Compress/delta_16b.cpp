//
//  delta_16b.cpp
//  Compress
//
//  Created by DB on 12/5/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>
#include <string.h>

#include "format.h"
#include "util.h"

// TODO self code this; basically just paste from delta.cpp

inline int64_t decode_delta_serial(const int16_t* src, uint16_t* dest,
    const uint16_t* dest_end, uint16_t lag, bool needs_initial_cpy)
{
    static const uint8_t elem_sz = sizeof(*src);

    int64_t len = (int64_t)(dest_end - dest);
    if (len < 1) { return -1; }
    if (lag < 1) { return -2; }

    if (needs_initial_cpy) {
        int64_t cpy_len = MIN(lag, len);
        memcpy(dest, src, cpy_len * elem_sz);
        dest += lag;
        src += lag;
    }
    for (; dest < dest_end; ++dest) {
        *dest = *src + *(dest - lag);
        src++;
    }
    return len;
}

uint32_t encode_delta_rowmajor(const uint16_t* src, uint32_t len, int16_t* dest,
    uint16_t ndims, bool write_size=true)
{
    static const uint8_t elem_sz = sizeof(*src);
    static const uint8_t vector_sz = 32 / elem_sz;
    static const uint8_t block_sz = 8;

    int16_t* orig_dest = dest;

    uint32_t block_sz_elems = block_sz * ndims;
    uint32_t nrows = len / ndims;
    uint32_t nblocks = nrows / block_sz;
    uint32_t padded_ndims = round_up_to_multiple(ndims, vector_sz);
    uint16_t nvectors = div_round_up(padded_ndims, vector_sz);

    uint16_t* prev_vals_ar = (uint16_t*)calloc(vector_sz * elem_sz, nvectors);

    uint16_t metadata_len = 0;
    if (write_size) {
        metadata_len = write_metadata_simple(dest, ndims, len);
        dest += metadata_len;
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
                const uint16_t* in_ptr = src + i * ndims + v * vector_sz;
                int16_t* out_ptr = dest + i * ndims + v * vector_sz;
                __m256i vals = _mm256_loadu_si256((__m256i*)in_ptr);
                __m256i vdeltas = _mm256_sub_epi16(vals, prev_vals);
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
        memcpy(dest, src, cpy_len * elem_sz);
        dest += ndims;
        src += ndims;
    }
    // printf("copying trailing %d bytes\n", (int)((orig_dest + len) - dest));
    for (; dest < (orig_dest + len); ++dest) {
        *dest = *(src) - *(src - ndims);
        src++;
    }

    free(prev_vals_ar);
    return len + metadata_len;
}

uint32_t decode_delta_rowmajor(const int16_t* src, uint32_t len, uint16_t* dest,
    uint16_t ndims)
{
    static const uint8_t elem_sz = sizeof(*src);
    static const uint8_t vector_sz = 32 / elem_sz;
    static const uint8_t block_sz = 8;
    uint16_t* orig_dest = dest;

    if (ndims == 0) { return 0; }

    uint32_t block_sz_elems = block_sz * ndims;
    uint32_t nrows = len / ndims;
    uint32_t nblocks = nrows / block_sz;
    uint32_t padded_ndims = round_up_to_multiple(ndims, vector_sz);
    uint16_t nvectors = div_round_up(padded_ndims, vector_sz);
    uint16_t* prev_vals_ar = (uint16_t*)calloc(vector_sz * elem_sz, nvectors);

    // ensure we don't write past the end of the output
    uint16_t overrun_ndims = vector_sz - (ndims % vector_sz);
    uint32_t trailing_nelements = len % block_sz_elems;
    if (nblocks > 1 && overrun_ndims > trailing_nelements) { nblocks -= 1; }

    for (uint32_t b = 0; b < nblocks; b++) { // for each block
        const int16_t* block_in_ptr = src + (nvectors - 1) * vector_sz;
        uint16_t* block_out_ptr = dest + (nvectors - 1) * vector_sz;
        for (int32_t v = nvectors - 1; v >= 0; v--) { // for each stripe
            __m256i* prev_vals_ptr = (__m256i*)(prev_vals_ar + v * vector_sz);
            __m256i prev_vals = _mm256_loadu_si256(prev_vals_ptr);
            const int16_t* in_ptr = block_in_ptr;
            uint16_t* out_ptr = block_out_ptr;
            for (uint8_t i = 0; i < block_sz; i++) {
                __m256i errs = _mm256_loadu_si256((__m256i*)in_ptr);
                __m256i vals = _mm256_add_epi16(errs, prev_vals);
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
    decode_delta_serial(src, dest, orig_dest + len, ndims, nblocks == 0);

    free(prev_vals_ar);
    return len;
}
uint32_t decode_delta_rowmajor(const int16_t* src, uint16_t* dest) {
    uint16_t ndims;
    uint32_t len;
    src += read_metadata_simple(src, &ndims, &len);
    return decode_delta_rowmajor(src, len, dest, ndims);
}
