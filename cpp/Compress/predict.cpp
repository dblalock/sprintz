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

// typedef struct _shift_pair {
//     uint8_t pos_shift;
//     uint8_t neg_shift;
// } ShiftPair;
// const ShiftPair kShiftPairs[16] = {
//     {7, 1}, {3, 1}, {2, 1}, {4, 2}, {3, 2}, {4, 3}, {0, 0}, {3, 4},
//     {2, 3}, {2, 4}, {1, 2}, {1, 3}, {0, 1}, {0, 2}, {0, 3}, {0, 7}
// };
// const int16_t kShiftCoeffs[16] = {
//     -126, -96, -64, -48, -32, -16,  0,  16,
//       32,  48,  64,  96, 128, 192,224, 254};

// TODO serial xff encoding instead
inline int64_t decode_delta_serial(const int8_t* src, uint8_t* dest,
    const uint8_t* dest_end, uint16_t lag, bool needs_initial_cpy)
{
    int64_t len = (int64_t)(dest_end - dest);
    if (len < 1) { return -1; }
    if (lag < 1) { return -2; }

    if (needs_initial_cpy) {
        int64_t cpy_len = MIN(lag, len);
        memcpy(dest, src, cpy_len);
        dest += lag;
        src += lag;
    }
    for (; dest < dest_end; ++dest) {
        *dest = *src + *(dest - lag);
        src++;
    }
    return len;
}

// see https://godbolt.org/g/1Q31mK for assembly
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

    uint8_t* prev_vals_ar = (uint8_t*)calloc(2 * vector_sz, nvectors);
    int8_t* prev_deltas_ar = (int8_t*)prev_vals_ar + vector_sz * nvectors;

    if (write_size) {
        *(uint32_t*)dest = len;
        dest += 4;
        *(uint16_t*)dest = ndims;
        dest += 2;
        orig_dest = dest; // NOTE: we don't do this in any other function
    }

    // printf("-------- compression (len = %lld, ndims = %d)\n", (int64_t)len, ndims);
    // printf("saw original data:\n"); dump_bytes(src, len, ndims);
    // printf("saw original data:\n"); dump_bytes(src, ndims * 8, ndims);

    // ensure we don't write past the end of the output
    uint16_t overrun_ndims = vector_sz - (ndims % vector_sz);
    uint32_t trailing_nelements = len % block_sz_elems;
    if (nblocks > 1 && overrun_ndims > trailing_nelements) { nblocks -= 1; }

    // const __m256i high_bits_one = _mm256_set1_epi8(-128);
    // const __m256i zeros = _mm256_setzero_si256();
    // const __m256i ones = _mm256_cmpeq_epi8(zeros, zeros);

    const __m256i filter_coeffs_even = _mm256_setr_epi16(
        16, 32, 64, 128, 256, 255, 128, 64,
        31, 32, 33, 200, 256, 200, 100, 50);
    const __m256i filter_coeffs_odd = _mm256_setr_epi16(
        256, 128, 64, 32, 16, 8, 4, 2,
        2, 4, 8, 16, 32, 64, 128, 256);
    const __m256i low_mask = _mm256_set1_epi16(0xff);

    for (int32_t b = 0; b < nblocks; b++) { // for each block
        for (int32_t v = nvectors - 1; v >= 0; v--) { // for each stripe
            __m256i* prev_vals_ptr = (__m256i*)(prev_vals_ar + v * vector_sz);
            __m256i* prev_deltas_ptr = (__m256i*)(prev_deltas_ar + v * vector_sz);
            __m256i prev_vals = _mm256_loadu_si256(prev_vals_ptr);
            __m256i prev_deltas = _mm256_loadu_si256(prev_deltas_ptr);
            for (uint8_t i = 0; i < block_sz; i++) {
                const uint8_t* in_ptr = src + i * ndims + v * vector_sz;
                int8_t* out_ptr = dest + i * ndims + v * vector_sz;
                __m256i vals = _mm256_loadu_si256((__m256i*)in_ptr);
                __m256i vdeltas = _mm256_sub_epi8(vals, prev_vals);

                // want: ([a8 b8 c8 d8] .* [e16 f16 g16 h16]) >> 8,
                //  where rhs vals <= 256; key here is that products end
                //  up in byte 1 for each multiplication
                //
                // 1) [a8; 0] [c8; 0] .* e16 f16 = [...;ae] [...;cg] //and, mul
                // 2) [b8; 0] [d8; 0] .* e16 f16 = [...;bf] [...;dh] //shft,mul
                // 3) [ae; 0] [cg; 0] // left shift
                //  | [...;bf][...;dh]
                //  = [ae; bf][cg; dh]  // vpblendv
                __m256i even_predictions = _mm256_mullo_epi16(
                    _mm256_and_si256(prev_deltas, low_mask), filter_coeffs_even);
                __m256i odd_predictions = _mm256_mullo_epi16(
                    _mm256_srai_epi16(prev_deltas, 8), filter_coeffs_odd);
                __m256i vpredictions = _mm256_blendv_epi8(odd_predictions,
                    _mm256_srli_epi16(even_predictions, 8), low_mask);

                // __m256i vpredictions = prev_deltas;
                __m256i verrs = _mm256_sub_epi8(vdeltas, vpredictions);

                _mm256_storeu_si256((__m256i*)out_ptr, verrs);
                prev_deltas = vdeltas;
                prev_vals = vals;
            }
            _mm256_storeu_si256((__m256i*)prev_vals_ptr, prev_vals);
            _mm256_storeu_si256((__m256i*)prev_deltas_ptr, prev_deltas);
        } // for each vector
        src += block_sz_elems;
        dest += block_sz_elems;
    } // for each block

    // uint32_t remaining_len = len - (uint32_t)(dest - orig_dest);
    // int8_t* use_prev_deltas = nullptr;
    // const uint8_t* use_prev_vals = nullptr;
    // if (nblocks > 0) {
    //     use_prev_vals = src - ndims;
    //     use_prev_deltas = prev_deltas_ar;
    // }
    // encode_doubledelta_serial(src, remaining_len, dest, ndims,
    //                           use_prev_vals, use_prev_deltas);

    // delta code trailing elements serially; note that if we jump straight
    // to this section, we need to not read past the beginning of the input
    if (nblocks == 0) {
        uint32_t cpy_len = MIN(ndims, len);
        memcpy(dest, src, cpy_len);
        dest += ndims;
        src += ndims;
    }
    for (; dest < (orig_dest + len); ++dest) {
        *dest = *(src) - *(src - ndims);
        src++;
    }

    free(prev_vals_ar);
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

    uint8_t* prev_vals_ar = (uint8_t*)calloc(2 * vector_sz, nvectors);
    int8_t* prev_deltas_ar = (int8_t*)prev_vals_ar + vector_sz * nvectors;

    // printf("-------- decompression (len = %lld, ndims = %d)\n", (int64_t)len, ndims);
    // printf("saw compressed data:\n");
    // dump_bytes(src, len, ndims);
    // dump_bytes(src, ndims * 8, ndims);

    // ensure we don't write past the end of the output
    uint16_t overrun_ndims = vector_sz - (ndims % vector_sz);
    uint32_t trailing_nelements = len % block_sz_elems;
    if (nblocks > 1 && overrun_ndims > trailing_nelements) { nblocks -= 1; }

    // const __m256i high_bits_one = _mm256_set1_epi8(-128);
    // const __m256i zeros = _mm256_setzero_si256();
    // const __m256i ones = _mm256_cmpeq_epi8(zeros, zeros);
    const __m256i filter_coeffs_even = _mm256_setr_epi16(
        16, 32, 64, 128, 256, 255, 128, 64,
        31, 32, 33, 200, 256, 200, 100, 50);
    const __m256i filter_coeffs_odd = _mm256_setr_epi16(
        256, 128, 64, 32, 16, 8, 4, 2,
        2, 4, 8, 16, 32, 64, 128, 256);
    const __m256i low_mask = _mm256_set1_epi16(0xff);

    for (uint32_t b = 0; b < nblocks; b++) { // for each block
        for (int32_t v = nvectors - 1; v >= 0; v--) { // for each stripe
            __m256i* prev_vals_ptr = (__m256i*)(prev_vals_ar + v * vector_sz);
            __m256i* prev_deltas_ptr = (__m256i*)(prev_deltas_ar + v * vector_sz);
            __m256i prev_vals = _mm256_loadu_si256(prev_vals_ptr);
            __m256i prev_deltas = _mm256_loadu_si256(prev_deltas_ptr);
            for (uint8_t i = 0; i < block_sz; i++) {
                const int8_t* in_ptr = src + i * ndims + v * vector_sz;
                uint8_t* out_ptr = dest + i * ndims + v * vector_sz;
                __m256i verrs = _mm256_loadu_si256((__m256i*)in_ptr);

                // see encode func for explanation
                __m256i even_predictions = _mm256_mullo_epi16(
                    _mm256_and_si256(prev_deltas, low_mask), filter_coeffs_even);
                __m256i odd_predictions = _mm256_mullo_epi16(
                    _mm256_srai_epi16(prev_deltas, 8), filter_coeffs_odd);
                __m256i vpredictions = _mm256_blendv_epi8(odd_predictions,
                    _mm256_srli_epi16(even_predictions, 8), low_mask);

                __m256i vdeltas = _mm256_add_epi8(verrs, vpredictions);
                __m256i vals = _mm256_add_epi8(vdeltas, prev_vals);

                _mm256_storeu_si256((__m256i*)out_ptr, vals);
                prev_deltas = vdeltas;
                prev_vals = vals;
            }
            _mm256_storeu_si256((__m256i*)prev_vals_ptr, prev_vals);
            _mm256_storeu_si256((__m256i*)prev_deltas_ptr, prev_deltas);
        } // for each vector
        src += block_sz_elems;
        dest += block_sz_elems;
    } // for each block

    // undo delta coding for trailing elements serially
    decode_delta_serial(src, dest, orig_dest + len, ndims, nblocks == 0);

    free(prev_vals_ar);
    return len;
}

// TODO actually operate in-place
uint32_t decode_xff_rowmajor_inplace(uint8_t* buff, uint32_t len,
                                      uint16_t ndims)
{
    uint8_t* tmp = (uint8_t*)malloc(len);
    uint32_t sz = decode_xff_rowmajor((int8_t*)buff, len, tmp, ndims);
    memcpy(buff, tmp, sz);
    free(tmp);
    return sz;
}

uint32_t decode_xff_rowmajor(const int8_t* src, uint8_t* dest) {
    uint32_t len = *(uint32_t*)src;
    src += 4;
    uint16_t ndims = *(uint16_t*)src;
    src += 2;
    return decode_xff_rowmajor(src, len, dest, ndims);
}
