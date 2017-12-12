//
//  sprintz3.cpp
//  Compress
//
//  Created by DB on 2017-12-2.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include "sprintz_xff.h"

#include <stdio.h>

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "bitpack.h"
#include "format.h"
#include "transpose.h"
#include "util.h" // for copysign

static constexpr uint64_t kHeaderMask8b = TILE_BYTE(0x07); // 3 ones

static const __m256i nbits_to_mask = _mm256_setr_epi8(
    0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // unused
    0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00); // unused


static const int kDefaultGroupSzBlocks = 2;
// static const int kDefaultGroupSzBlocks = 8;  // slight pareto improvement

static const int debug = 0;
// static const int debug = 3;

// ------------------------------------------------ delta + rle low ndims

int64_t compress8b_rowmajor_xff_rle_lowdim(const uint8_t* src, uint32_t len,
    int8_t* dest, uint16_t ndims, bool write_size)
{
    // constants that could, in principle, be changed (but not in this impl)
    static const uint8_t log2_block_sz = 3;
    // static const int stripe_sz = 8;
    static const int nbits_sz_bits = 3;
    // constants that could actually be changed in this impl
    static const int group_sz_blocks = kDefaultGroupSzBlocks;
    static const int length_header_nbytes = 8;
    static const uint8_t log2_learning_downsample = 1;
    static const uint16_t max_run_nblocks = 0x7fff; // 15 bit counter
    static const uint8_t learning_shift = 1;
    // derived consts
    static const int block_sz = 8;
    static const uint8_t learning_downsample = 1 << log2_learning_downsample;
    static const uint32_t min_data_size = 8 * block_sz * group_sz_blocks;

    const uint8_t* orig_src = src;
    int8_t* orig_dest = dest;
    const uint8_t* src_end = src + len;

    // ================================ one-time initialization

    // ------------------------ stats derived from ndims
    // uint16_t nstripes = ndims / stripe_sz + ((ndims % stripe_sz) > 0);
    uint32_t group_sz = ndims * block_sz * group_sz_blocks;
    uint32_t total_header_bits = ndims * nbits_sz_bits * group_sz_blocks;
    uint32_t total_header_bytes = (total_header_bits / 8) + ((total_header_bits % 8) > 0);

    // ------------------------ store data size and number of dimensions

    // handle low dims and low length; we'd read way past the end of the
    // input in this case
    if (len < min_data_size) {
        assert(min_data_size < ((uint64_t)1) << 16);
        // printf("data less than min data size: %lu\n", min_data_size);
        if (write_size) {
            // XXX: ((group_size_blocks * ndims * (block_sz - 1)) must fit
            // in 16 bits; for group size, block_sz = 8, need ndims < 1024
            // *(uint32_t*)dest = 0; // 0 groups
            // *(uint16_t*)(dest + 4) = (uint16_t)len;
            // *(uint16_t*)(dest + 6) = ndims;
            // dest += length_header_nbytes;
            dest += write_metadata_rle(dest, ndims, 0, (uint16_t)len);
        }
        memcpy(dest, src, len);
        return (dest - orig_dest) + len;
    }
    if (write_size) {
        dest += length_header_nbytes;
    }

    if (debug) {
        printf("------------ comp (len = %lld)\n", (int64_t)len);
        if (debug > 2) { printf("saw original data:\n"); dump_bytes(src, len); }
    }
    // // printf("saw original data:\n"); dump_bytes(src, len, ndims);
    // // printf("saw original data:\n"); dump_bytes(src, len, ndims * 2, 4);
    // // printf("saw original data:\n"); dump_bytes(src, len, 16);

    // ------------------------ temp storage
    // uint8_t*  stripe_bitwidths  = (uint8_t*) malloc(nstripes*sizeof(uint8_t));
    // uint32_t* stripe_bitoffsets = (uint32_t*)malloc(nstripes*sizeof(uint32_t));
    // uint64_t* stripe_masks      = (uint64_t*)malloc(nstripes*sizeof(uint64_t));
    // uint32_t* stripe_headers    = (uint32_t*)malloc(nstripes*sizeof(uint32_t));
    uint8_t* dims_nbits = (uint8_t*)(malloc(ndims*sizeof(uint8_t)));

    uint32_t total_header_bytes_padded = total_header_bytes + 4;
    uint8_t* header_bytes = (uint8_t*)calloc(total_header_bytes_padded, 1);

    // extra row is for storing previous values
    int8_t*  errs             = (int8_t* )calloc(1, (block_sz + 4) * ndims);
    uint8_t* prev_vals_ar     = (uint8_t*)(errs + (block_sz + 0) * ndims);
    int8_t*  prev_deltas_ar   = (int8_t* )(errs + (block_sz + 1) * ndims);
    int16_t* coef_counters_ar = (int16_t*)(errs + (block_sz + 2) * ndims);

    // ================================ main loop

    uint16_t run_length_nblocks = 0;

    const uint8_t* last_full_group_start = src_end - group_sz;
    uint32_t ngroups = 0;
    while (src <= last_full_group_start) {
        ngroups++;  // invariant: groups we start are always finished

        // printf("==== group %d\n", (int)ngroups - 1);

        int8_t* header_dest = dest;
        dest += total_header_bytes;

        memset(header_bytes, 0, total_header_bytes_padded);
        memset(header_dest, 0, total_header_bytes);

        uint32_t header_bit_offset = 0;
        int b = 0;
        while (b < group_sz_blocks) { // for each block

            // ------------------------ compute info for each stripe
            uint32_t total_dims_nbits = 0;
            for (uint16_t dim = 0; dim < ndims; dim++) {
                // compute maximum number of bits used by any value of this dim,
                // while simultaneously computing errs
                uint8_t mask = 0;
                uint8_t prev_val = prev_vals_ar[dim];
                int8_t prev_delta = prev_deltas_ar[dim];

                int16_t coef = (coef_counters_ar[dim] >> (learning_shift + 4)) << 4;
                int8_t grad_sum = 0;

                for (uint8_t i = 0; i < block_sz; i++) {
                    uint32_t in_offset = (i * ndims) + dim; // rowmajor
                    uint32_t out_offset = (dim * block_sz) + i; // colmajor
                    uint8_t val = src[in_offset];
                    int8_t delta = (int8_t)(val - prev_val);
                    int8_t prediction = (((int16_t)prev_delta) * coef) >> 8;
                    int8_t err = delta - prediction;
                    // uint8_t bits = zigzag_encode_i8(delta);
                    uint8_t bits = zigzag_encode_i8(err);

                    if (i % learning_downsample == learning_downsample - 1) {
                        grad_sum += copysign_i8(err, prev_delta);
                    }

                    mask |= bits;
                    errs[out_offset] = bits;
                    prev_val = val;
                    prev_delta = delta;
                }
                // write out value for delta encoding of next block
                prev_vals_ar[dim] = prev_val;
                prev_deltas_ar[dim] = prev_delta;

                // mask = 255;  // TODO rm
                // if (mask > 0) { mask = 255; } // TODO rm
                uint8_t max_nbits = (32 - _lzcnt_u32((uint32_t)mask));
                max_nbits += max_nbits == 7; // map 7 to 8

                dims_nbits[dim] = max_nbits;
                total_dims_nbits += max_nbits;
            }

just_read_block:

            // ------------------------ handle runs of zeros
            bool do_rle = total_dims_nbits == 0 && run_length_nblocks < max_run_nblocks;
            if (do_rle) {
do_rle:
                run_length_nblocks++; // TODO uncomment
                src += block_sz * ndims;

                if (src < last_full_group_start) {
                    continue; // still enough data to finish group
                } else { // not enough data to finish group
                    // write out headers for this block; they're all zero, so
                    // just increment addr at which to write next time
                    header_bit_offset += ndims * nbits_sz_bits;

                    if (debug) { printf("%d.%d nbits: ", (int)ngroups - 1, b); dump_bytes(dims_nbits, ndims); }
                    if (debug) { printf("aborting and compressing rle block of length %d ending at src offset %d\n", run_length_nblocks, (int)(src - orig_src)); }

                    b++; // we're finishing off this block

                    // write out length of the current constant section
                    *dest = run_length_nblocks & 0x7f; // bottom 7 bits
                    dest++;
                    if (run_length_nblocks > 0x7f) { // need another byte
                        *(dest-1) |= 0x80; // set MSB of previous byte
                        *dest = (uint8_t)(run_length_nblocks >> 7);
                        dest++;
                    }

                    // write out this const section, and use empty const
                    // sections to fill up rest of block
                    for (; b < group_sz_blocks; b++) {
                        *dest = 0;
                        dest++;
                    };

                    // we shouldn't need this write, but makes our invariants
                    // consistent
                    run_length_nblocks = 0;

                    goto main_loop_end; // just memcpy remaining data
                }
            }

            if (run_length_nblocks > 0) { // just finished a run
                if (debug) { printf("%d.%d nbits: ", (int)ngroups - 1, b); dump_bytes(dims_nbits, ndims); }
                if (debug) { printf("compressing rle block of length %d ending at offset %d\n", run_length_nblocks, (int)(src - orig_src)); }
                // printf("initial header bit offset: %d\n", header_bit_offset);
                b++;

                *dest = run_length_nblocks & 0x7f; // bottom 7 bits
                // printf("wrote low byte: "); dump_bits(*dest);
                dest++;
                if (run_length_nblocks > 0x7f) { // need another byte
                    *(dest-1) |= 0x80; // set MSB of previous byte
                    // printf("adding another byte to encode run length\n");
                    *dest = (uint8_t)(run_length_nblocks >> 7);
                    dest++;
                }

                run_length_nblocks = 0;

                // write out header (equivalent since already zeroed)
                header_bit_offset += ndims * nbits_sz_bits;

                // if closing the run finished this block, life is hard; we
                // can't just execute the below code because it would try to
                // write past the end of the header bytes and write a data
                // block the decoder isn't expecting
                if (b == group_sz_blocks) {
                    // printf("forcing group end...\n");
                    // memcpy(prev_vals_ar, src - ndims, ndims);
                    // continue;

                    // start new group, and pretend the block we read
                    // was the first block in that group (which it is now)
                    ngroups++;  // invariant: groups we start are always finished
                    header_bit_offset = 0;
                    b = 0;
                    header_dest = dest;
                    dest += total_header_bytes;
                    memset(header_bytes, 0, total_header_bytes_padded);
                    memset(header_dest, 0, total_header_bytes);

                    goto just_read_block;
                }

                // have to enforce invariant that bitwidth of 0 always gets
                // run-length encoded; this case can happen when the current
                // block was zeros, but we hit the limit `max_run_nblocks`
                // printf("total_header_bits: %d\n", )
                if (total_dims_nbits == 0) { goto do_rle; }

                // printf("row width bits for this nonzero block: %d\n", row_width_bits);
                // printf("modified header bit offset: %d\n", header_bit_offset);
            }

            // ------------------------ write out header bits for this block

            if (debug) { printf("%d.%d nbits: ", (int)ngroups - 1, b); dump_bytes(dims_nbits, ndims); }

            for (uint16_t dim = 0; dim < ndims; dim++) {
                uint16_t byte_offset = header_bit_offset >> 3;
                uint8_t bit_offset = header_bit_offset & 0x07;
                // Note: this write will get more complicated when upper byte
                // of each stripe_header isn't guaranteed to be 0
                uint8_t write_nbits = dims_nbits[dim] - (dims_nbits[dim] == 8);
                *(uint32_t*)(header_dest + byte_offset) |= \
                    write_nbits << bit_offset;
                header_bit_offset += nbits_sz_bits;
            }

            // ------------------------ write out block data

            // zero output so that we can just OR in bits (and also touch each
            // cache line in order to ensure that we prefetch, despite our
            // weird pattern of writes)
            // memset(dest, 0, nstripes * stripe_sz * block_sz);
            uint32_t data_sz = total_dims_nbits; // assumes block_sz == 8
            memset(dest, 0, data_sz);

            uint64_t* delta_buff_u64 = (uint64_t*)errs;
            for (uint16_t dim = 0; dim < ndims; dim++) {
                uint8_t nbits = dims_nbits[dim];
                uint64_t mask = kBitpackMasks8[nbits];
                *((uint64_t*)dest) = _pext_u64(delta_buff_u64[dim], mask);
                dest += nbits;
            }
            src += block_sz * ndims;
            // dest += block_sz * out_row_nbytes;
            b++;
        } // for each block
    } // for each group

main_loop_end:

    free(errs);
    free(dims_nbits);
    free(header_bytes);

    size_t remaining_len = src_end - src;
    if (write_size) {
        *(uint32_t*)orig_dest = ngroups;
        *(uint16_t*)(orig_dest + 4) = (uint16_t)remaining_len;
        *(uint16_t*)(orig_dest + 6) = ndims;
    }
    // printf("trailing len: %d\n", (int)remaining_len);
    memcpy(dest, src, remaining_len);
    return dest + remaining_len - orig_dest;
}

// int64_t decompress8b_rowmajor_xff_rle_lowdim(const int8_t* src, uint8_t* dest) {
// int64_t decompress8b_rowmajor_xff_rle_lowdim(const int8_t* src,
SPRINTZ_FORCE_INLINE int64_t decompress8b_rowmajor_xff_rle_lowdim(
    const int8_t* src, uint8_t* dest, uint16_t ndims, uint64_t ngroups,
    uint16_t remaining_len)
{
    // constants that could, in principle, be changed (but not in this impl)
    static const uint8_t log2_block_sz = 3;
    // static const uint8_t block_sz = 8;
    static const uint8_t vector_sz = 32;
    static const uint8_t stripe_sz = 8;
    static const uint8_t nbits_sz_bits = 3;
    static const __m256i low_mask = _mm256_set1_epi16(0xff);
    // constants that could actually be changed in this impl
    static const uint8_t group_sz_blocks = kDefaultGroupSzBlocks;
    static const uint8_t learning_shift = 1;
    static const uint8_t log2_learning_downsample = 1;
    // derived constants
    static const uint8_t block_sz = 1 << log2_block_sz;
    static const int group_sz_per_dim = block_sz * group_sz_blocks;
    const uint8_t elem_sz = sizeof(*src);
    static const uint8_t learning_downsample = 1 << log2_learning_downsample;
    static const uint8_t stripe_header_sz = nbits_sz_bits * stripe_sz / 8;
    static const uint8_t nbits_sz_mask = (1 << nbits_sz_bits) - 1;
    static const uint64_t kHeaderUnpackMask = TILE_BYTE(nbits_sz_mask);
    static const size_t min_data_size = 8 * block_sz * group_sz_blocks;

    uint8_t* orig_dest = dest;
    // const int8_t* orig_src = src;

    // ================================ one-time initialization

    bool just_cpy = (ngroups == 0) && remaining_len < min_data_size;
    if (just_cpy) { // if data was too small or failed to compress
        memcpy(dest, src, (size_t)remaining_len);
        return remaining_len;
    }
    if (ndims == 0) {
        perror("ERROR: decompress8b_rowmajor_xff_rle_lowdim: Received ndims of 0!");
        return 0;
    }

    if (debug) {
        int64_t min_orig_len = ngroups * group_sz_blocks * block_sz * ndims;
        printf("-------- decompression (orig_len = %lld)\n", (int64_t)min_orig_len);
        if (debug > 3) {
            printf("saw compressed data (with possible missing data if runs):\n");
            dump_bytes(src, min_orig_len + 8);
        }
    }

    // ------------------------ stats derived from ndims
    // header stats
    uint32_t nheader_vals = ndims * group_sz_blocks;
    uint32_t nheader_stripes = nheader_vals / stripe_sz + \
        ((nheader_vals % stripe_sz) > 0);
    uint32_t total_header_bits = ndims * nbits_sz_bits * group_sz_blocks;
    uint32_t total_header_bytes = (total_header_bits / 8) + ((total_header_bits % 8) > 0);
    uint32_t group_header_sz = nheader_vals * elem_sz;

    // stats for main decompression loop
    uint32_t group_sz = ndims * group_sz_per_dim;
    // uint16_t nstripes = ndims / stripe_sz + ((ndims % stripe_sz) > 0);
    uint32_t padded_ndims = round_up_to_multiple(ndims, vector_sz);
    uint16_t nvectors = padded_ndims / vector_sz + ((padded_ndims % vector_sz) > 0);

    // ------------------------ temp storage
    // allocate temp vars of minimal possible size such that we can
    // do vector loads and stores (except bitwidths, which are u64s so
    // that we can store directly after sad_epu8)
    uint8_t*  headers = (uint8_t*) calloc(1, group_header_sz);

    // extra row in errs is to store last decoded values
    // TODO just special case very first row
    // uint8_t* errs = (uint8_t*)calloc(block_sz * padded_ndims, 1);
    // uint8_t* prev_vals_ar = (uint8_t*)calloc(padded_ndims, 1);
    int8_t* errs_ar = (int8_t*)calloc((block_sz + 4) * padded_ndims, 1);
    uint8_t* prev_vals_ar = (uint8_t*)(errs_ar + (block_sz + 0) * padded_ndims);
    int8_t* prev_deltas_ar = (int8_t*)(errs_ar + (block_sz + 1) * padded_ndims);
    int8_t* coeffs_ar_even = (int8_t*)(errs_ar + (block_sz + 2) * padded_ndims);
    int8_t* coeffs_ar_odd =  (int8_t*)(errs_ar + (block_sz + 3) * padded_ndims);

    // ================================ main loop

    for (uint64_t g = 0; g < ngroups; g++) {
        const uint8_t* header_src = (uint8_t*)src;
        src += total_header_bytes;

        uint32_t header_bit_offset = 0;

        // printf("==== group %d\n", (int)g);

        // ------------------------ create unpacked headers array
        // unpack headers for all the blocks; note that this is tricky
        // because we need to zero-pad final stripe's headers in each block
        // so that stripe widths don't get messed up (from initial data in
        // next block's header)
        // uint64_t* header_write_ptr = (uint64_t*)headers_tmp;
        uint64_t* header_write_ptr = (uint64_t*)headers;
        for (size_t stripe = 0; stripe < nheader_stripes - 1; stripe++) {
            uint64_t packed_header = *(uint32_t*)header_src;
            uint64_t header = _pdep_u64(packed_header, kHeaderUnpackMask);
            *header_write_ptr = header;
            header_src += stripe_header_sz;
            header_write_ptr++;
        }
        // unpack header for the last stripe in the last block
        // uint64_t packed_header = (*(uint32_t*)header_src) & final_header_mask;
        uint64_t packed_header = (*(uint32_t*)header_src);
        uint64_t header = _pdep_u64(packed_header, kHeaderUnpackMask);
        *header_write_ptr = header;

        // printf("unpacked header: "); dump_bits(headers, ndims * group_sz_blocks);

        // ------------------------ inner loop; decompress each block
        // uint8_t* header_ptr = headers;
        for (int b = 0; b < group_sz_blocks; b++) { // for each block in group
            uint8_t* header_ptr = headers + (b * ndims);
            if (debug) { printf("%d.%d nbits: ", (int)g, b); dump_bytes(header_ptr); }

            // run-length decode if necessary
            bool all_zeros = true;
            for (uint16_t dim = 0; dim < ndims; dim++) {
                all_zeros = all_zeros && (header_ptr[dim] == 0);
            }
            if (all_zeros) {
                int8_t low_byte = (int8_t)*src;
                uint8_t high_byte = (uint8_t)*(src + 1);
                high_byte = high_byte & (low_byte >> 7); // 0 if low msb == 0
                uint16_t length = (low_byte & 0x7f) | (((uint16_t)high_byte) << 7);

                // write out the run
                if (g > 0 || b > 0) { // if not at very beginning of data
                    // const uint8_t* inptr = dest - ndims;
                    // uint32_t ncopies = length * block_sz;
                    // memrep(dest, inptr, ndims * elem_sz, ncopies);
                    // dest += ndims * ncopies;

                    __m256i* prev_vals_ptr = (__m256i*)(prev_vals_ar);
                    __m256i* prev_deltas_ptr = (__m256i*)(prev_deltas_ar);

                    __m256i* even_counters_ptr = (__m256i*)(coeffs_ar_even);
                    __m256i* odd_counters_ptr = (__m256i*)(coeffs_ar_odd);
                    __m256i coef_counters_even = _mm256_loadu_si256(
                        (const __m256i*)even_counters_ptr);
                    __m256i coef_counters_odd = _mm256_loadu_si256(
                        (const __m256i*)odd_counters_ptr);

                    // set coef[i] to ((counter[i] >> learn_shift) >> 4) << 4)
                    __m256i filter_coeffs_even  = _mm256_srai_epi16(
                        coef_counters_even, learning_shift + 4);
                    __m256i filter_coeffs_odd  = _mm256_srai_epi16(
                        coef_counters_odd, learning_shift + 4);
                    filter_coeffs_even = _mm256_slli_epi16(filter_coeffs_even, 4);
                    filter_coeffs_odd  = _mm256_slli_epi16(filter_coeffs_odd, 4);

                    for (int32_t bb = 0; bb < length; bb++) {
                        // uint8_t prev_val = prev_vals_ar[0];
                        __m256i prev_vals = _mm256_loadu_si256(prev_vals_ptr);
                        __m256i prev_deltas = _mm256_loadu_si256(prev_deltas_ptr);
                        __m256i vals = _mm256_setzero_si256();

                        __m256i even_prev_deltas = _mm256_srai_epi16(
                            _mm256_slli_epi16(prev_deltas, 8), 8);
                        __m256i odd_prev_deltas = _mm256_srai_epi16(
                            prev_deltas, 8);

                        for (uint8_t i = 0; i < block_sz; i++) {
                            __m256i even_predictions = _mm256_mullo_epi16(
                                even_prev_deltas, filter_coeffs_even);
                            __m256i odd_predictions = _mm256_mullo_epi16(
                                odd_prev_deltas, filter_coeffs_odd);
                            __m256i vpredictions = _mm256_blendv_epi8(odd_predictions,
                                _mm256_srli_epi16(even_predictions, 8), low_mask);

                            // since 0 err, predictions equal true deltas
                            __m256i vdeltas = vpredictions;
                            vals = _mm256_add_epi8(prev_vals, vdeltas);

                            _mm256_storeu_si256((__m256i*)(dest + i * ndims), vals);

                            even_prev_deltas = _mm256_srai_epi16(
                                even_predictions, 8);
                            odd_prev_deltas = _mm256_srai_epi16(
                                odd_predictions, 8);
                            prev_deltas = vdeltas;
                            prev_vals = vals;
                        } // for each row
                        _mm256_storeu_si256((__m256i*)prev_vals_ptr, prev_vals);
                        _mm256_storeu_si256((__m256i*)prev_deltas_ptr, prev_deltas);

                        dest += ndims * block_sz;
                    }
                } else { // errs of 0 at very start -> all zeros
                    size_t num_zeros = length * block_sz * ndims;
                    memset(dest, 0, num_zeros * elem_sz);
                    dest += num_zeros;
                }
                if (debug) { printf("decompressed rle block of length %d at offset %d\n", length, (int)(dest - orig_dest)); }

                src++;
                src += (high_byte > 0); // if 0, wasn't used for run length

                continue;
            }

            // ------------------------ unpack data for each dim

            // uint8_t* delta_buff = errs + (dim * block_sz);
            for (uint16_t dim = 0; dim < ndims; dim++) {
                int8_t* outptr = errs_ar + (dim * block_sz);
                uint8_t nbits = header_ptr[dim];
                uint64_t mask = kBitpackMasks8[nbits];
                *((uint64_t*)outptr) = _pdep_u64(*(uint64_t*)src, mask);
                // printf("%d.%d-%d: nbits=%d\t", (int)g, b, dim, nbits);
                // // printf("src:   "); dump_bytes(src, 8, false);
                // // printf(" -> dest:   "); dump_bytes(outptr, 8);
                src += (nbits + (nbits == 7)) * elem_sz; // assumes block_sz == 8
            }

            // ------------------------ transpose

            uint8_t* errs_bytes = (uint8_t*)errs_ar;
            switch (ndims) {
                // no zigzag or delta coding
                // case 1: memcpy(dest, errs, ndims*block_sz*elem_sz); break;
                // case 2: transpose_2x8_8b(errs, dest); break;
                // case 3: transpose_3x8_8b(errs, dest); break;
                // case 4: transpose_4x8_8b(errs, dest); break;

                case 1: break;
                case 2: transpose_2x8_8b(errs_bytes, errs_bytes); break;
                case 3: transpose_3x8_8b(errs_bytes, errs_bytes); break;
                case 4: transpose_4x8_8b(errs_bytes, errs_bytes); break;
                default:
                    printf("ERROR: decompress8b_rowmajor_xff_rle_lowdim: "
                        "received invalid ndims: %d\n", ndims);
            }

            __m256i raw_verrs = _mm256_loadu_si256((const __m256i*)errs_ar);
            __m256i verrs = mm256_zigzag_decode_epi8(raw_verrs);

            // vars that would have been initialized in various cases
            __m256i swapped128_verrs = _mm256_permute2x128_si256(
                verrs, verrs, 0x01);
            __m256i shifted15_verrs = _mm256_alignr_epi8(
                swapped128_verrs, verrs, 15);

            uint8_t prev_val = prev_vals_ar[0];
            uint8_t prev_delta = prev_deltas_ar[0];
            __m256i* prev_vals_ptr = (__m256i*)(prev_vals_ar);
            __m256i* prev_deltas_ptr = (__m256i*)(prev_deltas_ar);
            __m256i prev_vals = _mm256_loadu_si256(prev_vals_ptr);
            __m256i prev_deltas = _mm256_loadu_si256(prev_deltas_ptr);
            __m256i vals = _mm256_setzero_si256();

            __m256i* even_counters_ptr = (__m256i*)(coeffs_ar_even);
            __m256i* odd_counters_ptr = (__m256i*)(coeffs_ar_odd);
            __m256i coef_counters_even = _mm256_loadu_si256(
                (const __m256i*)even_counters_ptr);
            __m256i coef_counters_odd = _mm256_loadu_si256(
                (const __m256i*)odd_counters_ptr);

            // set coef[i] to ((counter[i] >> learn_shift) >> 4) << 4)
            __m256i filter_coeffs_even  = _mm256_srai_epi16(
                coef_counters_even, learning_shift + 4);
            __m256i filter_coeffs_odd  = _mm256_srai_epi16(
                coef_counters_odd, learning_shift + 4);
            filter_coeffs_even = _mm256_slli_epi16(filter_coeffs_even, 4);
            filter_coeffs_odd  = _mm256_slli_epi16(filter_coeffs_odd, 4);

            __m256i gradients_sum = _mm256_setzero_si256();

            switch (ndims) {

            // can't just use a loop because _mm256_srli_si256 demands that
            // the shift amount be a compile-time constant (which it is
            // if the loop is unrolled, but apparently that's insufficient)
/*
    #define LOOP_BODY(I, SHIFT, DELTA_ARRAY)                                \
        { __m256i shifted_errs = _mm256_srli_si256(DELTA_ARRAY, SHIFT);   \
        vals = _mm256_add_epi8(prev_vals, shifted_errs);                  \
        _mm256_storeu_si256((__m256i*)(dest + I * ndims), vals);            \
        prev_vals = vals; }
/*/
    #define LOOP_BODY(I, SHIFT, DELTA_ARRAY)                                \
    {   __m256i shifted_errs = _mm256_srli_si256(DELTA_ARRAY, SHIFT);       \
        __m256i even_prev_deltas = _mm256_srai_epi16(                       \
            _mm256_slli_epi16(prev_deltas, 8), 8);                          \
        __m256i odd_prev_deltas = _mm256_srai_epi16(                        \
            prev_deltas, 8);                                                \
                                                                            \
        __m256i even_predictions = _mm256_mullo_epi16(                      \
            even_prev_deltas, filter_coeffs_even);                          \
        __m256i odd_predictions = _mm256_mullo_epi16(                       \
            odd_prev_deltas, filter_coeffs_odd);                            \
        __m256i vpredictions = _mm256_blendv_epi8(odd_predictions,          \
            _mm256_srli_epi16(even_predictions, 8), low_mask);              \
                                                                            \
        __m256i vdeltas = _mm256_add_epi8(shifted_errs, vpredictions);      \
        vals = _mm256_add_epi8(prev_vals, vdeltas);                         \
                                                                            \
        _mm256_storeu_si256((__m256i*)(dest + I * ndims), vals);            \
        prev_deltas = vdeltas;                                              \
        prev_vals = vals;                                                   \
    }
//*/
            case 1:
                for (uint8_t i = 0; i < block_sz; i++) {
                    int8_t delta = _mm256_extract_epi8(verrs, i);
                    uint8_t val = prev_val + delta;
                    dest[i] = val;
                    prev_val = val;
                }
                prev_vals_ar[0] = prev_val;
                prev_deltas_ar[0] = prev_delta;
                break;
            case 2: // everything fits in lower 128b
                LOOP_BODY(0, 0, verrs); LOOP_BODY(1, 2, verrs);
                LOOP_BODY(2, 4, verrs); LOOP_BODY(3, 6, verrs);
                LOOP_BODY(4, 8, verrs); LOOP_BODY(5, 10, verrs);
                LOOP_BODY(6, 12, verrs); LOOP_BODY(7, 14, verrs);
                // _mm256_storeu_si256((__m256i*)prev_vals_ar, vals);
                _mm256_storeu_si256((__m256i*)prev_vals_ptr, prev_vals);
                _mm256_storeu_si256((__m256i*)prev_deltas_ptr, prev_deltas);
                break;
            case 3:
                LOOP_BODY(0, 0, verrs);
                LOOP_BODY(1, 3, verrs);
                LOOP_BODY(2, 6, verrs);
                LOOP_BODY(3, 9, verrs);
                LOOP_BODY(4, 12, verrs);
                LOOP_BODY(5, 0, shifted15_verrs);
                LOOP_BODY(6, 3, shifted15_verrs);
                LOOP_BODY(7, 6, shifted15_verrs);
                // _mm256_storeu_si256((__m256i*)prev_vals_ar, vals);
                _mm256_storeu_si256((__m256i*)prev_vals_ptr, prev_vals);
                _mm256_storeu_si256((__m256i*)prev_deltas_ptr, prev_deltas);
                break;
            case 4:
                LOOP_BODY(0, 0, verrs);
                LOOP_BODY(1, 4, verrs);
                LOOP_BODY(2, 8, verrs);
                LOOP_BODY(3, 12, verrs);
                LOOP_BODY(4, 0, swapped128_verrs);
                LOOP_BODY(5, 4, swapped128_verrs);
                LOOP_BODY(6, 8, swapped128_verrs);
                LOOP_BODY(7, 12, swapped128_verrs);
                // _mm256_storeu_si256((__m256i*)prev_vals_ar, vals);
                _mm256_storeu_si256((__m256i*)prev_vals_ptr, prev_vals);
                _mm256_storeu_si256((__m256i*)prev_deltas_ptr, prev_deltas);
                break;

    #undef LOOP_BODY
            }

            // mean of gradients in block, for even and odd indices
            const uint8_t rshift = 8 + log2_block_sz - log2_learning_downsample;
            __m256i even_grads = _mm256_srai_epi16(
                _mm256_slli_epi16(gradients_sum, 8), rshift);
            __m256i odd_grads = _mm256_srai_epi16(gradients_sum, rshift);

            // store updated coefficients (or, technically, the counters)
            coef_counters_even = _mm256_add_epi16(coef_counters_even, even_grads);
            coef_counters_odd = _mm256_add_epi16(coef_counters_odd, odd_grads);
            _mm256_storeu_si256(even_counters_ptr, coef_counters_even);
            _mm256_storeu_si256(odd_counters_ptr, coef_counters_odd);

            dest += block_sz * ndims;
        } // for each block
    } // for each group

    free(headers);
    free(errs_ar);
    // free(prev_vals_ar);

    memcpy(dest, src, remaining_len);

    if (debug > 2) {
        size_t dest_sz = dest + remaining_len - orig_dest;
        printf("decompressed data:\n"); dump_bytes(orig_dest, dest_sz);
    }
    // // printf("decompressed data:\n"); dump_bytes(orig_dest, dest_sz, ndims);
    // // printf("decompressed data:\n"); dump_bytes(orig_dest, dest_sz, 16);

    return dest + remaining_len - orig_dest;
}

int64_t decompress8b_rowmajor_xff_rle_lowdim(const int8_t* src, uint8_t* dest) {
    uint16_t ndims;
    uint32_t ngroups;
    uint16_t remaining_len;
    src += read_metadata_rle(src, &ndims, &ngroups, &remaining_len);
    return decompress8b_rowmajor_xff_rle_lowdim(
        src, dest, ndims, ngroups, remaining_len);
}
