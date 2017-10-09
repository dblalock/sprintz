//
//  sprintz2.c
//  Compress
//
//  Created by DB on 9/16/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "bitpack.h"

// #define VERBOSE_COMPRESS

#include "debug_utils.hpp" // TODO rm
#include "array_utils.hpp" // TODO rm


static constexpr uint64_t kHeaderMask8b = TILE_BYTE(0x07); // 3 ones

// byte shuffle values to construct data masks; note that nbits == 7 yields
// a byte of all ones (0xff); also note that rows 1 and 3 below are unused
static const __m256i nbits_to_mask = _mm256_setr_epi8(
    0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // unused
    0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00); // unused


// ------------------------------------------------ row-major, no delta or RLE

size_t round_up_to_multiple(size_t x, size_t multipleof) {
    size_t remainder = x % multipleof;
    return remainder ? (x + multipleof - remainder) : x;
}

int64_t compress8b_rowmajor(uint8_t* src, size_t len, int8_t* dest,
                            uint16_t ndims, bool write_size)
{
    static const int block_sz = 8;
    static const int group_sz_blocks = 1;
    static const int nbits_sz_bits = 3;
    static const int stripe_sz = 8;
    // static const int vector_sz = 32; // TODO block tiled storage
    const uint8_t* orig_src = src;
    const int8_t* orig_dest = dest;

    // static const uint16_t ndims = 8;
    int nfullstripes = ndims / stripe_sz;
    int nstripes = nfullstripes + ((ndims % stripe_sz) > 0);
    int stripe_header_sz = (nstripes * block_sz * nbits_sz_bits) / 8;
    size_t group_sz = ndims * block_sz * group_sz_blocks;

    // store how long this is and how many dimensions there are
    if (write_size) {
        assert(len < ((uint64_t)1) << 48);
        *(uint64_t*)dest = len;
        *(uint16_t*)(dest + 6) = ndims;
        dest += 8;
    }

    printf("-------- compression\n");
    // printf("saw original data (with possible extra at end):\n");
    // dumpBytes(src, len);

    uint32_t rounded_ndims = (uint32_t)round_up_to_multiple(ndims, 8);
    uint8_t* nbits_ar = (uint8_t*)malloc(rounded_ndims); // used for header
    uint8_t* stripe_bitwidths = (uint8_t*)malloc(nstripes);
    uint32_t* stripe_bitoffsets = (uint32_t*)malloc(nstripes * 4);
    uint64_t* stripe_masks = (uint64_t*)malloc(nstripes * sizeof(uint64_t));

    size_t nblocks = len / (block_sz * ndims);
    // printf("nblocks: %lu\n", nblocks);
    for (size_t b = 0; b < nblocks; b++) {
        int8_t* header_dest = dest;
        dest += stripe_header_sz * nstripes;

        // zero info from previous iter
        memset(stripe_bitwidths, 0, nstripes * sizeof(uint8_t));
        memset(stripe_masks, 0, nstripes * sizeof(uint64_t));

        // compute nbits for each variable (dim)
        for (size_t dim = 0; dim < ndims; dim++) {
            // uint8_t max_nbits = 0;
            uint8_t mask = 0;
            for (int i = 0; i < block_sz; i++) {
                uint8_t val = src[(i * ndims) + dim];
                // max_nbits = MAX(max_nbits, NBITS_COST_U8[val]);
                mask |= NBITS_MASKS_U8[val];
            }
            // store nbits for this dim (only used in header)
            uint8_t max_nbits = (32 - _lzcnt_u32((uint32_t)mask));
            nbits_ar[dim] = max_nbits - (max_nbits == 8); // map 8 to 7

            // accumulate stats about this stripe
            size_t stripe = dim / stripe_sz;
            stripe_bitwidths[stripe] += max_nbits;
            stripe_masks[stripe] |= ((uint64_t)mask) << ((dim % stripe_sz) * 8);
        }

        // compute start offsets of each stripe (in bits)
        stripe_bitoffsets[0] = 0;
        for (size_t stripe = 1; stripe < nstripes; stripe++) {
            stripe_bitoffsets[stripe] = stripe_bitoffsets[stripe - 1] +
                stripe_bitwidths[stripe - 1];
        }
        // compute width of each row (in bytes); note that we byte align
        uint32_t row_width_bits = stripe_bitoffsets[nstripes - 1] +
            stripe_bitwidths[nstripes-1];
        uint32_t row_sz_bytes =
            (row_width_bits >> 3) + ((row_width_bits % 8) > 0);

        // ar::print(nbits_ar, ndims, "nbits_ar");

        for (size_t stripe = 0; stripe < nstripes; stripe++) {
            // write out headers; 3b for each nbits
            uint64_t nbits_u64 = *(uint64_t*)(nbits_ar + stripe * stripe_sz);
            uint32_t packed_header = (uint32_t)_pext_u64(nbits_u64, kHeaderMask8b);
            memcpy(header_dest, &packed_header, stripe_header_sz);
            // printf("packed header: "); dumpEndianBits(packed_header);
            // printf("wrote stripe header:\n");
            // dumpEndianBits(*(uint32_t*)header_dest);
            header_dest += stripe_header_sz;

            // // compute stripe widths (in bits)
            // size_t start_idx = block_sz * stripe;
            // size_t end_idx = MIN(start_idx + block_sz, ndims);
            // uint8_t width_bits = 0;
            // for (size_t dim = start_idx; dim < end_idx; dim++) {
            //     width_bits += nbits_ar[dim] + (nbits_ar[dim] == 7);
            //     // width_bits += nbits_ar[dim];
            // }
            // stripe_bitwidths[stripe] = width_bits;
        }

        // ar::print(stripe_bitwidths, nstripes, "stripe_bitwidths");
        // printf("row width bits: %d\n", row_width_bits);

        // write out packed data
        // TODO last stripe will clobber next row starts if (ndims % 8 != 0)
        for (size_t stripe = 0; stripe < nstripes; stripe++) { // for each stripe of dims
            uint32_t offset_bits = stripe_bitoffsets[stripe] & 0x07;
            uint32_t offset_bytes = stripe_bitoffsets[stripe] >> 3;
            uint64_t mask = stripe_masks[stripe];

            uint8_t nbits = stripe_bitwidths[stripe];
            int total_bits = nbits + offset_bits;

            // printf("total bits, nbits lost = %d, %d\n", total_bits, nbits_lost);

            int8_t* outptr = dest + offset_bytes;
            uint8_t* inptr = src + (stripe * stripe_sz);

            // uint64_t mask = 0;
            // for (int dim = 0; dim < stripe_sz; dim++) {
                // uint8_t dim_nbits = nbits_ar[dim + stripe * stripe_sz];
                // uint64_t dim_mask = (1 << dim_nbits) - 1;
                // dim_mask |= dim_nbits == 7 ? 0x80 : 0;
                // mask |= dim_mask << (8 * dim);
            // }

            // printf("stripe sz: %d\n", stripe_sz);
            // printf("data mask:"); dumpEndianBits(mask);
            // printf("\n");

            // printf("offset bytes, bits = %u, %u\n", offset_bytes, offset_bits);

            // XXX Note that this impl assumes that output buff is zeroed
            if (total_bits <= 64) { // always fits in one u64
                // printf("using mask: "); dumpEndianBits(mask);
                for (int i = 0; i < block_sz; i++) { // for each sample in block

                    // horizontally pack the bits
                    // uint64_t packed_data = 0;
                    // uint8_t nbits_filled = 0;
                    // for (int dim = 0; dim < stripe_sz; dim++) {
                    //     uint8_t dim_nbits = nbits_ar[dim + stripe * stripe_sz];
                    //     uint8_t shift_that_zeros_this_dim = 64 - (8 * dim);
                    //     uint8_t left_shift_amt = shift_that_zeros_this_dim - dim_nbits;
                    //     uint8_t right_shft_amt = 64 - dim_nbits - nbits_filled;
                    //     packed_data |= (data << left_shift_amt) >> right_shft_amt;
                    //     nbits_filled += dim_nbits;
                    // }

                    // 8B write to store (at least most of) the data
                    uint64_t data = *(uint64_t*)inptr;
                    uint64_t packed_data = _pext_u64(data, mask);
                    // ar::print(inptr, stripe_sz, "stripe");
                    // printf("packed data: "); dumpBytes(packed_data);
                    uint64_t write_data = packed_data << offset_bits;
                    // printf("data"); dumpEndianBits(data);
                    // printf("packed_data"); dumpEndianBits(packed_data);
                    *(uint64_t*)outptr = write_data | (*outptr);

                    outptr += row_sz_bytes;
                    inptr += ndims;
                }
            } else { // XXX can't test this for real with ndims=8
                uint8_t nbits_lost = MAX((int)0, total_bits - (int)64);
                // data spans 9 bytes, so we can't rely on single 64b write
                for (int i = 0; i < block_sz; i++) { // for each sample in block
                    uint64_t data = *(uint64_t*)inptr;
                    uint64_t packed_data = _pext_u64(data, mask);
                    // printf("packed data: "); dumpBytes(packed_data);

                    uint64_t write_data = packed_data << offset_bits;
                    *(uint64_t*)outptr = write_data | (*outptr);
                    uint8_t extra_byte = packed_data >> (64 - nbits_lost);
                    *(outptr + 8) |= extra_byte;

                    outptr += row_sz_bytes;
                }
            }
        }
        src += group_sz;
        dest += block_sz * group_sz_blocks * row_sz_bytes;
    }
    // just memcpy remaining bytes (up to 63 of them)
    // size_t remaining_len = len % block_sz;
    size_t remaining_len = len - (src - orig_src);
    // printf("read src bytes: %lu\n", (size_t)(src - orig_src));
    // printf("remaining_len: %lu\n", remaining_len);
    // printf("remaining data: "); ar::print(src, remaining_len);
    memcpy(dest, src, remaining_len);

    free(nbits_ar);
    free(stripe_bitwidths);
    free(stripe_bitoffsets);
    free(stripe_masks);

    return dest + remaining_len - orig_dest;
}

int64_t decompress8b_rowmajor(int8_t* src, uint8_t* dest) {
    static const uint8_t block_sz = 8;
    static const uint8_t group_sz_blocks = 1;
    static const uint8_t vector_sz = 32;
    static const uint8_t stripe_sz = 8;
    // static const uint8_t stripes_per_vector = vector_sz / stripe_sz;
    static const uint8_t nbits_sz_bits = 3;
    static const uint8_t stripe_header_sz =
        (group_sz_blocks * nbits_sz_bits * block_sz) / 8;
    static const uint8_t nbits_sz_mask = 0x07;
    static const uint64_t header_unpack_mask = TILE_BYTE(nbits_sz_mask);
    assert(vector_sz % stripe_sz == 0);

    // TODO ndims shouldn't be a constant, but others should be (which
    // we can do by having multiple impls for different ranges of nbits)
    // static const size_t ndims = 8;
    // static const size_t nstripes = ndims / stripe_sz + ((ndims % stripe_sz) > 0);
    // static const size_t nvectors = (ndims / vector_sz) + ((ndims % vector_sz) > 0);
    // static const size_t group_sz = ndims * block_sz * group_sz_blocks;

    uint8_t* orig_dest = dest;

    // read in size of original data and number of dimensions
    static const size_t len_nbytes = 6;
    uint64_t one = 1; // make next line legible
    uint64_t len_mask = (one << (8 * len_nbytes)) - 1;
    uint64_t orig_len = (*(uint64_t*)src) & len_mask;
    uint16_t ndims = (*(uint16_t*)(src + len_nbytes));
    src += 8;

    // compute stats derived from ndims
    size_t nstripes = ndims / stripe_sz + ((ndims % stripe_sz) > 0);
    size_t nvectors = (ndims / vector_sz) + ((ndims % vector_sz) > 0);
    size_t group_sz = ndims * block_sz * group_sz_blocks;

    printf("-------- decompression\n");
    // printf("saw compressed data (with possible extra at end):\n");
    // dumpBytes(src, orig_len + 16);

    size_t ndims_padded = round_up_to_multiple(ndims, vector_sz);
    uint8_t* headers = (uint8_t*)calloc(1, ndims_padded);
    size_t rounded_up_nstripes = ndims_padded / stripe_sz;
    uint64_t* data_masks = (uint64_t*)calloc(rounded_up_nstripes, 8);
    uint64_t* stripe_bitwidths = (uint64_t*)calloc(rounded_up_nstripes, 8);
    uint32_t* stripe_bitoffsets = (uint32_t*)calloc(rounded_up_nstripes, 4);

    size_t nblocks = orig_len / (block_sz * ndims);
    // printf("decomp nblocks: %lu\n", nblocks);
    for (size_t b = 0; b < nblocks; b++) {
        uint8_t* header_src = (uint8_t*)src;
        src += stripe_header_sz * nstripes;

        // ar::print(header_src, stripe_header_sz * nstripes, "header src");

        // unpack all headers
        for (size_t stripe = 0; stripe < nstripes; stripe++) {
            uint64_t packed_header = *(uint32_t*)header_src;
            header_src += stripe_header_sz;
            uint64_t header = _pdep_u64(packed_header, header_unpack_mask);
            // printf("header mask u64:"); dumpEndianBits(header_unpack_mask);
            // printf("packed header u64:"); dumpEndianBits(packed_header);
            // printf("header u64:"); dumpEndianBits(header);
            *(uint64_t*)(headers + stripe * stripe_sz) = header;
        }
        // compute masks and bitwidths for all stripes
        for (size_t v = 0; v < nvectors; v++) {
            __m256i raw_header = _mm256_loadu_si256(
                (const __m256i*)(headers + v * vector_sz));
            // map nbits of 7 to 8
            static const __m256i sevens = _mm256_set1_epi8(0x07);
            __m256i header = _mm256_sub_epi8(
                raw_header, _mm256_cmpeq_epi8(raw_header, sevens));

            // ar::print(headers, ndims_padded, "headers array");
            // printf("header: "); dump_m256i(header);

            // compute and store bit widths
            __m256i bitwidths = _mm256_sad_epu8(
                header, _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i*)(stripe_bitwidths + v * vector_sz),
                                bitwidths);

            // compute and store masks
            __m256i masks = _mm256_shuffle_epi8(nbits_to_mask, raw_header);
            printf("masks: "); dump_m256i(masks);
            _mm256_storeu_si256((__m256i*)(data_masks + v * vector_sz), masks);
        }

        // compute where each stripe begins, as well as width of a row
        stripe_bitoffsets[0] = 0;
        for (size_t stripe = 1; stripe < nstripes; stripe++) {
            stripe_bitoffsets[stripe] = (uint32_t)(stripe_bitoffsets[stripe - 1]
                + stripe_bitwidths[stripe - 1]);
        }
        uint32_t row_sz_bits = (uint32_t)(stripe_bitoffsets[nstripes - 1] +
            stripe_bitwidths[nstripes - 1]);
        uint32_t row_sz_bytes = (row_sz_bits >> 3) + ((row_sz_bits % 8) > 0);

        ar::print(stripe_bitwidths, nstripes, "stripe_bitwidths");
        printf("row width bits: %d\n", row_sz_bits);

        // unpack data for each stripe
        for (size_t stripe = 0; stripe < nstripes; stripe++) {
            uint32_t offset_bits = stripe_bitoffsets[stripe] & 0x07;
            uint32_t offset_bytes = stripe_bitoffsets[stripe] >> 3;

            uint64_t mask = data_masks[stripe];
            uint8_t nbits = stripe_bitwidths[stripe];

            int8_t* inptr = src + offset_bytes;
            uint8_t* outptr = dest + (stripe * stripe_sz);

            uint8_t total_bits = nbits + offset_bits;
            // printf("total bits: %d\n", total_bits);

            // this is the hot loop
            if (total_bits <= 64) { // guaranteed to fit in 8B
                for (int i = 0; i < block_sz; i++) {
                    uint64_t packed_data = (*(uint64_t*)inptr) >> offset_bits;
                    *(uint64_t*)outptr = _pdep_u64(packed_data, mask);
                    inptr += row_sz_bytes;
                    outptr += ndims;
                }
            } else { // spans 9 bytes
                for (int i = 0; i < block_sz; i++) {
                    uint64_t packed_data = (*(uint64_t*)inptr) >> offset_bits;
                    // printf("packed_data "); dumpBytes(packed_data);
                    packed_data |= (*(uint64_t*)(inptr + 8)) << nbits;
                    // printf("packed_data after OR "); dumpBytes(packed_data);
                    *(uint64_t*)outptr = _pdep_u64(packed_data, mask);
                    inptr += row_sz_bytes;
                    outptr += ndims;
                }
                // printf("data we wrote: "); dumpBytes(dest, group_sz);
            }
        }
        src += block_sz * group_sz_blocks * row_sz_bytes;
        dest += group_sz;
    }
    // copy over trailing data
    size_t remaining_len = orig_len % group_sz;
    printf("remaining len: %lu\n", remaining_len);
    // printf("read bytes: %lu\n", remaining_len);
    printf("remaining data: "); ar::print(src, remaining_len);
    memcpy(dest, src, remaining_len);

    free(headers);
    free(data_masks);
    free(stripe_bitwidths);
    free(stripe_bitoffsets);

    return dest + remaining_len - orig_dest;
}
