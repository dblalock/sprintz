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


// static constexpr uint64_t kHeaderMask8b = TILE_BYTE(0x07); // 3 ones

// byte shuffle values to construct data masks; note that nbits == 7 yields
// a byte of all ones (0xff); also note that rows 1 and 3 below are unused
static const __m256i nbits_to_mask = _mm256_setr_epi8(
    0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // unused
    0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00); // unused


// ------------------------------------------------ row-major, no delta or RLE

template<typename T, typename T2>
inline T round_up_to_multiple(T x, T2 multipleof) {
    T remainder = x % multipleof;
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
    int stripe_header_sz = nbits_sz_bits * block_sz / 8;
    uint32_t group_sz = ndims * block_sz * group_sz_blocks;

    // store how long this is and how many dimensions there are
    if (write_size) {
        assert(len < ((uint64_t)1) << 48);
        *(uint64_t*)dest = len;
        *(uint16_t*)(dest + 6) = ndims;
        dest += 8;
    }

    // handle low dims and low length; we'd read way past the end of the
    // input in this case
    // if (len < 8 * block_sz) { goto memcpy_remainder; }
    if (len < 8 * block_sz) {
        size_t remaining_len = len - (src - orig_src);
        memcpy(dest, src, remaining_len);
        return dest + remaining_len - orig_dest;
    }

    // printf("-------- compression (len = %lld)\n", (int64_t)len);
    // printf("saw original data:\n"); dumpBytes(src, len, ndims);

    uint8_t*  stripe_bitwidths  = (uint8_t*) malloc(nstripes*sizeof(uint8_t));
    uint32_t* stripe_bitoffsets = (uint32_t*)malloc(nstripes*sizeof(uint32_t));
    uint64_t* stripe_masks      = (uint64_t*)malloc(nstripes*sizeof(uint64_t));
    uint32_t* stripe_headers    = (uint32_t*)malloc(nstripes*sizeof(uint32_t));

    size_t nblocks = len / (block_sz * ndims);
    // printf("nblocks: %lu\n", nblocks);
    for (size_t b = 0; b < nblocks; b++) {
        int8_t* header_dest = dest;
        dest += stripe_header_sz * nstripes;

        // printf("reading input at offset %d\n", (uint32_t)(src - orig_src));

        // zero stripe info from previous iter
        memset(stripe_bitwidths, 0, nstripes * sizeof(stripe_bitwidths[0]));
        memset(stripe_masks,     0, nstripes * sizeof(stripe_masks[0]));
        memset(stripe_headers,   0, nstripes * sizeof(stripe_headers[0]));

        // compute info for each stripe
        for (uint16_t dim = 0; dim < ndims; dim++) {
            // compute maximum number of bits used by any value of this dim
            uint8_t mask = 0;
            // printf("vals: ");
            for (int i = 0; i < block_sz; i++) {
                uint8_t val = src[(i * ndims) + dim];
                // printf("%d ", val);
                // max_nbits = MAX(max_nbits, NBITS_COST_U8[val]);
                mask |= NBITS_MASKS_U8[val];
            }
            // mask = NBITS_MASKS_U8[255]; // TODO rm
            // printf("\t"); printf("(mask = %d)\n", mask);
            uint8_t max_nbits = (32 - _lzcnt_u32((uint32_t)mask));

            uint16_t stripe = dim / stripe_sz;
            uint8_t idx_in_stripe = dim % stripe_sz;

            // accumulate stats about this stripe
            stripe_bitwidths[stripe] += max_nbits;
            stripe_masks[stripe] |= ((uint64_t)mask) << (idx_in_stripe * 8);

            // accumulate header info for this stripe
            uint32_t write_nbits = max_nbits - (max_nbits == 8); // map 8 to 7
            stripe_headers[stripe] |= write_nbits << (idx_in_stripe * 3);
            // printf("write_nbits = %d, header = ", write_nbits); dumpBytes(stripe_headers[stripe]);
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

        // ar::print(stripe_bitwidths, nstripes, "stripe_bitwidths");
        // printf("row width bits: %d\n", row_width_bits);

        // zero output so that we can just OR in bits (and also touch each
        // cache line in order to ensure that we prefetch, despite our
        // weird pattern of writes)
        // memset(dest, 0, nstripes * stripe_sz * block_sz);
        memset(dest, 0, row_sz_bytes * block_sz); // above line can overrun dest buff
        // memset(dest, 0, ndims * block_sz); // above line can overrun dest buff

        // write out packed data; we iterate thru stripes in reverse order
        // since (nbits % stripe_sz) != 0 will make the last stripe in each
        // row write into the start of the first stripe in the next row
        // for (int16_t stripe = nstripes - 1; stripe >= 0; stripe--) {

        for (int16_t stripe = 0; stripe < nstripes; stripe++) {

            // write out packed header for this stripe
            uint32_t packed_header = stripe_headers[stripe];
            memcpy(header_dest, &packed_header, stripe_header_sz);
            header_dest += stripe_header_sz;
            // printf("wrote out header: "); dumpEndianBits(packed_header);
            // printf("wrote out header: "); dumpBytes(packed_header);

            // load info for this stripe
            uint8_t offset_bits = (uint8_t)(stripe_bitoffsets[stripe] & 0x07);
            uint32_t offset_bytes = stripe_bitoffsets[stripe] >> 3;
            uint64_t mask = stripe_masks[stripe];
            uint16_t nbits = stripe_bitwidths[stripe];
            uint16_t total_bits = nbits + offset_bits;

            // printf("total bits, nbits lost = %d, %d\n", total_bits, nbits_lost);

            int8_t* outptr = dest + offset_bytes;
            uint8_t* inptr = src + (stripe * stripe_sz);

            // printf("offset bytes, offset bits, nbits, total_bits = %u, %u, %u, %d\n",
            //     offset_bytes, offset_bits, nbits, total_bits);

            // printf("using mask: "); dumpBytes(mask);

            // XXX Note that this impl assumes that output buff is zeroed
            if (total_bits <= 64) { // always fits in one u64
                // printf("using mask: "); dumpEndianBits(mask);
                for (int i = 0; i < block_sz; i++) { // for each sample in block

                    // uint8_t orig_outptr_byte = *outptr;

                    // 8B write to store (at least most of) the data
                    uint64_t data = *(uint64_t*)inptr;
                    uint64_t packed_data = _pext_u64(data, mask);
                    // ar::print(inptr, stripe_sz, "stripe");
                    // printf("packed data: "); dumpBytes(packed_data);
                    uint64_t write_data = packed_data << offset_bits;
                    // printf("data"); dumpEndianBits(data);
                    // printf("packed_data"); dumpEndianBits(packed_data);
                    *(uint64_t*)outptr = write_data | (*(uint64_t*)outptr);

                    // printf("packing block: "); dumpBytes(data, false);
                    // printf("\t-> "); dumpBytes(*(uint64_t*)outptr);

                    // printf("packing block: "); dumpEndianBits(data, false);
                    // printf("\t-> "); dumpEndianBits(orig_outptr_byte, false); printf(" | "); dumpEndianBits(packed_data, false); printf(" >> %d", offset_bits);
                    // printf("\t-> "); dumpEndianBits(*(uint64_t*)outptr, false); dumpEndianBits(*(outptr+8));

                    outptr += row_sz_bytes;
                    inptr += ndims;
                }
            } else { // data spans 9 bytes
                // XXX can't test this for real with ndims=8
                // printf(">>> executing the slow path!\n");
                // uint8_t nbits_lost = MAX((int)0, total_bits - (int)64);
                uint8_t nbits_lost = total_bits - 64;
                // printf("nbits_lost = %d\n", nbits_lost);
                for (int i = 0; i < block_sz; i++) { // for each sample in block
                    // uint8_t orig_outptr_byte = *outptr; // TODO rm after debug
                    uint64_t data = *(uint64_t*)inptr;
                    uint64_t packed_data = _pext_u64(data, mask);
                    // printf("packed data: "); dumpBytes(packed_data);
                    uint8_t extra_byte = (uint8_t)(packed_data >> (nbits - nbits_lost));
                    uint64_t write_data = packed_data << offset_bits;
                    *(uint64_t*)outptr = write_data | (*(uint64_t*)outptr);
                    *(outptr + 8) = extra_byte;

                    // printf("packing block: "); dumpBytes(data, false);
                    // printf("\t-> "); dumpBytes(*(uint64_t*)outptr);

                    // printf("extra byte = "); dumpEndianBits(extra_byte);
                    // printf("packing block: "); dumpEndianBits(data, false);
                    // printf("\t-> "); dumpEndianBits(orig_outptr_byte, false); printf(" | "); dumpEndianBits(packed_data, false); printf(" >> %d", offset_bits);
                    // printf("\t-> "); dumpEndianBits(*(uint64_t*)outptr, false); dumpEndianBits(*(outptr+8));

                    outptr += row_sz_bytes;
                    inptr += ndims;
                }
            }

            // printf("read back header: "); dumpEndianBits(*(uint32_t*)(header_dest - stripe_header_sz));
        }

        src += group_sz;
        dest += block_sz * group_sz_blocks * row_sz_bytes;
    }

    free(stripe_bitwidths);
    free(stripe_bitoffsets);
    free(stripe_masks);
    free(stripe_headers);

// memcpy_remainder:

    // just memcpy remaining bytes (up to 63 of them)
    // size_t remaining_len = len % block_sz;
    size_t remaining_len = len - (src - orig_src);
    // printf("read src bytes: %lu\n", (size_t)(src - orig_src));
    // printf("remaining_len: %lu\n", remaining_len);
    // printf("remaining data: "); ar::print(src, remaining_len);
    memcpy(dest, src, remaining_len);

    return dest + remaining_len - orig_dest;
}

int64_t decompress8b_rowmajor(int8_t* src, uint8_t* dest) {
    static const uint8_t block_sz = 8;
    static const uint8_t group_sz_blocks = 1;
    static const uint8_t vector_sz = 32;
    static const uint8_t stripe_sz = 8;
    // static const uint8_t stripes_per_vector = vector_sz / stripe_sz;
    static const uint8_t nbits_sz_bits = 3;
    static const uint8_t stripe_header_sz = nbits_sz_bits * block_sz / 8;
    static const uint8_t nbits_sz_mask = 0x07;
    static const uint64_t header_unpack_mask = TILE_BYTE(nbits_sz_mask);
    assert(vector_sz % stripe_sz == 0);
    assert(vector_sz >= stripe_sz);

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

    // if (orig_len < 8 * block_sz) { goto memcpy_remainder; }
    if (orig_len < 8 * block_sz) {
        memcpy(dest, src, orig_len);
        return dest + orig_len - orig_dest;
    }

    // compute stats derived from ndims
    uint16_t nstripes = ndims / stripe_sz + ((ndims % stripe_sz) > 0);
    uint16_t nvectors = (ndims / vector_sz) + ((ndims % vector_sz) > 0);
    uint32_t group_sz = ndims * block_sz * group_sz_blocks;

    // printf("-------- decompression (orig_len = %lld)\n", (int64_t)orig_len);
    // printf("saw compressed data (with possible extra at end):\n");
    // dumpBytes(src, orig_len + 16);

    // printf("nstripes, nvectors = %d, %d\n", nstripes, nvectors);

    // allocate temp vars of minimal possible size such that we can
    // do vector loads and stores (except bitwidths, which are u64s so
    // that we can store directly after sad_epu8)
    uint32_t ndims_padded = round_up_to_multiple(ndims, vector_sz);
    uint8_t* headers = (uint8_t*)calloc(1, ndims_padded);
    uint16_t rounded_up_nstripes = ndims_padded / stripe_sz;
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
            uint8_t* store_addr = ((uint8_t*)stripe_bitwidths) + v * vector_sz;
            _mm256_storeu_si256((__m256i*)store_addr, bitwidths);

            // compute and store masks
            __m256i masks = _mm256_shuffle_epi8(nbits_to_mask, raw_header);
            // printf("masks: "); dump_m256i(masks);
            uint8_t* store_addr2 = ((uint8_t*)data_masks) + v * vector_sz;
            _mm256_storeu_si256((__m256i*)store_addr2, masks);
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

        // ar::print(stripe_bitwidths, nstripes, "stripe_bitwidths");
        // ar::print(stripe_bitoffsets, nstripes, "stripe_bitoffsets");
        // printf("row sz bits: %d\n", row_sz_bits);

        // TODO see if prefetching and/or touching input and output cache
        // lines in order (instead of weird strided order below) helps
        // performance

        // unpack data for each stripe
        // for (size_t stripe = 0; stripe < nstripes; stripe++) {
        for (int stripe = nstripes - 1; stripe >= 0; stripe--) {
            uint32_t offset_bits = stripe_bitoffsets[stripe] & 0x07;
            uint32_t offset_bytes = stripe_bitoffsets[stripe] >> 3;

            uint64_t mask = data_masks[stripe];
            uint8_t nbits = stripe_bitwidths[stripe];

            int8_t* inptr = src + offset_bytes;
            uint8_t* outptr = dest + (stripe * stripe_sz);

            uint8_t total_bits = nbits + offset_bits;
            // printf("total bits: %d\n", total_bits);

            // printf("offset bytes, bits = %u, %u\n", offset_bytes, offset_bits);

            // this is the hot loop
            if (total_bits <= 64) { // input guaranteed to fit in 8B
                for (int i = 0; i < block_sz; i++) {
                    uint64_t packed_data = (*(uint64_t*)inptr) >> offset_bits;
                    // printf("packed_data "); dumpBytes(packed_data);
                    *(uint64_t*)outptr = _pdep_u64(packed_data, mask);
                    inptr += row_sz_bytes;
                    outptr += ndims;
                }
            } else { // input spans 9 bytes
                // printf(">>> executing the slow path!\n");
                uint8_t nbits_lost = total_bits - 64;
                for (int i = 0; i < block_sz; i++) {
                    uint64_t packed_data = (*(uint64_t*)inptr) >> offset_bits;
                    // printf("packed_data "); dumpBytes(packed_data);
                    packed_data |= (*(uint64_t*)(inptr + 8)) << (nbits - nbits_lost);
                    // printf("packed_data after OR "); dumpBytes(packed_data);
                    *(uint64_t*)outptr = _pdep_u64(packed_data, mask);
                    inptr += row_sz_bytes;
                    outptr += ndims;
                }
            }
            // printf("data we wrote:\n"); dumpBytes(dest, group_sz);
        }
        src += block_sz * group_sz_blocks * row_sz_bytes;
        dest += group_sz;
    }

    free(headers);
    free(data_masks);
    free(stripe_bitwidths);
    free(stripe_bitoffsets);

// memcpy_remainder:

    // copy over trailing data
    size_t remaining_len = orig_len % group_sz;
    // printf("remaining len: %lu\n", remaining_len);
    // printf("read bytes: %lu\n", remaining_len);
    // printf("remaining data: "); ar::print(src, remaining_len);
    memcpy(dest, src, remaining_len);

    return dest + remaining_len - orig_dest;
}
