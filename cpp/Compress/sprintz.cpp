//
//  sprintz.c
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

//#include "debug_utils.hpp" // TODO rm

// #ifdef USE_X86_INTRINSICS
// #include "immintrin.h"
// #include "emmintrin.h"  // for _mm_set1_epi16
// #include "smmintrin.h"  // for _mm_minpos_epu16
// #endif


static constexpr uint64_t kHeaderMask8b = TILE_BYTE(0x07); // 3 ones
static constexpr uint64_t kHeaderMaskDynDelta8b = TILE_BYTE(0x0f); // 4 ones
//static constexpr uint64_t kHeaderMask16b = TILE_BYTE(0x0f); // 4 ones


// ------------------------------------------------ just delta encoding

uint16_t compressed_size_bound(uint16_t in_sz) {
    return (in_sz * 9) / 8 + 9; // at worst, all literals
}


/** Writes one frame to out storing the first in_sz bytes at in
 * @return the number of bytes written
 *
 * This is not actually a compression function; it just computes the first
 * discrete derivative so we can get our test infra and design patterns
 * up and running.
 *
 * Note 1: in_sz should be small enough that the return value doesn't
 * overflow
 * Note 2: to determine how large a buffer is needed for dest, one can
 * use compressed_size_bound(in_sz)
 */
uint16_t compress8b_naiveDelta(const uint8_t* src, uint16_t in_sz, int8_t* dest)
{
    if (in_sz == 0) {
        return 0;
    }
    *dest = *src;
    src++;
    dest++;
    for (int i = 1; i < in_sz; i++) {
        *dest = *src - *(src - 1);
        src++;
        dest++;
    }
    return in_sz;
}

uint16_t decompress8b_naiveDelta(const int8_t* src, uint16_t in_sz,
                                 uint8_t* dest)
{
    if (in_sz == 0) {
        return 0;
    }
    *dest = *src;
    src++;
    dest++;
    for (int i = 1; i < in_sz; i++) {
        *dest = *src + *(dest - 1);
        src++;
        dest++;
    }
    return in_sz;
}

// ------------------------------------------------ delta + bit packing simple

int64_t compress8b_delta_simple(uint8_t* src, size_t len, int8_t* dest,
                                bool write_size)
{
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 2;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    size_t nblocks = len / block_sz;
    size_t ngroups = len / group_sz;
    int8_t* orig_dest = dest;

    // store how long this is
    if (write_size) {
        *(uint64_t*)dest = len;
        dest += 8;
    }

    // figure out where header bytes end and packed values begin
    uint8_t* header_dest = (uint8_t*)dest;
    dest += (nblocks / 2);

    int8_t delta_buff[group_sz];
    uint8_t prev_val = 0;

    // for each pair of 2 blocks
    for (int g = 0; g < ngroups; g++) {
        for (int i = 0; i < group_sz; i++) {
            delta_buff[i] = (*src - prev_val);
            prev_val = *src;
            src++;
        }

        // write header
        uint8_t nbits0 = needed_nbits_i8x8((int8_t*)delta_buff + 8 * 0);
        uint8_t nbits1 = needed_nbits_i8x8((int8_t*)delta_buff + 8 * 1);
        nbits0 += nbits0 == 7; // 7b will be treated as 8b at decoder
        nbits1 += nbits1 == 7;
        uint8_t write_nbits0 = nbits0 - (nbits0 == 8);
        uint8_t write_nbits1 = nbits1 - (nbits1 == 8);
        *header_dest = (write_nbits0 | (write_nbits1 << 4));
        header_dest++;

        // write packed data
        uint64_t mask0 = kBitpackMasks8[nbits0];
        uint64_t mask1 = kBitpackMasks8[nbits1];
        uint64_t data0 = *(uint64_t*)(delta_buff + 8 * 0);
        uint64_t data1 = *(uint64_t*)(delta_buff + 8 * 1);

        *((uint64_t*)dest) = _pext_u64(data0, mask0);
        dest += nbits0;
        *((uint64_t*)dest) = _pext_u64(data1, mask1);
        dest += nbits1;
    }
    // final trailing samples just get copied with no compression
    size_t remaining_len = len % group_sz;
    // printf("trailing data: (%lu samples)\n", remaining_len);
    // for (int i = 0; i < remaining_len; i++) {
    //     printf("%d ", src[i]);
    // }
    // printf("\n");
    memcpy(dest, src, remaining_len);
    dest += remaining_len;

    return dest - orig_dest;
}

int64_t decompress8b_delta_simple(int8_t* src, uint8_t* dest, uint64_t orig_len)
{
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 2;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    uint8_t* orig_dest = dest;

    // TODO only use 6B for length and last 2B for other info, such
    // as whether anything took > 6 bits (shorter loop if so)

    // read in size of original data, if not provided
    if (orig_len < 1) {
        orig_len = *(uint64_t*)src;
        src += 8;
    }

    // figure out number of blocks and where packed data starts
    size_t nblocks = orig_len / block_sz;
    size_t ngroups = orig_len / group_sz;
    uint8_t* header_src = (uint8_t*)src;
    src += nblocks / 2;

    int8_t prev_val = 0;
    for (int g = 0; g < ngroups; g++) {
        // read header to get nbits for each block
        uint8_t header = *header_src;
        header_src++;
        uint8_t nbits0 = header & 0x0f;
        uint8_t nbits1 = header >> 4;
        nbits0 += nbits0 == 7;
        nbits1 += nbits1 == 7;

        uint64_t mask0 = kBitUnpackMasks8[nbits0];
        uint64_t mask1 = kBitUnpackMasks8[nbits1];
        int64_t deltas0 = _pdep_u64(*(uint64_t*)src, mask0);
        src += nbits0;
        int64_t deltas1 = _pdep_u64(*(uint64_t*)src, mask1);
        src += nbits1;

        // cumsum each block; the tricky part here is that we unpacked
        // everything into the most significant bits, so that we could
        // shift right arithmetic to get sign extension
        for (int shift = 56; shift >= 0; shift -= 8) {
            int64_t delta = (deltas0 << shift) >> (64 - nbits0);
            *dest = prev_val + (int8_t)delta;
            prev_val = *dest;
            dest++;
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            int64_t delta = (deltas1 << shift) >> (64 - nbits1);
            *dest = prev_val + (int8_t)delta;
            prev_val = *dest;
            dest++;
        }
    }
    size_t remaining_len = orig_len % group_sz;
    memcpy(dest, src, remaining_len);

    assert(orig_len == (dest + remaining_len - orig_dest));
    return dest + remaining_len - orig_dest;
}

// ------------------------------------------------ delta + bit packing offline

int64_t compress8b_delta(uint8_t* src, size_t len, int8_t* dest,
                         bool write_size)
{
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 8;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    static constexpr int nbits_sz_bits = 3;
    // static constexpr int delta_delay = 1;

    // printf("delta8: received data of length: %lu\n", len);

    // store how long this is
    if (write_size) {
        *(uint64_t*)dest = len;
        dest += 8;
    }
    // if (len <= 8) {
    //     // printf("len %lu <= 8; just memcpying\n", len);
    //     memcpy(dest, src, len);
    //     dest += len;

    //     printf("src data:\n");
    //     for (int i = 0; i < len; i++) { printf("%d", (int)src[i]); } printf("\n");
    //     printf("output data:\n");
    //     for (auto ptr = orig_dest; ptr < dest; ptr++) { printf("%d", (int)*ptr); } printf("\n");
    //     return dest - orig_dest;
    //     // return len + 8;
    // }

    // printf("src data:\n");
    // for (int i = 0; i < len; i++) { printf("%d ", (int)src[i]); } printf("\n");

   // for (int i = 0; i < delta_delay; i++) {
   //     *dest = 0;
   //     dest++;
   // }

    size_t cpy_len = MIN(8, len);
    memcpy(dest, src, cpy_len); // copy first 8B to simplify delta computation
    dest += cpy_len;
    src += cpy_len;
    len -= cpy_len;

    size_t ngroups = len / group_sz;
    size_t header_sz = ngroups ? 1 + (ngroups * group_sz_blocks * nbits_sz_bits) / 8 : 0;
    int8_t* orig_dest = dest;

    // printf("delta8: read back length: %lu\n", *(uint64_t*)(dest - 8));

    // figure out where header bytes end and packed values begin
    uint8_t* header_dest = (uint8_t*)dest;
    // src += (nblocks / group_sz_blocks) * 3; // 3 bits per block
    // byte after end of headers; 3 bits per block, 8bits/B; we also have
    // 1B of padding so that we can do 4B writes for each block headers
    dest += header_sz;

    // printf("delta8: header length: %lu (should be %lu)\n",
    //     (uint64_t)(dest - (int8_t*)header_dest),
    //     1 + (ngroups * group_sz_blocks * nbits_sz_bits) / 8);
#ifdef VERBOSE_COMPRESS
    uint64_t counts[9] = {0,0,0,0, 0,0,0,0, 0};
    int64_t delta_counts[17] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0};
    uint8_t prev_nbits = 0;
#endif

    uint64_t delta_buff_u64;  // requires block_sz = 8
    uint64_t nbits_buff_u64;  // requires group_sz_blocks = 8
    uint8_t* delta_buff = (uint8_t*)&delta_buff_u64;
    uint8_t* nbits_buff = (uint8_t*)&nbits_buff_u64;

    // for each group of blocks
     // uint8_t prev_val = 0;
    for (int g = 0; g < ngroups; g++) { // for each group

        for (int b = 0; b < group_sz_blocks; b++) { // for each block
            for (int i = block_sz - 1; i >= 0; i--) {
                delta_buff[i] = src[i] - src[i-1];
            }
            src += block_sz;
            // for (int i = 0; i < block_sz; i++) { // for each sample
            //     delta_buff[i] = (*src - prev_val);
            //     prev_val = *src;
            //     src++;
            // }

            // loop at the bottom lets us replace an op on reg,mem,
            // with an op on just two registers (which may or may
            // not help), but isn't safe for very first delta ever,
            // because it would read src[-1]; so just don't bother
            //
            // if (g == 0) { // first loop iter better get pulled out...
            //     for (int i = block_sz - 1; i >= 1; i--) { // for each sample
            //         delta_buff[i] = src[i] - src[i-1];
            //     }
            //     delta_buff[0] = src[0] - prev_val;
            //     src += block_sz;
            //     prev_val = src[-1];
            // } else {
            //     for (int i = block_sz - 1; i >= 0; i--) {
            //         delta_buff[i] = src[i] - src[i-1];
            //     }
            //     src += block_sz;
            // }

            // info for header
            uint8_t nbits = needed_nbits_i8x8((int8_t*)delta_buff);
            // uint8_t nbits = 8; // TODO rm
#ifdef VERBOSE_COMPRESS
            counts[nbits]++;
            uint8_t delta_nbits_idx;
            if (nbits >= prev_nbits) { // can be simplified but be explicit
                delta_nbits_idx = nbits - prev_nbits;
                delta_nbits_idx += 8;
            } else {
                delta_nbits_idx = prev_nbits - nbits;
                delta_nbits_idx = 8 - delta_nbits_idx;
            }
            delta_counts[delta_nbits_idx]++;
            prev_nbits = nbits;
#endif
            nbits -= (nbits == 8);
            nbits_buff[b] = nbits;

            // write out packed data
            uint64_t mask = kBitpackMasks8[nbits];
            *((uint64_t*)dest) = _pext_u64(delta_buff_u64, mask);
            dest += nbits + (nbits == 7);
        }
        // write out header for whole group; 3b for each nbits
        uint32_t packed_header = (uint32_t)_pext_u64(nbits_buff_u64, kHeaderMask8b);
        *(uint32_t*)header_dest = packed_header;
        header_dest += (group_sz_blocks * nbits_sz_bits) / 8;
    }
    // compress (up to 63) trailing samples using a smaller block size / memcpy
    size_t remaining_len = len % group_sz;
    memcpy(dest, src, remaining_len);

#ifdef VERBOSE_COMPRESS
    printf("------------------------ ");
    printf("fraction of size from header: %f\n", header_sz / (float)(dest + remaining_len - orig_dest));
    printf("delta: nbits counts: ");
    for (int i = 0; i <= 8; i++) { printf("%llu ", counts[i]); } printf("\n");
    printf("delta: change in nbits counts:\n");
    for (int i = 0; i <= 16; i++) { printf("%llu ", delta_counts[i]); } printf("\n");
    // printf("delta8: final compressed length: %lu\n", (size_t)(dest + remaining_len - orig_dest));;
#endif

    return dest + remaining_len - orig_dest;

    // // version where we compress tail with a smaller block size
    // size_t tail_len = compress8b_delta_simple(src, remaining_len, dest, false);
    // return dest + tail_len - orig_dest;
}

int64_t decompress8b_delta(int8_t* src, uint8_t* dest) {
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 8;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    static constexpr int nbits_sz_bits = 3;
    static constexpr int nbits_sz_mask = 0x07;
    // static constexpr int delta_delay = 1;
    uint8_t* orig_dest = dest;

    // int8_t* orig_src = src;  // TODO rm

    // read in size of original data
    uint64_t orig_len = *(uint64_t*)src;
    src += 8;

//    src += delta_delay;

    // if (orig_len <= 8) {
    //     memcpy(dest, src, orig_len);
    //     return orig_len;
    // }
    size_t cpy_len = MIN(8, orig_len);
    memcpy(dest, src, cpy_len); // copy first 8B to simplify delta computation
    dest += cpy_len;
    src += cpy_len;
    orig_len -= cpy_len;

    // figure out number of blocks and where packed data starts; we add
    // 1 extra to src so that 4B header writes don't clobber anything
    size_t ngroups = orig_len / group_sz;
    uint8_t* header_src = (uint8_t*)src;
    src += ngroups ? 1 + (ngroups * group_sz_blocks * nbits_sz_bits) / 8 : 0;

    // int8_t prev_val = 0;
    int8_t prev_val = header_src[-1]; // because src just got increased
    for (int g = 0; g < ngroups; g++) {
        // read header to get nbits for each block
        uint32_t header = *(uint32_t*)header_src;
        header_src += (group_sz_blocks * nbits_sz_bits) / 8;

        // read deltas for each block
        for (int b = 0; b < group_sz_blocks; b++) {
            uint8_t nbits = (header >> (nbits_sz_bits * b)) & nbits_sz_mask;
            uint64_t mask = kBitUnpackMasks8[nbits];
            int64_t deltas = _pdep_u64(*(uint64_t*)src, mask);
            nbits += nbits == 7;
            src += nbits;

            // cumsum deltas (stored in upper bits to get sign extension)
            for (int shift = 56; shift >= 0; shift -= 8) {
                int64_t delta = (deltas << shift) >> (64 - nbits);
                *dest = prev_val + (int8_t)delta;
                prev_val = *dest;
                dest++;
            }
        }
    }
    size_t remaining_orig_len = orig_len % group_sz;
    memcpy(dest, src, remaining_orig_len);

    // printf("orig len, remaining_orig_len = %llu, %lu:\n", orig_len, remaining_orig_len);
    // printf("saw compressed data:\n");
    // for (auto ptr = orig_src + 8; ptr < orig_src + 8 + orig_len + 8; ptr++) { printf("%d ", (int)*ptr); } printf("\n");
    // // for (auto ptr = src; ptr < src + 2; ptr++) { printf("%d ", (int)*ptr); } printf("\n");
    // printf("decompressed data:\n");
    // for (auto ptr = orig_dest; ptr < dest + remaining_orig_len; ptr++) { printf("%d ", (int)*ptr); } printf("\n");

    // assert(orig_len == (dest + remaining_orig_len - orig_dest));
    return dest + remaining_orig_len - orig_dest;
}


// ------------------------------------------------ double delta + bit packing

int64_t compress8b_doubledelta(uint8_t* src, size_t len, int8_t* dest,
                               bool write_size)
{
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 8;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    static constexpr int nbits_sz_bits = 3;
    size_t ngroups = len / group_sz;
    size_t header_sz = 1 + (ngroups * group_sz_blocks * nbits_sz_bits) / 8;
    int8_t* orig_dest = dest;

    // printf("------------------------ ");
    // printf("dbldelta8: received data of length: %lu\n", len);

    // store how long this is
    if (write_size) {
        *(uint64_t*)dest = len;
        dest += 8;
    }

    // figure out where header bytes end and packed values begin
    uint8_t* header_dest = (uint8_t*)dest;
    // src += (nblocks / group_sz_blocks) * 3; // 3 bits per block
    // byte after end of headers; 3 bits per block, 8bits/B; we also have
    // 1B of padding so that we can do 4B writes for each block headers
    dest += header_sz;

    // printf("dbldelta8: header length: %lu (should be %lu)\n",
    //     (uint64_t)(dest - (int8_t*)header_dest),
    //     1 + (ngroups * group_sz_blocks * nbits_sz_bits) / 8);
#ifdef VERBOSE_COMPRESS
    uint64_t counts[9] = {0,0,0,0,0,0,0,0,0};
#endif

    uint64_t delta_buff_u64;  // requires block_sz = 8
    uint64_t nbits_buff_u64;  // requires group_sz_blocks = 8
    uint8_t* delta_buff = (uint8_t*)&delta_buff_u64;
    uint8_t* nbits_buff = (uint8_t*)&nbits_buff_u64;

    // for each group of blocks
    uint8_t prev_val = 0;
    int8_t prev_delta = 0;
    for (int g = 0; g < ngroups; g++) { // for each group

        for (int b = 0; b < group_sz_blocks; b++) { // for each block
            for (int i = 0; i < block_sz; i++) { // for each sample
                int8_t delta = (*src - prev_val);
                delta_buff[i] = delta - prev_delta;
                prev_val = *src;
                prev_delta = delta;
                src++;
            }
            // info for header
            uint8_t nbits = needed_nbits_i8x8((int8_t*)delta_buff);
#ifdef VERBOSE_COMPRESS
            counts[nbits]++;
#endif
            nbits -= (nbits == 8);
            nbits_buff[b] = nbits;

            // write out packed data
            uint64_t mask = kBitpackMasks8[nbits];
            *((uint64_t*)dest) = _pext_u64(delta_buff_u64, mask);
            dest += nbits + (nbits == 7);
        }

        // write out header for whole group; 3b for each nbits
        uint32_t packed_header = (uint32_t)_pext_u64(nbits_buff_u64, kHeaderMask8b);
        *(uint32_t*)header_dest = packed_header;
        header_dest += (group_sz_blocks * nbits_sz_bits) / 8;
    }
    // compress (up to 63) trailing samples using a smaller block size / memcpy
    size_t remaining_len = len % group_sz;
    memcpy(dest, src, remaining_len);

#ifdef VERBOSE_COMPRESS
    printf("dbldelta8: nbits counts: ");
    for (int i = 0; i <= 8; i++) { printf("%llu ", counts[i]); } printf("\n");;
    // printf("dbldelta8: final compressed length: %lu\n", (size_t)(dest + remaining_len - orig_dest));
    printf("fraction of size from header: %f\n", header_sz / (float)(dest + remaining_len - orig_dest));
#endif
    return dest + remaining_len - orig_dest;
}

int64_t decompress8b_doubledelta(int8_t* src, uint8_t* dest) {
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 8;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    static constexpr int nbits_sz_bits = 3;
    static constexpr int nbits_sz_mask = 0x07;
    uint8_t* orig_dest = dest;

    // read in size of original data
    uint64_t orig_len = *(uint64_t*)src;
    src += 8;

    // figure out number of blocks and where packed data starts
    size_t ngroups = orig_len / group_sz;
    uint8_t* header_src = (uint8_t*)src;
    src += 1 + (ngroups * group_sz_blocks * nbits_sz_bits) / 8;

    int8_t prev_val = 0;
    int8_t prev_delta = 0;
    for (int g = 0; g < ngroups; g++) {
        // read header to get nbits for each block
        uint32_t header = *(uint32_t*)header_src;
        header_src += (group_sz_blocks * nbits_sz_bits) / 8;

        // read deltas for each block
        for (int b = 0; b < group_sz_blocks; b++) {
            uint8_t nbits = (header >> (nbits_sz_bits * b)) & nbits_sz_mask;
            uint64_t mask = kBitUnpackMasks8[nbits];
            int64_t errs = _pdep_u64(*(uint64_t*)src, mask);
            nbits += nbits == 7;
            src += nbits;

            // cumsum deltas (stored in upper bits to get sign extension)
            for (int shift = 56; shift >= 0; shift -= 8) {
                int8_t err = (int8_t)((errs << shift) >> (64 - nbits));
                int8_t delta = err + prev_delta;
                *dest = prev_val + (int8_t)delta;
                prev_val = *dest;
                prev_delta = delta;
                dest++;
            }
        }
    }
    size_t remaining_orig_len = orig_len % group_sz;
    memcpy(dest, src, remaining_orig_len);

    assert(orig_len == (dest + remaining_orig_len - orig_dest));
    return dest + remaining_orig_len - orig_dest;
}

// ------------------------------------------------ dyndelta + bit packing

int64_t compress8b_dyndelta(uint8_t* src, size_t len, int8_t* dest,
                            bool write_size)
{
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 8;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    static constexpr int nbits_sz_bits = 3;
    static constexpr int block_header_sz_bits = nbits_sz_bits + 1;
    size_t ngroups = len / group_sz;
    size_t header_sz = (ngroups * group_sz_blocks * block_header_sz_bits) / 8;
    int8_t* orig_dest = dest;

    // store how long this is
    if (write_size) {
        *(uint64_t*)dest = len;
        dest += 8;
    }

    // figure out where header bytes end and packed values begin
    uint8_t* header_dest = (uint8_t*)dest;
    // src += (nblocks / group_sz_blocks) * 3; // 3 bits per block
    // byte after end of headers; 3 bits per block, 8bits/B; we also have
    // 1B of padding so that we can do 4B writes for each block headers
    dest += header_sz;
#ifdef VERBOSE_COMPRESS
    uint64_t counts[9] = {0,0,0,0,0,0,0,0,0};
#endif

    uint64_t nbits_buff_u64;
    uint8_t* nbits_buff = (uint8_t*)&nbits_buff_u64;
    uint64_t delta_buff_u64;
    uint8_t* delta_buff = (uint8_t*)&delta_buff_u64;
    uint64_t double_delta_buff_u64;
    uint8_t* double_delta_buff = (uint8_t*)&double_delta_buff_u64;

    // for each group of blocks
    uint8_t prev_val = 0;
    int8_t prev_delta = 0;
    for (int g = 0; g < ngroups; g++) { // for each group
        for (int b = 0; b < group_sz_blocks; b++) { // for each block
            for (int i = 0; i < block_sz; i++) { // for each sample
                int8_t delta = (*src - prev_val);
                delta_buff[i] = delta;
                double_delta_buff[i] = delta - prev_delta;
                prev_val = *src;
                prev_delta = delta;
                src++;
            }
            // info for header
            uint8_t nbits_delta = needed_nbits_i8x8((int8_t*)delta_buff);
            uint8_t nbits_double_delta = needed_nbits_i8x8((int8_t*)double_delta_buff);
            uint8_t nbits = MIN(nbits_delta, nbits_double_delta);
#ifdef VERBOSE_COMPRESS
            counts[nbits]++;
#endif
            // uint8_t nbits = 8; // TODO rm
            // uint8_t nbits = nbits_delta; // TODO rm
            // uint8_t nbits = nbits_double_delta; // TODO rm

            uint8_t indicator = nbits < nbits_delta;
            // uint8_t indicator = 0; // TODO rm
            // uint8_t indicator = 1; // TODO rm
            nbits -= (nbits == 8);
            uint8_t nbits_write = nbits | (indicator << nbits_sz_bits);
            nbits_buff[b] = nbits_write;

            // write out packed data
            uint64_t mask = kBitpackMasks8[nbits];
            if (indicator) {
                *((uint64_t*)dest) = _pext_u64(double_delta_buff_u64, mask);
            } else {
                *((uint64_t*)dest) = _pext_u64(delta_buff_u64, mask);
            }
            dest += nbits + (nbits == 7);
        }

        // write out header for whole group; 3b for each nbits
        uint32_t packed_header = (uint32_t)_pext_u64(nbits_buff_u64, kHeaderMaskDynDelta8b);
        *(uint32_t*)header_dest = packed_header;
        header_dest += (group_sz_blocks * block_header_sz_bits) / 8;
        // if (g == 0) {printf("wrote packed header:\n"); dumpEndianBits(packed_header); }
    };
    // compress (up to 63) trailing samples using a smaller block size / memcpy
    size_t remaining_len = len % group_sz;
    memcpy(dest, src, remaining_len);

#ifdef VERBOSE_COMPRESS
    printf("dbldelta8: nbits counts: ");
    for (int i = 0; i <= 8; i++) { printf("%llu ", counts[i]); } printf("\n");;
    // printf("dbldelta8: final compressed length: %lu\n", (size_t)(dest + remaining_len - orig_dest));
    printf("fraction of size from header: %f\n", header_sz / (float)(dest + remaining_len - orig_dest));
#endif
    return dest + remaining_len - orig_dest;
}

int64_t decompress8b_dyndelta(int8_t* src, uint8_t* dest) {
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 8;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    static constexpr int nbits_sz_bits = 3;
    static constexpr int block_header_sz_bits = nbits_sz_bits + 1;
    static constexpr int nbits_sz_mask = 0x07;
    uint8_t* orig_dest = dest;

    // read in size of original data
    uint64_t orig_len = *(uint64_t*)src;
    src += 8;

    // figure out number of blocks and where packed data starts
    size_t ngroups = orig_len / group_sz;
    uint8_t* header_src = (uint8_t*)src;
    src += (ngroups * group_sz_blocks * block_header_sz_bits) / 8;

    int8_t prev_val = 0;
    int8_t prev_delta = 0;
    for (int g = 0; g < ngroups; g++) {
        // read header to get nbits for each block
        uint32_t header = *(uint32_t*)header_src;
        header_src += (group_sz_blocks * block_header_sz_bits) / 8;

        // if (g == 0) {printf("read packed header:\n"); dumpEndianBits(header); }

        // read deltas for each block
        for (int b = 0; b < group_sz_blocks; b++) {
            int8_t block_header = (header >> (block_header_sz_bits * b));
            uint8_t nbits = block_header & nbits_sz_mask;
            uint64_t mask = kBitUnpackMasks8[nbits];
            int64_t errs = _pdep_u64(*(uint64_t*)src, mask);
            nbits += nbits == 7;
            src += nbits;

            // uint8_t indicator = block_header >> nbits_sz_bits;  // dbl delta?
            uint8_t prev_delta_mask = (block_header & 0x08) ? 0xff : 0x00;
            // uint8_t prev_delta_mask = 0; // TODO rm

            // assert(prev_delta_mask == 0);

            // cumsum deltas (stored in upper bits to get sign extension)
            for (int shift = 56; shift >= 0; shift -= 8) {
                int8_t err = (int8_t)((errs << shift) >> (64 - nbits));
                int8_t delta = err + (prev_delta & prev_delta_mask);
                // int8_t delta = err;
                *dest = prev_val + delta;
                prev_val = *dest;
                prev_delta = delta;
                dest++;
            }
        }
    }
    size_t remaining_orig_len = orig_len % group_sz;
    memcpy(dest, src, remaining_orig_len);

    assert(orig_len == (dest + remaining_orig_len - orig_dest));
    return dest + remaining_orig_len - orig_dest;
}

