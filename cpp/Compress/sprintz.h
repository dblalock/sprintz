//
//  sprintz.h
//  Compress
//
//  Created by DB on 7/3/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef sprintz_h
#define sprintz_h

#include <stdint.h>
#include <assert.h>

#include "debug_utils.hpp" // TODO rm

#define USE_X86_INTRINSICS
#define USE_AVX2

#ifdef USE_AVX2
    static_assert(__AVX2__, "AVX 2 is required! Try --march=native or -mavx2");
    #define USE_X86_INTRINSICS
#endif

#ifdef USE_X86_INTRINSICS
     #include "immintrin.h"
    #include "emmintrin.h"  // for _mm_set1_epi16
    #include "smmintrin.h"  // for _mm_minpos_epu16
//    #include "xmmintrin.h" // for _mm_movemask_pi8
#endif

#define max(x, y) ( ((x) > (y)) ? (x) : (y) )
#define min(x, y) ( ((x) < (y)) ? (x) : (y) )

static const uint8_t _NBITS_COST_I8[256] = {
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5,
    5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 2, 1, 0, 2, 3, 3, 4, 4, 4, 4, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8};
static const uint8_t* NBITS_COST_I8 = _NBITS_COST_I8 + 128; // so offsets can be signed

#define _TILE_BYTE(byte)                                                    \
    (byte << 0 | byte << 8 | byte << 16 | byte << 24 |                      \
    byte << 32 | byte << 40 | byte << 48 | byte << 56)

#define TILE_BYTE(byte) _TILE_BYTE(((uint64_t)byte))

#define _TILE_SHORT(short)                                                  \
    (short << 0 | short << 16 | short << 32 | short << 48)

#define TILE_SHORT(short) _TILE_SHORT(((uint64_t)short))

// this one allows specifying both seven and 8 bits
static const uint64_t kBitpackMasks_any_nbits[9] = {
    0,
    TILE_BYTE(0x01), TILE_BYTE(0x03), TILE_BYTE(0x07), TILE_BYTE(0x0f),
    TILE_BYTE(0x1f), TILE_BYTE(0x3f), TILE_BYTE(0x7f), TILE_BYTE(0xff),
};

// this one has idx 7 indicate 8 bits, so that it's always possible
// to encode enough
// NOTE to self: if you get SIGABRTs, you might be using index 8 here
static const uint64_t kBitpackMasks8[9] = {
    0,
    TILE_BYTE(0x01), TILE_BYTE(0x03), TILE_BYTE(0x07), TILE_BYTE(0x0f),
    TILE_BYTE(0x1f), TILE_BYTE(0x3f), TILE_BYTE(0xff), TILE_BYTE(0xff),
};

static const uint64_t kBitUnpackMasks8[9] = {
    0,
    TILE_BYTE(0x01) << 7, TILE_BYTE(0x03) << 6, TILE_BYTE(0x07) << 5,
    TILE_BYTE(0x0f) << 4, TILE_BYTE(0x1f) << 3, TILE_BYTE(0x3f) << 2,
    TILE_BYTE(0xff) << 0, TILE_BYTE(0xff) << 0,
};

// like above, second-highest value (here 0x7fff) is replaced with highest value
static const uint64_t kBitpackMasks16[17] = {
    0,
    TILE_SHORT(0x01), TILE_SHORT(0x03), TILE_SHORT(0x07), TILE_SHORT(0x0f),
    TILE_SHORT(0x1f), TILE_SHORT(0x3f), TILE_SHORT(0x7f), TILE_SHORT(0xff),
    TILE_SHORT(0x01ff), TILE_SHORT(0x03ff), TILE_SHORT(0x07ff), TILE_SHORT(0x0fff),
    TILE_SHORT(0x1fff), TILE_SHORT(0x3fff), TILE_SHORT(0xffff), TILE_SHORT(0xffff),
};

static constexpr uint64_t kHeaderMask8b = TILE_BYTE(0x07); // 3 ones
static constexpr uint64_t kHeaderMask16b = TILE_BYTE(0x0f); // 4 ones

// ------------------------------------------------ Computing needed nbits

uint8_t needed_nbits_epi16x8_v1(__m128i v) {
    // TODO even faster to zigzag and sub fromm 255 16 at once, then minpos
    // on each half

    static const __m128i max_vals = _mm_set1_epi16(0xffff);

    // zigzag encode
    v = _mm_xor_si128(_mm_srai_epi16(v, 15), _mm_slli_epi16(v, 1));
    // quasi-zigzag encode; lsb is wrong, but doesn't matter
    // v = _mm_xor_epi16(_mm128_srai_epi16(v, 15), _mm_slli_epi16(v, 1));

    __m128i subbed = _mm_subs_epu16(max_vals, v); // turn max value into min value

    // extract minimum value
    __m128i minval_and_idx = _mm_minpos_epu16(subbed);
    uint16_t minval = (uint16_t) _mm_extract_epi16(minval_and_idx, 0);
    // uint16_t idx = (uint16_t) _mm_extract_epi16(minval_and_idx, 1);
    uint16_t maxval = 0xffff - minval;

    return maxval ? _bit_scan_reverse(maxval) + 1 : 0; // bsr undefined if input all 0s
}

uint8_t needed_nbits_epi16x8(__m128i v) {
    int all_zeros = _mm_test_all_zeros(v, v);
    if (all_zeros) { return 0; }

    // get everything negative
    __m128i all_ones = _mm_cmpeq_epi16(v, v);
    v = _mm_xor_si128(v, _mm_cmpgt_epi16(v, all_ones));

    // invert and extract minimum value; now has a leading 0
    __m128i minval_and_idx = _mm_minpos_epu16(v);
    __m128i neg_minval_etc = _mm_xor_si128(minval_and_idx, all_ones);

    // this version uses andn to omit the above xor and mask upper 2B of u32
    uint16_t maxval = (uint16_t) _mm_extract_epi16(neg_minval_etc, 0);
    return 33 - __builtin_clz((uint32_t)maxval); // 33 because treats as uint32

    // this version has to do the xor, but doesn't have to andn
//    uint32_t maxval = (uint32_t) _mm_extract_epi32(neg_minval_etc, 0);
//    maxval = (maxval << 16) | (maxval >> 16); // compiles to ROL or ROR
//    return 17 - __builtin_clz(maxval);
}

uint8_t needed_nbits_epi8x8(__m128i v) {
    return needed_nbits_epi16x8(_mm_cvtepi8_epi16(v));
}

uint8_t needed_nbits_i16x8(int16_t* x) {
    __m128i v = _mm_loadu_si128((__m128i*)x);
    return needed_nbits_epi16x8(v);
}

uint8_t needed_nbits_i8x8(int8_t* x) {
    __m128i v = _mm_loadu_si128((__m128i*)x);
    return needed_nbits_epi8x8(v);
}

uint8_t needed_nbits_i8x8_simple(int8_t* x) {
    uint8_t max_nbits = NBITS_COST_I8[*x];
    int8_t* end = x + 8;
    x++;
    for ( ; x < end; x++) {
        max_nbits = max(max_nbits, NBITS_COST_I8[*x]);
    }
    return max_nbits;
}
uint8_t needed_nbits_i16x8_simple(int16_t* x) {
    int16_t val = *x;
    bool all_zeros = val == 0;

    val ^= val >> 15;  // flip bits if negative
    uint8_t min_nlz = __builtin_clz(val);
    int16_t* end = x + 8;
    x++;
    for ( ; x < end; x++) {
        val = *x;
        all_zeros &= val == 0;
        val ^= val >> 15;
        min_nlz = min(min_nlz, __builtin_clz(val));
    }
    return all_zeros ? 0: 33 - min_nlz;
}

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
        // if (*src > (*(src-1) + 127)) {
        //     std::cout << "this should totally overflow...\n";
        // }
        // if (*src < (*(src-1) - 128)) {
        //     std::cout << "this should totally overflow...\n";
        // }
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

// ------------------------------------------------ horz bit packing

uint64_t compress8b_bitpack(const uint8_t* src, uint64_t in_sz, uint8_t* dest,
    uint8_t nbits)
{
    static const int block_sz = 8;
    // static const int group_sz = 4;
    assert(in_sz % block_sz == 0);
    uint64_t nblocks = in_sz / block_sz;

    uint8_t* orig_dest = dest;
    uint64_t mask = kBitpackMasks_any_nbits[nbits];
    // dest[0] = nbits;

    // std::cout << "using nbits: " << (uint16_t)nbits << "\n";
    // std::cout << "using pext mask: ";
    // dumpBits(mask);

    // actually, no; vectorizing this part is nontrivial because what
    // we're writing is only a weird number of bytes; what we should
    // actually do is handle each nbits case separately, if we bother
    // optimizing this futher at all
    //
    // __m256i fourVals = _mm256_setzero_si256();
    // for (uint64_t b = 0; b < nblocks; b += 4) {
    //     uint64_t data0 = *(uint64_t*)(&src[block_sz*(b + 0)]);
    //     uint64_t data1 = *(uint64_t*)(&src[block_sz*(b + 1)]);
    //     uint64_t data2 = *(uint64_t*)(&src[block_sz*(b + 2)]);
    //     uint64_t data3 = *(uint64_t*)(&src[block_sz*(b + 3)]);
    //     uint64_t packed0 = _pext_u64(data0, mask);
    //     uint64_t packed1 = _pext_u64(data1, mask);
    //     uint64_t packed2 = _pext_u64(data2, mask);
    //     uint64_t packed3 = _pext_u64(data3, mask);

    //     dest += nbits;
    // }

    for (uint64_t b = 0; b < nblocks; b++) {
        uint64_t data = *(uint64_t*)(&src[block_sz*b]);
        uint64_t packed = _pext_u64(data, mask);
        *((uint64_t*)dest) = packed;
        dest += nbits;
    }
    return dest - orig_dest; // TODO
}
uint64_t decompress8b_bitpack(const uint8_t* src, uint64_t in_sz, uint8_t* dest,
    uint8_t nbits)
{
    static const int block_sz = 8;
    static const int group_sz = 4; // set this to 0 to not partially unroll loops
    assert(in_sz % nbits == 0);

    uint64_t nblocks = in_sz / nbits;
    uint64_t ngroups = group_sz > 0 ? nblocks / group_sz : 0;

    uint8_t* orig_dest = dest;
    uint64_t mask = kBitpackMasks_any_nbits[nbits];

    // TODO if more complicated stuff doesn't help (which it doesn't look
    // like it's going to), revert to simplest impl

#define MAIN_LOOP(nbits)                                                        \
    for (uint64_t g = 0; g < ngroups; g++) {                                    \
        for (int b = 0; b < group_sz; b++) {                                    \
            uint64_t unpacked = _pdep_u64(*(uint64_t*)src + (b * nbits), mask); \
            *((uint64_t*)(dest + (b * block_sz))) = unpacked;                   \
        }                                                                       \
        src += nbits * group_sz;                                                \
        dest += block_sz * group_sz;                                            \
    }                                                                           \
    for (uint64_t b = ngroups * group_sz * block_sz; b < nblocks; b++) {        \
        uint64_t unpacked = _pdep_u64(*(uint64_t*)src, mask);                   \
        *((uint64_t*)dest) = unpacked;                                          \
        src += nbits;                                                           \
        dest += block_sz;                                                       \
    }

#ifdef SWITCH_ON_NBITS
    switch(nbits) {
    case 1: MAIN_LOOP(1); break;
    case 2: MAIN_LOOP(2); break;
    case 3: MAIN_LOOP(3); break;
    case 4: MAIN_LOOP(4); break;
    case 5: MAIN_LOOP(5); break;
    case 6: MAIN_LOOP(6); break;
    case 7: MAIN_LOOP(7); break;
    case 8: MAIN_LOOP(8); break;
    default: break;
    }
#else
    // MAIN_LOOP(nbits);

#ifdef VECTOR_STORES
    __m256i fourVals = _mm256_setzero_si256();
    for (uint64_t g = 0; g < ngroups; g++) {
        for (int b = 0; b < group_sz; b++) {
            uint64_t unpacked = _pdep_u64(*(uint64_t*)src + (b * nbits), mask);
            fourVals = _mm256_insert_epi64(fourVals, unpacked, b);
        }
        _mm256_storeu_si256((__m256i*)dest, fourVals);
#else
    __m256i fourVals = _mm256_setzero_si256();
    for (uint64_t g = 0; g < ngroups; g++) {
        for (int b = 0; b < group_sz; b++) {
            uint64_t unpacked = _pdep_u64(*(uint64_t*)src + (b * nbits), mask);
            *((uint64_t*)(dest + (b * block_sz))) = unpacked;
        }
#endif
        src += nbits * group_sz;
        dest += block_sz * group_sz;
    }
    for (uint64_t b = ngroups * group_sz * block_sz; b < nblocks; b++) {
        uint64_t unpacked = _pdep_u64(*(uint64_t*)src, mask);
        *((uint64_t*)dest) = unpacked;
        src += nbits;
        dest += block_sz;
    }
#endif

#undef MAIN_LOOP

    return dest - orig_dest;
}

// ------------------------------------------------ delta + bit packing simple

int64_t compress8b_delta_simple(uint8_t* src, size_t len, int8_t* dest,
    bool write_size=true)
{
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 2;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    size_t nblocks = len / block_sz;
    size_t ngroups = len / group_sz;
    int8_t* orig_dest = dest;

    // assert(len % group_sz == 0);  // TODO handle stuff at end

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
    size_t nblocks_even = (nblocks / group_sz_blocks) * group_sz_blocks;
    // for (int b = 0; b < nblocks_even; b += 2) {
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

int64_t decompress8b_delta_simple(int8_t* src, size_t len, uint8_t* dest, uint64_t orig_len=0) {
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

// ------------------------------------------------ delta + bit packing for real

int64_t compress8b_delta(uint8_t* src, size_t len, int8_t* dest, bool write_size=true) {
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 8;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    static constexpr int nbits_sz_bits = 3;
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
    // src += (nblocks / group_sz_blocks) * 3; // 3 bits per block
    // byte after end of headers; 3 bits per block, 8bits/B; we also have
    // 1B of padding so that we can do 4B writes for each block headers
    dest += 1 + (ngroups * group_sz_blocks * nbits_sz_bits) / 8;

    uint64_t delta_buff_u64;  // requires block_sz = 8
    uint64_t nbits_buff_u64;  // requires group_sz_blocks = 8
    uint8_t* delta_buff = (uint8_t*)&delta_buff_u64;
    uint8_t* nbits_buff = (uint8_t*)&nbits_buff_u64;

    // for each group of blocks
    uint8_t prev_val = 0;
    for (int g = 0; g < ngroups; g++) { // for each group

        for (int b = 0; b < group_sz_blocks; b++) { // for each block
            // TODO delta computation can be vectorized
            for (int i = 0; i < block_sz; i++) { // for each sample
                delta_buff[i] = (*src - prev_val);
                prev_val = *src;
                src++;
            }
            // info for header  // TODO vectorize these operations
            uint8_t nbits = needed_nbits_i8x8((int8_t*)delta_buff);
            nbits -= (nbits == 8);
            nbits_buff[b] = nbits;

            // write out packed data
            uint64_t mask = kBitpackMasks8[nbits];
            *((uint64_t*)dest) = _pext_u64(delta_buff_u64, mask);
            // if (b == 0) {
            //     // printf("dest offset from orig dest: %ld\n", dest - orig_dest);
            //     printf("read back deltas:\n");
            //     uint64_t deltas2 = _pdep_u64(*((uint64_t*)dest), kBitpackMasks8[nbits]);
            //     dumpEndianBits(deltas2);
            // }
            dest += nbits + (nbits == 7);
        }
        // printf("wrote nbits, header bytes: ");
        // for (int i = 0; i < 8; i++) { printf("%d ", nbits_buff[i]); }
        // printf("\n");

        // write out header for whole group; 3b for each nbits
        uint32_t packed_header = (uint32_t)_pext_u64(nbits_buff_u64, kHeaderMask8b);
        *(uint32_t*)header_dest = packed_header;
        // dumpEndianBits(packed_header);
        header_dest += (group_sz_blocks * nbits_sz_bits) / 8;

        // printf("wrote header bytes:\n"); dumpEndianBits(packed_header);
    }
    // last header write clobbers first entry of data, since we write 4B but
    // each block only has a 3B header
    // *orig_data_dest = *orig_src; // XXX only works if first nbits == 8

    // compress (up to 63) trailing samples using a smaller block size / memcpy
    size_t remaining_len = len % group_sz;
    memcpy(dest, src, remaining_len);
    return dest + remaining_len - orig_dest;

    // // version where we compress tail with a smaller block size
    // size_t tail_len = compress8b_delta_simple(src, remaining_len, dest, false);
    // return dest + tail_len - orig_dest;
}

int64_t decompress8b_delta(int8_t* src, size_t len, uint8_t* dest) {
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 8;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    static constexpr int nbits_sz_bits = 3;
    static constexpr int nbits_sz_mask = 0x07;
    uint8_t* orig_dest = dest;

    // TODO only use 6B for length and last 2B for other info, such
    // as whether anything took > 6 bits (shorter loop if so)

    // read in size of original data
    uint64_t orig_len = *(uint64_t*)src;
    src += 8;

    // figure out number of blocks and where packed data starts; we add
    // 1 extra to src so that 4B header writes don't clobber anything
    size_t nblocks = orig_len / block_sz;
    size_t ngroups = orig_len / group_sz;
    uint8_t* header_src = (uint8_t*)src;
    src += 1 + (ngroups * group_sz_blocks * nbits_sz_bits) / 8;

    int8_t prev_val = 0;
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
            // src += nbits + (nbits == 7); // nbits of 7 counts as 8

            // cumsum deltas (stored in upper bits to get sign extension)
            for (int shift = 56; shift >= 0; shift -= 8) {
                int64_t delta = (deltas << shift) >> (64 - nbits);
                *dest = prev_val + (int8_t)delta;
                prev_val = *dest;
                dest++;
            }
        }
    }
    // size_t remaining_len = len % group_sz;
    // size_t remaining_orig_len = orig_len % group_sz;
    // size_t tail_len = decompress8b_delta_simple(
    //     src, remaining_len, dest, remaining_orig_len);
    // return de

    size_t remaining_orig_len = len % group_sz;
    memcpy(dest, src, remaining_orig_len);

    assert(orig_len == (dest + remaining_orig_len - orig_dest));
    return dest + remaining_orig_len - orig_dest;
}

// ------------------------------------------------ double delta + bit packing

int64_t compress8b_doubledelta(uint8_t* src, size_t len, int8_t* dest, bool write_size=true) {
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 8;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    static constexpr int nbits_sz_bits = 3;
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
    // src += (nblocks / group_sz_blocks) * 3; // 3 bits per block
    // byte after end of headers; 3 bits per block, 8bits/B; we also have
    // 1B of padding so that we can do 4B writes for each block headers
    dest += 1 + (ngroups * group_sz_blocks * nbits_sz_bits) / 8;

    uint64_t delta_buff_u64;  // requires block_sz = 8
    uint64_t nbits_buff_u64;  // requires group_sz_blocks = 8
    uint8_t* delta_buff = (uint8_t*)&delta_buff_u64;
    uint8_t* nbits_buff = (uint8_t*)&nbits_buff_u64;

    // for each group of blocks
    uint8_t prev_val = 0;
    int8_t prev_delta = 0;
    for (int g = 0; g < ngroups; g++) { // for each group

        for (int b = 0; b < group_sz_blocks; b++) { // for each block
            // TODO delta computation can be vectorized
            for (int i = 0; i < block_sz; i++) { // for each sample
                // delta_buff[i] = (*src - prev_val);
                // prev_val = *src;
                int8_t delta = (*src - prev_val);
                delta_buff[i] = delta - prev_delta;
                prev_val = *src;
                prev_delta = delta;
                src++;
            }
            // info for header  // TODO vectorize these operations
            uint8_t nbits = needed_nbits_i8x8((int8_t*)delta_buff);
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
    return dest + remaining_len - orig_dest;
}

int64_t decompress8b_doubledelta(int8_t* src, size_t len, uint8_t* dest) {
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
    size_t nblocks = orig_len / block_sz;
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
            // src += nbits + (nbits == 7); // nbits of 7 counts as 8
            nbits += nbits == 7;
            src += nbits;

            // cumsum deltas (stored in upper bits to get sign extension)
            for (int shift = 56; shift >= 0; shift -= 8) {
                // int64_t delta = (deltas << shift) >> (64 - nbits);
                // *dest = prev_val + (int8_t)delta;
                int8_t err = (int8_t)((errs << shift) >> (64 - nbits));
                int8_t delta = err + prev_delta;
                *dest = prev_val + (int8_t)delta;
                prev_val = *dest;
                prev_delta = delta;
                dest++;
            }
        }
    }
    size_t remaining_orig_len = len % group_sz;
    memcpy(dest, src, remaining_orig_len);

    assert(orig_len == (dest + remaining_orig_len - orig_dest));
    return dest + remaining_orig_len - orig_dest;
}

// ------------------------------------------------ Sprintz LL

uint64_t sprintz_ll_compress_8b(const uint8_t* src, uint64_t in_sz, uint8_t* dest,
    uint8_t nbits)
{
    static const int block_sz = 8;
    // static const int group_sz = 4;
    assert(in_sz % block_sz == 0);
    uint64_t nblocks = in_sz / block_sz;

    uint8_t* orig_dest = dest;
    uint64_t mask = kBitpackMasks8[nbits];
    // dest[0] = nbits;

    for (uint64_t b = 0; b < nblocks; b++) {

    }

    // std::cout << "using nbits: " << (uint16_t)nbits << "\n";
    std::cout << "using pext mask: ";
    dumpBits(mask);

    return -1; // TODO
}


#endif /* sprintz_h */
