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
    TILE_BYTE(0x1f), TILE_BYTE(0x3f), TILE_BYTE(0x7f), TILE_BYTE(0xff),
};

// like above, second-highest value (here 0x7fff) is replaced with highest value
static const uint64_t kBitpackMasks16[17] = {
    0,
    TILE_SHORT(0x01), TILE_SHORT(0x03), TILE_SHORT(0x07), TILE_SHORT(0x0f),
    TILE_SHORT(0x1f), TILE_SHORT(0x3f), TILE_SHORT(0x7f), TILE_SHORT(0xff),
    TILE_SHORT(0x01ff), TILE_SHORT(0x03ff), TILE_SHORT(0x07ff), TILE_SHORT(0x0fff),
    TILE_SHORT(0x1fff), TILE_SHORT(0x3fff), TILE_SHORT(0x7fff), TILE_SHORT(0xffff),
};


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

    // printf("input vector:\n");
    // dump_m128i(v);

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

//    printf("maxval = %d, minpos = %d\n", (maxval << 16) >> 16, _mm_extract_epi32(minval_and_idx, 1));
//    dumpEndianBits(maxval);
//    printf("maxval << 16:\n");
//    dumpEndianBits(maxval << 16);
//    printf("nlz: %d\n", __builtin_clz(maxval << 16));

    // rotate bits left by 16; we don't just shift so that bit 15 is
    // guaranteed to be a 1, and we therefore don't fail when maxval == 0
//    __asm__ ("roll %1, %0" : "+g" (maxval) : "cI" ((uint8_t)16));
//    __asm__ ("roll $16, %0" : "+g" (maxval));
//    maxval = (maxval << 16) | (maxval >> 16); // compiles to ROL or ROR

//    printf("maxval rotated:\n");
//    dumpEndianBits(maxval);

//    return 17 - __builtin_clz(maxval);
}

uint8_t needed_nbits_epi8x8(__m128i v) {
    return needed_nbits_epi16x8(_mm_cvtepi8_epi16(v));

    // unpack low half of v into epi16s and use epi16 func
    // v = _mm_unpacklo_epi8(v, _mm_xor_si128(v, v));
    // v = _mm_unpacklo_epi8(_mm_xor_si128(v, v), v);
    // uint8_t nbits_16 = needed_nbits_epi16x8(v)
    // return max(0, ((int8_t)nbits_16) - 8);
}

uint8_t needed_nbits_i16x8(int16_t* x) {
    // printf("bits, treating it as int array:\n");
    // for (int i = 0; i < 8; i++) {
    //     printf("%d: ", x[i]);
    //     dumpEndianBits(x[i]);
    // }

    __m128i v = _mm_loadu_si128((__m128i*)x);
    // printf("bits right after loading input:\n");
    // dump_m128i(v);
    return needed_nbits_epi16x8(v);
}

uint8_t needed_nbits_i8x8(int8_t* x) {
    __m128i v = _mm_loadu_si128((__m128i*)x);
    return needed_nbits_epi8x8(v);
}

uint8_t needed_nbits_i8x8_simple(int8_t* x) {
    uint8_t max_nbits = NBITS_COST_I8[*x];
    int8_t* end = x + 8;
    ++x;
    for ( ; x < end; ++x) {
        max_nbits = max(max_nbits, NBITS_COST_I8[*x]);
    }
    return max_nbits;
}
uint8_t needed_nbits_i16x8_simple(int16_t* x) {
    int16_t val = *x;
    bool all_zeros = val == 0;

    // printf("val, xord val:\n");
    // dumpBits(val);
    val ^= val >> 15;  // flip bits if negative
    // dumpBits(val);

    uint8_t min_nlz = __builtin_clz(val);
    int16_t* end = x + 8;
    x++;
    for ( ; x < end; ++x) {
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

// ------------------------------------------------ delta + bit packing

int64_t compress8b_delta(uint8_t* src, size_t len, int8_t* dest) {

    // TODO blocks of 8 instead of 2 so that headers don't waste bits
    // -can just do a loop and then pext to get the header bytes

    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 2;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    assert(len % group_sz == 0);  // TODO handle stuff at end
    size_t nblocks = len / block_sz;
    int8_t* orig_dest = dest;

    // store how long this is
    *(uint64_t*)dest = len;
    dest += 8;

    // figure out where header bytes end and packed values begin
    uint8_t* header_dest = (uint8_t*)dest;
    dest += (nblocks / 2);
//    int8_t* dest = dest + (nblocks / 2);

    // TODO shift stuff straight into u64s; don't touch L1 cache
    uint8_t delta_buff[group_sz];
    uint8_t prev_val = 0;

    // for each pair of 2 blocks
    size_t nblocks_even = (nblocks / group_sz_blocks) * group_sz_blocks;
    for (int b = 0; b < nblocks_even; b += 2) {
        // delta_buff[0] = *src - prev_val;
        // prev_val = *src;
        // src++;
        // if (b == 2) printf("writing deltas:\n");
        for (int i = 0; i < group_sz; i++) {
            delta_buff[i] = (*src - prev_val);
            // if (b == 2) printf("%d ", delta_buff[i]);
            prev_val = *src;
            src++;
        }
        // if (b == 2) printf("\n");

        uint8_t nbits0 = needed_nbits_i8x8((int8_t*)delta_buff + 8 * 0);
        uint8_t nbits1 = needed_nbits_i8x8((int8_t*)delta_buff + 8 * 1);
        // nbits0 -= nbits0 == 8; // map 8 to 7, since only 3b
        // nbits1 -= nbits1 == 8;
        nbits0 += nbits0 == 7; // 7b will be treated as 8b at decoder
        nbits1 += nbits1 == 7;

        // uint8_t nbits0 = 8, nbits1 = 8; // TODO rm

        // write headers  // TODO write 7s if nbits is 8
        uint8_t write_nbits0 = nbits0 - (nbits0 == 8);
        uint8_t write_nbits1 = nbits1 - (nbits1 == 8);
        *header_dest = (write_nbits0 | (write_nbits1 << 4));
        // *header_dest = (nbits0 | (nbits1 << 4));
        header_dest++;

        // printf("nbits = %d, %d; wrote header:\n", nbits0, nbits1);
        // dumpEndianBits(*(header_dest - 1));

        // write packed data
        uint64_t mask0 = kBitpackMasks8[nbits0];
        uint64_t mask1 = kBitpackMasks8[nbits1];
        uint64_t data0 = *(uint64_t*)(delta_buff + 8 * 0);
        uint64_t data1 = *(uint64_t*)(delta_buff + 8 * 1);

        // printf("deltas0, deltas1:\n");
        // dumpEndianBits(data0);
        // dumpEndianBits(data1);
        // printf("deltas0, deltas1, data1 with mask:\n");
        // dumpEndianBits(_pext_u64(data1, mask1));

        // if (b == 2) printf("immediately read back deltas:\n");
        *((uint64_t*)dest) = _pext_u64(data0, mask0);
        // uint64_t deltas0 = _pdep_u64(*((uint64_t*)dest), mask0);
        dest += nbits0;
        *((uint64_t*)dest) = _pext_u64(data1, mask1);
        // uint64_t deltas1 = _pdep_u64(*((uint64_t*)dest), mask1);
        dest += nbits1;

        // if (b == 2) dumpEndianBits(deltas0);
        // if (b == 2) dumpEndianBits(deltas1);

        // if (b == 2) {
        //     printf("read back deltas:\n");
        //     uint64_t deltas0 = _pdep_u64(*(uint64_t*)(dest - nbits0 - nbits1), mask0);
        //     uint64_t deltas1 = _pdep_u64(*(uint64_t*)(dest - nbits1), mask1);
        //     dumpEndianBits(deltas0);
        //     dumpEndianBits(deltas1);
        // }

        // printf("wrote second block to dest:\n");
        // dumpEndianBits(*((uint64_t*)(dest - nbits1)));
    }
    return dest - orig_dest;
}

int64_t decompress8b_delta(int8_t* src, size_t len, uint8_t* dest) {
    static constexpr int block_sz = 8;
    static constexpr int group_sz_blocks = 2;
    static constexpr int group_sz = group_sz_blocks * block_sz;
    uint8_t* orig_dest = dest;

    // TODO only use 6B for length and last 2B for other info, such
    // as whether anything took > 6 bits (shorter loop if so)

    // read in size of original data
    uint64_t orig_len = *(uint64_t*)src;
    src += 8;
    assert(orig_len % group_sz == 0);  // TODO handle stuff at end

    // figure out number of blocks and where packed data starts
    size_t nblocks = orig_len / block_sz;
    // printf("decomp sees header:\n");
    // dumpEndianBits(*src);
    int8_t* packed_src = src + (nblocks / 2);

    // int8_t deltas[16];
    int8_t prev_val = 0;

    size_t nblocks_even = (nblocks / group_sz_blocks) * group_sz_blocks;
    for (int b = 0; b < nblocks_even; b += 2) {
        // read header to get nbits for each block
        uint8_t header = *src;
        src++;
        uint8_t nbits0 = header & 0x0f;
        uint8_t nbits1 = header >> 4;
        nbits0 += nbits0 == 7;
        nbits1 += nbits1 == 7;

        // unpack deltas
        uint64_t mask0 = kBitpackMasks8[nbits0];
        uint64_t mask1 = kBitpackMasks8[nbits1];
        uint64_t deltas0 = _pdep_u64(*(uint64_t*)packed_src, mask0);
        packed_src += nbits0;
        uint64_t deltas1 = _pdep_u64(*(uint64_t*)packed_src, mask1);
        packed_src += nbits1;

        // printf("unpacked deltas:\n");
        // dumpEndianBits(deltas0);
        // dumpEndianBits(deltas1);

        // printf("got nbits sizes: %d, %d\n", nbits0, nbits1);

        // if (b == 2) {
            // printf("got nbits sizes: %d, %d\n", nbits0, nbits1);
            // printf("using masks:\n");
            // dumpEndianBits(mask0);
            // dumpEndianBits(mask1);
        // }

        // cumsum each block
        // if (b == 2) printf("reading deltas:\n");
        for (int shift = 0; shift < 64; shift += 8) {
            // if (b == 2) printf("%d ", (int8_t)((deltas0 >> shift) & 0xff));
            *dest = prev_val + ((deltas0 >> shift) & 0xff);
            prev_val = *dest;
            // *dest = ((deltas0 >> shift) & 0xff);
            dest++;
        }
        for (int shift = 0; shift < 64; shift += 8) {
            // if (b == 2) printf("%d ", (int8_t)((deltas1 >> shift) & 0xff));
            *dest = prev_val + ((deltas1 >> shift) & 0xff);
            prev_val = *dest;
            // *dest = ((deltas1 >> shift) & 0xff);
            dest++;
        }
        // if (b == 2) printf("\n");
    }
    return dest - orig_dest;
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
