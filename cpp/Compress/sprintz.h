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
    // #include "immintrin.h"
    #include "xmmintrin.h" // for _mm_movemask_pi8
#endif


uint16_t compressed_size_bound(uint16_t in_sz) {
    return (in_sz * 9) / 8 + 9; // at worst, all literals
}

// ------------------------------------------------ just delta encoding

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

// #define GATHER_MASK(nbytes) (((uint64_t)1) << (nbytes * 8)) - 1
//
// static const uint64_t kGatherMasks[9] = {
//     0,
//     GATHER_MASK(1), GATHER_MASK(2), GATHER_MASK(3), GATHER_MASK(4),
//     GATHER_MASK(5), GATHER_MASK(6), GATHER_MASK(7),
//     GATHER_MASK(7) + (GATHER_MASK(7) - 1)   // avoid overflow
// };


#define _TILE_BYTE(byte)                                                    \
    (byte << 0 | byte << 8 | byte << 16 | byte << 24 |                        \
    byte << 32 | byte << 40 | byte << 48 | byte << 56)

#define TILE_BYTE(byte) _TILE_BYTE(((uint64_t)byte))


static const uint64_t kBitpackMasks[9] = {
    0,
    TILE_BYTE(0x01), TILE_BYTE(0x03), TILE_BYTE(0x07), TILE_BYTE(0x0f),
    TILE_BYTE(0x1f), TILE_BYTE(0x3f), TILE_BYTE(0x7f), TILE_BYTE(0xff),
};

uint64_t compress8b_bitpack(const uint8_t* src, uint64_t in_sz, uint8_t* dest,
    uint8_t nbits)
{
    static const int block_sz = 8;
    // static const int group_sz = 4;
    assert(in_sz % block_sz == 0);
    uint64_t nblocks = in_sz / block_sz;

    uint8_t* orig_dest = dest;
    uint64_t mask = kBitpackMasks[nbits];
    // dest[0] = nbits;

    // std::cout << "using nbits: " << (uint16_t)nbits << "\n";
    std::cout << "using pext mask: ";
    dumpBits(mask);

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
    uint64_t mask = kBitpackMasks[nbits];

    // TODO try packing into simd reg to minimize number of stores; also
    // consider repeating this loop for each value of nbits because it
    // isn't actually making the unrolled offsets to src be immediates
    //
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

// ------------------------------------------------ just bit shuffling

// see
//  -https://github.com/kiyo-masui/bitshuffle/blob/master/src/bitshuffle_core.c
//  -http://www.hackersdelight.org/hdcodetxt/transpose8.c.txt

/* Transpose 8x8 bit array packed into a single quadword *x*.
 * *t* is workspace. */
#define TRANS_BIT_8X8_64_LE(x, t) {                                               \
        t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AALL;                          \
        x = x ^ t ^ (t << 7);                                               \
        t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCLL;                         \
        x = x ^ t ^ (t << 14);                                              \
        t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0LL;                         \
        x = x ^ t ^ (t << 28);                                              \
    }

// /* Transpose 8x8 bit array along the diagonal from upper right
//    to lower left */
// #define TRANS_BIT_8X8_64_BE(x, t) {                                            \
//         t = (x ^ (x >> 9)) & 0x0055005500550055LL;                          \
//         x = x ^ t ^ (t << 9);                                               \
//         t = (x ^ (x >> 18)) & 0x0000333300003333LL;                         \
//         x = x ^ t ^ (t << 18);                                              \
//         t = (x ^ (x >> 36)) & 0x000000000F0F0F0FLL;                         \
//         x = x ^ t ^ (t << 36);                                              \
//     }
//
//uint16_t compress8b_bitshuf_8b(const uint8_t* src, uint16_t in_sz,
//    int8_t* dest, uint8_t nbits)
//{
//    assert(in_sz % 8 == 0);
//    auto nblocks = in_sz / 8;
//    static const int nbits = 8;
//    for (uint16_t b = 0; b < nblocks; b++) {
//        // for (uint8_t i = 0; i < 8; i++) { // will be needed for nbits < 8
//            // dest[i] = 0;
//        // }
//        // TODO version that uses 64bit ops instead of 8b ops
//        // for (uint8_t i = nbits - 1; i >= 0; i--) { // note descending order
//        for (uint8_t bit = nbits - 1; bit >= 0; bit--) { // descending order
//            uint8_t mask = 1 << bit;
//            *dest = (src[0] & mask) >> bit | (src[1] & mask) >> bit |
//                    (src[2] & mask) >> bit | (src[3] & mask) >> bit |
//                    (src[4] & mask) >> bit | (src[5] & mask) >> bit |
//                    (src[6] & mask) >> bit | (src[7] & mask) >> bit;
//            dest++;
//        }
//        dest += 8;
//    }
//}
//uint16_t compress8b_bitshuf_sse(const uint8_t* src, uint16_t in_sz,
//    int8_t* dest, uint8_t nbits)
//{
//    assert(in_sz % 8 == 0);
//    auto nblocks = in_sz / 8;
//    static const int nbits = 8;
//    for (uint16_t b = 0; b < nblocks; b++) {
//
//// #if defined(SSE2) // needs to also imply 64_BIT is defined
//        uint64_t src64 = *(uint64_t*)&src;
//        const uint64_t low_bits64 = 0x0101010101010101LL;
//
//        // TODO replace loop with switch that jumps based on nbits
//
//        for (uint8_t bit = nbits - 1; bit >= 0; bit--) { // descending order
//            uint64_t mask = low_bits64 << bit;
//            auto tmp = (src64 ^ mask) << (7 - bit); // store in MSBs
//            *dest = (uint8_t) _mm_movemask_pi8(tmp);
//            dest++;
//        }
//        dest += 8;
//    }
//}
//
//uint16_t compress8b_bitshuf_64(const uint8_t* src, uint16_t in_sz,
//    int8_t* dest, uint8_t nbits)
//{
//    assert(in_sz % 8 == 0);
//    auto nblocks = in_sz / 8;
//    uint8_t max_bit = (nbits - 1) & 0x07;
//
//    // uint64_t e = 1;
//    // const int little_endian = *((uint8_t *) &e) == 1;
//
//    for (uint16_t b = 0; b < nblocks; b++) {
//
//#ifdef HAS_UNALIGNED_LOADS
//        uint64_t src64 = *(uint64_t*)&src;
//#else
//        uint64_t src64;
//        for (int i = 0; i < 8; i++) {
//            src64 = (src64 << 8) | src[i];
//        }
//#endif
//        uint64_t tmp64;
//        TRANS_BIT_8X8_64_LE(src64, tmp64);
//
//        for (uint8_t bit = nbits - 1; bit >= 0; bit--) { // descending order
//            uint64_t mask = low_bits64 << bit;
//
//
//            // auto tmp = (src64 ^ mask) >> bit; // store in LSBs to save a shift
//            // *dest = (int8_t) (tmp >> 0  | tmp >> 7  | tmp >> 14 | tmp >> 21 |
//            //                   tmp >> 28 | tmp >> 35 | tmp >> 42 | tmp >> 49);
//            dest++;
//        }
//        dest += 8;
//    }
//}
//
//uint16_t compress8b_bitshuf_8b(const uint8_t* src, uint16_t in_sz,
//    int8_t* dest, uint8_t nbits)
//{
//    assert(in_sz % 8 == 0);
//    auto nblocks = in_sz / 8;
//    uint8_t max_bit = (nbits - 1) & 0x07;
//    for (uint16_t b = 0; b < nblocks; b++) {
//        // TODO replace loop with switch(nbits)
//        for (uint8_t bit = max_bit; bit >= 0; bit--) { // descending order
//            uint8_t mask = 1 << bit;
//            *dest = (src[0] & mask) >> bit | (src[1] & mask) >> bit |
//                    (src[2] & mask) >> bit | (src[3] & mask) >> bit |
//                    (src[4] & mask) >> bit | (src[5] & mask) >> bit |
//                    (src[6] & mask) >> bit | (src[7] & mask) >> bit;
//            dest++;
//        }
//        dest += 8;
//    }
//}
//
//uint16_t decompress8b_bitshuf(const int8_t* src, uint16_t in_sz, uint8_t* dest)
//{
//
//}

#endif /* sprintz_h */
