//
//  util.h
//  Compress
//
//  Created by DB on 11/4/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef util_h
#define util_h

#ifndef MAX
    #define MAX(x, y) ( ((x) > (y)) ? (x) : (y) )
#endif
#ifndef MIN
    #define MIN(x, y) ( ((x) < (y)) ? (x) : (y) )
#endif

#if __cpp_constexpr >= 201304
    #define CONSTEXPR constexpr
#else
    #define CONSTEXPR
#endif

#include "immintrin.h"  // TODO memrep impl without avx2


template<typename T, typename T2>
static CONSTEXPR inline T round_up_to_multiple(T x, T2 multipleof) {
    T remainder = x % multipleof;
    return remainder ? (x + multipleof - remainder) : x;
}

inline void memrep(uint8_t* dest, const uint8_t* src, int32_t in_nbytes,
                    int32_t ncopies)
{
    if (in_nbytes < 1 || ncopies < 1) { return; }

    static const int vector_sz = 32;

    // uint8_t* orig_dest = dest;

    // printf("------------------------\n");
    // printf("requested in_nbytes, ncopies = %d, %d\n", in_nbytes, ncopies);

    // ------------------------ nbytes > vector_sz
    if (in_nbytes > vector_sz) {
        size_t nvectors = (in_nbytes / vector_sz) + ((in_nbytes % vector_sz) > 0);
        size_t trailing_nbytes = in_nbytes % vector_sz;
        trailing_nbytes += trailing_nbytes == 0 ? vector_sz : 0;
        for (size_t i = 0; i < ncopies - 1; i++) {
            for (size_t v = 0; v < nvectors - 1; v++) {
                __m256i x = _mm256_loadu_si256(
                    (const __m256i*)(src + v * vector_sz));
                _mm256_storeu_si256((__m256i*)dest, x);
                dest += vector_sz;
            }
            __m256i x = _mm256_loadu_si256(
                (const __m256i*)(src + (nvectors - 1) * vector_sz));
            _mm256_storeu_si256((__m256i*)dest, x);
            dest += trailing_nbytes;
        }
        // for last copy, don't vectorize to avoid writing past end
        memcpy(dest, src, in_nbytes);
        return;
    }

    // ------------------------ 1 <= in_nbytes <= vector_sz

    uint32_t total_nvectors = (ncopies * in_nbytes) / vector_sz;

    // if data fills less than two vectors, just memcpy
    // if (total_nvectors <= (1 << 30)) { // TODO rm after debug
    // if (total_nvectors <= 2) { // TODO uncomment after debug
    if (total_nvectors < 1) {
        for (size_t i = 0; i < ncopies; i++) {
            memcpy(dest, src, in_nbytes);
            dest += in_nbytes;
        }
    }

    // populate a vector with as many copies of the data as possible
    __m256i v;
    switch (in_nbytes) {
    case 1:
        v = _mm256_set1_epi8(*(const uint8_t*)src); break;
    case 2:
        v = _mm256_set1_epi16(*(const uint16_t*)src); break;
    case 3:
        memcpy(dest, src, 3); memcpy(dest + 3, src, 3);
        memcpy(dest + 6, dest, 6); memcpy(dest + 12, dest, 12);
        memcpy(dest + 24, dest, 6); // bytes 24-30
        // memcpy(dest, src, 4); memcpy(dest + 3, src, 4);
        // memcpy(dest + 6, dest, 6); memcpy(dest + 12, dest, 12);
        v = _mm256_loadu_si256((const __m256i*)dest);
        // printf("initial stuff we wrote: "); dump_bytes(dest, 32);
        dest += 8 * in_nbytes;
        ncopies -= 8;
        break;
    case 4:
        v = _mm256_set1_epi32(*(const uint32_t*)src); break;
    case 5:
        memcpy(dest, src, in_nbytes);
        memcpy(dest + in_nbytes, dest, in_nbytes);
        memcpy(dest + 2*in_nbytes, dest, 2*in_nbytes);
        memcpy(dest + 4*in_nbytes, dest, 2*in_nbytes); // bytes 20-30
        v = _mm256_loadu_si256((const __m256i*)dest);
        dest += 6 * in_nbytes;
        ncopies -= 6;
        break;
    case 6:
        memcpy(dest, src, in_nbytes);
        memcpy(dest + in_nbytes, dest, in_nbytes);
        memcpy(dest + 2*in_nbytes, dest, 2*in_nbytes);
        memcpy(dest + 4*in_nbytes, dest, in_nbytes); // bytes 24-30
        v = _mm256_loadu_si256((const __m256i*)dest);
        dest += 5 * in_nbytes;
        ncopies -= 5;
        break;
    case 7:
        memcpy(dest, src, in_nbytes);
        memcpy(dest + in_nbytes, dest, in_nbytes);
        memcpy(dest + 2*in_nbytes, dest, 2*in_nbytes);
        v = _mm256_loadu_si256((const __m256i*)dest);
        dest += 4 * in_nbytes;
        ncopies -= 4;
        break;
    case 8:
        v = _mm256_set1_epi64x(*(const uint64_t*)src); break;
    case 9: case 10:
        memcpy(dest, src, in_nbytes);
        memcpy(dest + in_nbytes, dest, in_nbytes);
        memcpy(dest + 2*in_nbytes, dest, in_nbytes); // third copy
        v = _mm256_loadu_si256((const __m256i*)dest);
        dest += 3 * in_nbytes;
        ncopies -= 3;
        break;
    case 11: case 12: case 13: case 14: case 15:
        memcpy(dest, src, in_nbytes);
        memcpy(dest + in_nbytes, dest, in_nbytes);
        v = _mm256_loadu_si256((const __m256i*)dest);
        dest += 2 * in_nbytes;
        ncopies -= 2;
        break;
    case 16:
        v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)src));
        break;
    case 32:
        v = _mm256_loadu_si256((const __m256i*)src); break;
    default: // 17 through 31
        memcpy(dest, src, in_nbytes);
        v = _mm256_loadu_si256((const __m256i*)dest);
        dest += in_nbytes;
        ncopies -= 1;
        break;
    }

    // printf("vector we're copying: "); dump_m256i(v);

    // compute number of vectors to write (ncopies might have changed) and
    // write them
    total_nvectors = (ncopies * in_nbytes) / vector_sz;
    uint8_t copies_per_vector = vector_sz / in_nbytes;
    uint8_t stride_nbytes = copies_per_vector * in_nbytes;
    for (size_t i = 0; i < total_nvectors; i++) {
        _mm256_storeu_si256((__m256i*)dest, v);
        dest += stride_nbytes;
    }

    // copy remaining bytes using memcpy (to avoid writing past end)
    uint32_t written_ncopies = copies_per_vector * total_nvectors;
    uint32_t remaining_ncopies = ncopies - written_ncopies;
    // printf("total nvectors, copies per vector = %d, %d\n", total_nvectors, copies_per_vector);
    // printf("stride_nbytes, written_ncopies, remaining_ncopies = %d, %d, %d\n", stride_nbytes, written_ncopies, remaining_ncopies);
    for (uint32_t i = 0; i < remaining_ncopies; i++) {
        memcpy(dest, src, in_nbytes);
        dest += in_nbytes;
    }

    // printf("everything we wrote: "); dump_bytes(orig_dest, (int)(dest - orig_dest));
}


#endif /* util_h */
