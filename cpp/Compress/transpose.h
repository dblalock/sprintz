//
//  transpose.h
//  Compress
//
//  Created by DB on 10/14/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef transpose_h
#define transpose_h

#include <stdint.h>
#include "immintrin.h" // for pext, pdep



/* 2x8 (rowmajor) -> 8x2 (rowmajor) transpose.
 *
 * This transforms:
 *
 * a0 a1 a2 a3 a4 a5 a6 a7
 * b0 b1 b2 b3 b4 b5 b6 b7
 * c0 c1 c2 c3 c4 c5 c6 c7
 * d0 d1 d2 d3 d4 d5 d6 d7
 *
 * to
 *
 * a0 b0
 * a1 b1
 * a2 b2
 * a3 b3
 * a4 b4
 * a5 b5
 * a6 b6
 * a7 b7
 * c0 d0
 * c1 d1
 * c2 d2
 * c3 d3
 * c4 d4
 * c5 d5
 * c6 d6
 * c7 d7
 */
void transpose_2x8_8b(const uint8_t* src, uint8_t* dest) {
    // static const __m256i idxs= _mm256_setr_epi8(
    //     0, 2, 4, 6, 8, 10, 12, 14,
    //     1, 3, 5, 7, 9, 11, 13, 15,
    //     0, 2, 4, 6, 8, 10, 12, 14,
    //     1, 3, 5, 7, 9, 11, 13, 15);
    static const int a = 0;
    static const int b = 8;
    static const int c = 0;
    static const int d = 8;
    static const __m256i idxs= _mm256_setr_epi8(
        // first 16B of both input and output
        a+0, b+0,
        a+1, b+1,
        a+2, b+2,
        a+3, b+3,
        a+4, b+4,
        a+5, b+5,
        a+6, b+6,
        a+7, b+7,
        // second 16B of both input and output
        c+0, d+0,
        c+1, d+1,
        c+2, d+2,
        c+3, d+3,
        c+4, d+4,
        c+5, d+5,
        c+6, d+6,
        c+7, d+7);

    __m256i vsrc = _mm256_loadu_si256((const __m256i*)src);
    __m256i shuffled = _mm256_shuffle_epi8(vsrc, idxs);
    _mm256_storeu_si256((__m256i*)dest, shuffled);
}

/* 4x8 (rowmajor) -> 8x3 (rowmajor) transpose.
 *
 * This transforms:
 *
 * a0 a1 a2 a3 a4 a5 a6 a7
 * b0 b1 b2 b3 b4 b5 b6 b7
 * c0 c1 c2 c3 c4 c5 c6 c7
 *
 * to
 *
 * a0 b0 c0
 * a1 b1 c1
 * a2 b2 c2
 * a3 b3 c3
 * a4 b4 c4
 * a5 b5 c5
 * a6 b6 c6
 * a7 b7 c7
 * 0 0 0 0 0 0 0 0
 */
void transpose_3x8_8b(const uint8_t* src, uint8_t* dest) {
    static const int a = 0;
    static const int b = 8;
    static const int c = 16 - 16;

    // shuffle idxs for first 2 rows of input (broadcast to both lanes)
    static const __m256i idxs0 = _mm256_setr_epi8(
        // low lane of output (reads from first 2 rows of input)
        a+0, b+0, 0xff,
        a+1, b+1, 0xff,
        a+2, b+2, 0xff,
        a+3, b+3, 0xff,
        a+4, b+4, 0xff,
        a+5,
        // high lane of ouptput (also reads from first 2 rows of input)
             b+5, 0xff,
        a+6, b+6, 0xff,
        a+7, b+7, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);

    static const __m256i idxs1 = _mm256_setr_epi8(
        // low lane of output (reads from row 2 of input)
        0xff, 0xff, c+0,
        0xff, 0xff, c+1,
        0xff, 0xff, c+2,
        0xff, 0xff, c+3,
        0xff, 0xff, c+4,
        0xff,
        // high lane of output (also reads row 2 of input)
              0xff, c+5,
        0xff, 0xff, c+6,
        0xff, 0xff, c+7,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);

    __m256i vsrc0 = _mm256_loadu_si256((const __m256i*)src);
    __m256i first_two_rows = _mm256_permute2x128_si256(vsrc0, vsrc0, 0x00);
    __m256i third_row = _mm256_set1_epi64x(*(uint64_t*)(src + 16));

    __m256i shuffled0 = _mm256_shuffle_epi8(first_two_rows, idxs0);
    __m256i shuffled1 = _mm256_shuffle_epi8(third_row, idxs1);
    __m256i blended = _mm256_or_si256(shuffled0, shuffled1);

    _mm256_storeu_si256((__m256i*)dest, blended);
}

/* 4x8 (rowmajor) -> 8x4 (rowmajor) transpose.
 *
 * This transforms:
 *
 * a0 a1 a2 a3 a4 a5 a6 a7
 * b0 b1 b2 b3 b4 b5 b6 b7
 * c0 c1 c2 c3 c4 c5 c6 c7
 * d0 d1 d2 d3 d4 d5 d6 d7
 *
 * to
 *
 * a0 b0 c0 d0
 * a1 b1 c1 d1
 * a2 b2 c2 d2
 * a3 b3 c3 d3
 * a4 b4 c4 d4
 * a5 b5 c5 d5
 * a6 b6 c6 d6
 * a7 b7 c7 d7
 */
void transpose_4x8_8b(const uint8_t* src, uint8_t* dest) {
    static const int a = 0;
    static const int b = 8;
    static const int c = 16 - 16;
    static const int d = 24 - 16;

    // shuffle idxs for first 2 rows of input (broadcast to both lanes)
    static const __m256i idxs0 = _mm256_setr_epi8(
        // low lane of output (reads from first 2 rows of input)
        a+0, b+0, 0xff, 0xff,
        a+1, b+1, 0xff, 0xff,
        a+2, b+2, 0xff, 0xff,
        a+3, b+3, 0xff, 0xff,
        // high lane of ouptput (also reads from last 2 rows of input)
        0xff, 0xff, c+4, d+4,
        0xff, 0xff, c+5, d+5,
        0xff, 0xff, c+6, d+6,
        0xff, 0xff, c+7, d+7);

    // shuffle idxs for last 2 rows of input (broadcast to both lanes)
    static const __m256i idxs1 = _mm256_setr_epi8(
        // low lane of output (reads from last two rows of input)
        0xff, 0xff, c+0, d+0,
        0xff, 0xff, c+1, d+1,
        0xff, 0xff, c+2, d+2,
        0xff, 0xff, c+3, d+3,
        // high lane of output (reads from first two rows of input)
        a+4, b+4, 0xff, 0xff,
        a+5, b+5, 0xff, 0xff,
        a+6, b+6, 0xff, 0xff,
        a+7, b+7, 0xff, 0xff);

    __m256i vsrc = _mm256_loadu_si256((const __m256i*)src);
    __m256i swapped128_vsrc = _mm256_permute2x128_si256(vsrc, vsrc, 0x01);

    __m256i shuffled0 = _mm256_shuffle_epi8(vsrc, idxs0);
    __m256i shuffled1 = _mm256_shuffle_epi8(swapped128_vsrc, idxs1);
    __m256i blended = _mm256_or_si256(shuffled0, shuffled1);

    _mm256_storeu_si256((__m256i*)dest, blended);
}


#endif /* transpose_h */
