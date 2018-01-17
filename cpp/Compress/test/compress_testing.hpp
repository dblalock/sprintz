//
//  compress_testing.hpp
//  Compress
//
//  Created by DB on 10/9/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

// TODO nothing in this file really has to be a macro anymore...

#ifndef compress_testing_h
#define compress_testing_h

#include "catch.hpp"
#include "eigen/Core"

#include "array_utils.hpp"
#include "testing_utils.hpp"

static const uint16_t kDefaultMinNdims = 1;
static const uint16_t kDefaultMaxNdims = 129;

#define TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC)                         \
    Vec_i8 compressed((SZ) * 3/2 + 64);                                     \
    Vec_u8 decompressed((SZ)+ 64);                                          \
    compressed += 0x55; /* poison memory */                                 \
    decompressed += 0xaa;                                                   \
    auto len = COMP_FUNC(raw.data(), (SZ), compressed.data());              \
    len = DECOMP_FUNC(compressed.data(), decompressed.data());              \
    CAPTURE(SZ);                                                            \
    REQUIRE(len == (SZ));                                                   \
    auto arrays_eq = ar::all_eq(raw.data(), decompressed.data(), (SZ));     \
    if (!arrays_eq) {                                                       \
        printf("\n**** Test Failed! ****\n");                               \
        for (uint64_t i = 0; i < (SZ); i++) {                               \
            if (raw.data()[i] != decompressed.data()[i]) {                  \
                printf("first disagreement at index: %llu", i);             \
                printf(" (input %lld != output %lld)\n",                    \
                    (long long)raw.data()[i],                               \
                    (long long)decompressed.data()[i]);                     \
                break;                                                      \
            }                                                               \
        }                                                                   \
        if ((SZ) < 100) {                                                   \
            auto input_as_str = ar::to_string(raw.data(), (SZ));            \
            auto output_as_str = ar::to_string(decompressed.data(), (SZ));  \
            printf("input:\t%s\n", input_as_str.c_str());                   \
            printf("output:\t%s\n", output_as_str.c_str());                 \
        }                                                                   \
    }                                                                       \
    REQUIRE(arrays_eq);

template<int bitwidth>
struct elemsize_traits {
    static_assert(bitwidth == 1 || bitwidth == 2, "Invalid bitwidth!");
};

template<> struct elemsize_traits<1> {
    using uint_t = uint8_t;
    using int_t = int8_t;
    using uvec_t = Vec_u8;
    using ivec_t = Vec_i8;
};

template<> struct elemsize_traits<2> {
    using uint_t = uint16_t;
    using int_t = int16_t;
    using uvec_t = Vec_u16;
    using ivec_t = Vec_i16;
};

template<int ElemSz, class RawT, class CompF, class DecompF>
static inline void test_compressor(const RawT& raw, CompF&& f_comp,
    DecompF&& f_decomp, const char* name="", bool check_overrun=false,
    size_t sz=0)
{
    using traits = elemsize_traits<ElemSz>;
    using uint_t = typename traits::uint_t;
    using UVec = typename traits::uvec_t;
    using IVec = typename traits::ivec_t;
    if (sz == 0) {
        sz = raw.size();
    }
    IVec compressed(sz * 3/2 + 64);
    uint16_t decomp_padding = 64;
    UVec decompressed(sz + decomp_padding);
    compressed.setZero();
    decompressed.setZero();
    // poison memory
    uint_t comp_poison = ElemSz == 1 ? 0x55 : 0x55 + 1024 + 2048;
    uint_t decomp_poison = ElemSz == 1 ? 0xaa : 0xaa + 512 + 4096;
    compressed += comp_poison;
    decompressed += decomp_poison;

    auto compressed_len = f_comp(raw.data(), sz, compressed.data());
    CAPTURE(compressed_len);
    auto decompressed_len = f_decomp(compressed.data(), decompressed.data());
    auto arrays_eq = ar::all_eq(raw.data(), decompressed.data(), sz);
    if (!arrays_eq) {
        printf("\n**** Test Failed: '%s' ****\n", name);
        for (uint64_t i = 0; i < sz; i++) {
            if (raw.data()[i] != decompressed.data()[i]) {
                printf("first disagreement at index: %llu", i);
                printf(" (input %lld != output %lld)\n",
                    (long long)raw.data()[i],
                    (long long)decompressed.data()[i]);
                break;
            }
        }
        if (sz < 100) {
            auto input_as_str = ar::to_string(raw.data(), sz);
            auto output_as_str = ar::to_string(decompressed.data(), sz);
            printf("input:\t%s\n", input_as_str.c_str());
            printf("output:\t%s\n", output_as_str.c_str());
        }
    }
    if (check_overrun) {
        bool fail = false;
        for (uint64_t i = 0; i < decomp_padding; i++) {
            if (decompressed(sz + i) != decomp_poison) {
                fail = true;
                printf("\n**** Test Failed: '%s' ****\n", name);
                printf("First wrote past end of decompress buffer " \
                    "at element %lld\n", i + sz);
                break;
            }
        }
        if (fail) {
            ar::print(decompressed.data() + sz, decomp_padding,
                "Data past end of buffer");
        }
    }
    CAPTURE(sz);
    REQUIRE(decompressed_len == sz);
    REQUIRE(arrays_eq);
}

// would be nice to separate tests into Catch sections, but Catch runs the
// whole enclosing test_case a number of times equal to the number of sections
// for mysterious reasons
#define TEST_SIMPLE_INPUTS(_SZ, COMP_FUNC, DECOMP_FUNC)                 \
    do {                                                                \
    size_t SZ = (size_t)(_SZ);                                          \
    Vec_u8 raw(SZ);                                                     \
    {                                                                   \
        for (int i = 0; i < SZ; i++) { raw(i) = i % 64; }               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) { raw(i) = (i + 64) % 128; }       \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) { raw(i) = (i + 96) % 256; }       \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 0;                      \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 62 + (i + 1) % 4;       \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 126 + (i + 1) % 4;      \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    {                                                                   \
        for (int i = 0; i < SZ; i++) {                                  \
            raw(i) = (i % 2) ? (i + 64) % 128 : 72;                     \
        }                                                               \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                    \
    }                                                                   \
    } while(0);

template<int ElemSz, class CompF, class DecompF>
void test_simple_inputs(size_t sz, CompF&& f_comp, DecompF&& f_decomp) {
    using UVec = typename elemsize_traits<ElemSz>::uvec_t;
    UVec raw(sz);
    for (int i = 0; i < sz; i++) { raw(i) = i % 64; }
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "i mod 64");

    for (int i = 0; i < sz; i++) { raw(i) = (i + 64) % 128; }
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "i + 64 mod 128");

    for (int i = 0; i < sz; i++) { raw(i) = (i + 96) % 256; }
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "i + 96 mod 256");

    for (int i = 0; i < sz; i++) {
        raw(i) = (i % 2) ? (i + 64) % 128 : 0;
    }
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "i + 96 mod 256");

    for (int i = 0; i < sz; i++) {
        raw(i) = (i % 2) ? (i + 64) % 128 : 72;
    }
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "i 72 or in [64,128)");

    for (int i = 0; i < sz; i++) {
        raw(i) = (i % 2) ? (i + 64) % 128 : 62 + (i + 1) % 4;
    }
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "i in [62,66) or [64,128)");

    for (int i = 0; i < sz; i++) {
        raw(i) = (i % 2) ? (i + 64) % 128 : 126 + (i + 1) % 4;
    }
    test_compressor<ElemSz>(raw, f_comp, f_decomp,
        "i in [126,130) or [64,128)");
}

#define TEST_SQUARES_INPUT(SZ, COMP_FUNC, DECOMP_FUNC)                  \
    do {                                                                \
        Vec_u8 raw((SZ));                                               \
        for (int i = 0; i < (SZ); i++) {                                \
            raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);             \
        }                                                               \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                  \
    } while(0);

template<int ElemSz, class CompF, class DecompF>
void test_squares_input(size_t sz, CompF&& f_comp, DecompF&& f_decomp) {
    using UVec = typename elemsize_traits<ElemSz>::uvec_t;
    UVec raw(sz);
    for (int i = 0; i < sz; i++) {
        raw(i) = (i % 16) * (i % 16) + ((i / 16) % 16);
    }
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "(i%16)^2 + ((i/16)%16)");
}


#define TEST_KNOWN_INPUT(SZ, COMP_FUNC, DECOMP_FUNC)                    \
    TEST_SQUARES_INPUT(SZ, COMP_FUNC, DECOMP_FUNC);                     \
    TEST_SIMPLE_INPUTS(SZ, COMP_FUNC, DECOMP_FUNC);

template<int ElemSz, class CompF, class DecompF>
void test_known_input(size_t sz, CompF&& f_comp, DecompF&& f_decomp) {
    test_squares_input<ElemSz>(sz, f_comp, f_decomp);
    test_simple_inputs<ElemSz>(sz, f_comp, f_decomp);
}


#define TEST_FUZZ(SZ, COMP_FUNC, DECOMP_FUNC)                               \
    do {                                                                    \
    srand(123);                                                             \
    Vec_u8 raw((SZ));                                                       \
    raw.setRandom();                                                        \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 2;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 2;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 2;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 2;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    raw /= 8;                                                               \
    {                                                                       \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    } while(0);

template<int ElemSz, class CompF, class DecompF>
void test_fuzz(size_t sz, CompF&& f_comp, DecompF&& f_decomp) {
    using UVec = typename elemsize_traits<ElemSz>::uvec_t;
    uint32_t shift = 0;
    if (ElemSz == 2) {
        shift = 1;
    } else if (ElemSz == 4) {
        shift = 2;
    }
    UVec raw(sz);
    srand(123);
    raw.setRandom();
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "fuzz 0-256");
    raw /= (2 << shift);
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "fuzz 0-128");
    raw /= (2 << shift);
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "fuzz 0-64");
    raw /= (2 << shift);
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "fuzz 0-32");
    raw /= (2 << shift);
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "fuzz 0-16");
    raw /= (8 << (shift * 2));
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "fuzz 0-2");
}


#define TEST_ZEROS(SZ, COMP_FUNC, DECOMP_FUNC)                              \
    do {                                                                    \
        Vec_u8 raw(SZ);                                                     \
        raw.setZero();                                                      \
        TEST_COMPRESSOR(SZ, COMP_FUNC, DECOMP_FUNC);                        \
    } while(0);

template<int ElemSz, class CompF, class DecompF>
void test_zeros(size_t sz, CompF&& f_comp, DecompF&& f_decomp) {
    using UVec = typename elemsize_traits<ElemSz>::uvec_t;
    UVec raw(sz);
    raw.setZero();
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "zeros");
}

#define TEST_SPARSE(SZ, COMP_FUNC, DECOMP_FUNC)                             \
    do {                                                                    \
    srand(123);                                                             \
    Vec_u8 orig((SZ));                                                      \
    Vec_u8 raw((SZ));                                                       \
    orig.setRandom();                                                       \
    {                                                                       \
        raw = orig / 200;                                                   \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    {                                                                       \
        raw = orig / 250;                                                   \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    {                                                                       \
        raw = orig / 254;                                                   \
        TEST_COMPRESSOR((SZ), COMP_FUNC, DECOMP_FUNC);                      \
    }                                                                       \
    } while(0);


template<int ElemSz, class CompF, class DecompF>
void test_sparse(size_t sz, CompF&& f_comp, DecompF&& f_decomp) {
    using UVec = typename elemsize_traits<ElemSz>::uvec_t;
    uint32_t denominator_shift = 8 * (ElemSz - 1);
    UVec orig(sz);
    UVec raw(sz);
    srand(123);
    orig.setRandom();
    raw = orig / (193 << denominator_shift);
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "sparse 56/256");
    raw = orig / (250 << denominator_shift);
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "sparse 6/256");
    raw = orig / (254 << denominator_shift);
    test_compressor<ElemSz>(raw, f_comp, f_decomp, "sparse 2/256");
}

#define TEST_COMP_DECOMP_PAIR(COMP_FUNC, DECOMP_FUNC)                       \
    do {                                                                    \
        vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,  \
            66, 71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17}; \
        SECTION("known input") {                                            \
            for (auto sz : sizes) {                                         \
                TEST_KNOWN_INPUT(sz, COMP_FUNC, DECOMP_FUNC);               \
            }                                                               \
        }                                                                   \
        SECTION("zeros") {                                                  \
            for (auto sz : sizes) {                                         \
                TEST_ZEROS(sz, COMP_FUNC, DECOMP_FUNC);                     \
            }                                                               \
        }                                                                   \
        SECTION("fuzz_multiple_sizes") {                                    \
            for (auto sz : sizes) {                                         \
                TEST_FUZZ(sz, COMP_FUNC, DECOMP_FUNC);                      \
            }                                                               \
        }                                                                   \
        SECTION("long fuzz") {                                              \
            TEST_FUZZ(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);             \
        }                                                                   \
        SECTION("long sparse fuzz") {                                       \
            TEST_SPARSE(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);           \
        }                                                                   \
    } while (0);

template<int ElemSz, class CompF, class DecompF>
static inline void test_codec(CompF&& f_comp, DecompF&& f_decomp) {
    vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,
        66, 71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17};

    for (auto sz : sizes) { test_known_input<ElemSz>(sz, f_comp, f_decomp); }
    for (auto sz : sizes) { test_zeros<ElemSz>(sz, f_comp, f_decomp); }
    for (auto sz : sizes) { test_fuzz<ElemSz>(sz, f_comp, f_decomp); }
    for (auto sz : sizes) { test_sparse<ElemSz>(sz, f_comp, f_decomp); }
    test_fuzz<ElemSz>(1024 * 1024 + 7, f_comp, f_decomp);
    test_sparse<ElemSz>(1024 * 1024 + 7, f_comp, f_decomp);
}


template<int ElemSz, class CompF, class DecompF>
static inline void test_codec_many_ndims(CompF f_comp, DecompF f_decomp,
    uint16_t min_ndims=kDefaultMinNdims, uint16_t max_ndims=kDefaultMaxNdims)
{
    using uint_t = typename elemsize_traits<ElemSz>::uint_t;
    using int_t = typename elemsize_traits<ElemSz>::int_t;
    // auto ndims_list = ar::range(min_ndims, max_ndims + 1);
    // for (auto _ndims : ndims_list) {
        // auto ndims = (uint16_t)_ndims;
    for (uint16_t ndims = min_ndims; ndims <= max_ndims; ndims++) {
        printf("---- ndims = %d\n", ndims);
        CAPTURE(ndims);
        auto comp = [ndims, &f_comp](const uint_t* src, size_t len, int_t* dest) {
            return f_comp(src, (uint32_t)len, dest, ndims);
        };
        // auto decomp = [](int8_t* src, uint8_t* dest) {
        //     return f_decomp(src, dest);
        // };
        test_codec<ElemSz>(comp, f_decomp);
    }
}

// we need a macro here because overloaded compress and decompress functions
// prevent the compiler from infering the types `CompF` and `DecompF` if we
// try to call `test_codec_many_ndims` directly
#define TEST_CODEC_NDIMS_RANGE(ELEM_SZ, F_COMP, F_DECOMP, MIN_NDIMS, MAX_NDIMS)\
    {                                                                        \
        using uint_t = typename elemsize_traits<ELEM_SZ>::uint_t;            \
        using int_t = typename elemsize_traits<ELEM_SZ>::int_t;              \
        auto comp = [](const uint_t* src, size_t len,                        \
                int_t* dest, uint16_t ndims)                                 \
        {                                                                    \
            return F_COMP(src, (uint32_t)len, dest, ndims);                  \
        };                                                                   \
        auto decomp = [](int_t* src, uint_t* dest) {                         \
            return F_DECOMP(src, dest);                                      \
        };                                                                   \
        return test_codec_many_ndims<ELEM_SZ>(                               \
            comp, decomp, MIN_NDIMS, MAX_NDIMS);                             \
    }

#define TEST_CODEC_MANY_NDIMS(ELEM_SZ, F_COMP, F_DECOMP)    \
    TEST_CODEC_NDIMS_RANGE(ELEM_SZ, F_COMP, F_DECOMP,       \
        kDefaultMinNdims, kDefaultMaxNdims)

#define TEST_CODEC_MANY_NDIMS_8b(F_COMP, F_DECOMP)    \
    TEST_CODEC_MANY_NDIMS(1, F_COMP, F_DECOMP)

#define TEST_CODEC_MANY_NDIMS_16b(F_COMP, F_DECOMP)    \
    TEST_CODEC_MANY_NDIMS(2, F_COMP, F_DECOMP)


#define TEST_COMP_DECOMP_PAIR_NO_SECTIONS(COMP_FUNC, DECOMP_FUNC)           \
    do {                                                                    \
        vector<int64_t> sizes {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64,  \
            66, 71, 72, 73, 127, 128, 129, 135, 136, 137, 4096, 4096 + 17}; \
        {                                                                   \
            for (auto sz : sizes) {                                         \
                TEST_KNOWN_INPUT(sz, COMP_FUNC, DECOMP_FUNC);               \
            }                                                               \
        }                                                                   \
        {                                                                   \
            for (auto sz : sizes) {                                         \
                TEST_ZEROS(sz, COMP_FUNC, DECOMP_FUNC);                     \
            }                                                               \
        }                                                                   \
        {                                                                   \
            for (auto sz : sizes) {                                         \
                TEST_FUZZ(sz, COMP_FUNC, DECOMP_FUNC);                      \
            }                                                               \
        }                                                                   \
        {                                                                   \
            TEST_FUZZ(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);             \
        }                                                                   \
        {                                                                   \
            TEST_SPARSE(1024 * 1024 + 7, COMP_FUNC, DECOMP_FUNC);           \
        }                                                                   \
    } while (0);

#endif /* compress_testing_h */
