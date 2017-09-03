//
//  test_sprintz.cpp
//  Compress
//
//  Created by DB on 7/3/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include <stdio.h>

#include "catch.hpp"
#include "eigen/Eigen"

#include "array_utils.hpp"
#include "sprintz.h"
#include "timing_utils.hpp"

#include "debug_utils.hpp" // TODO rm

template<class T> using Vec = Eigen::Array<T, Eigen::Dynamic, 1>;
using Vec_i8 = Vec<int8_t>;
using Vec_u8 = Vec<uint8_t>;
using Vec_i16 = Vec<int16_t>;
using Vec_u16 = Vec<uint16_t>;


TEST_CASE("smoke test", "[sanity]") {
    int x = 0;
    REQUIRE(x == 0);
}

TEST_CASE("building blocks", "[sanity]") {
    uint16_t sz = 1024;
    Vec_u8 raw(sz);
    raw.setRandom(); // random vals in [0, 255]

    Vec_i8 compressed(sz);
    Vec_u8 decompressed(sz);

    SECTION("naiveDelta") {
//        std::cout << raw.cast<uint16_t>();
        auto len = compress8b_naiveDelta(raw.data(), sz, compressed.data());
        REQUIRE(len == sz);
        auto len2 = decompress8b_naiveDelta(compressed.data(), len, decompressed.data());
        REQUIRE(len2 == sz);

        REQUIRE(ar::all_eq(raw, decompressed));
//        std::cout << raw.cast<int16_t>();
//        std::cout << "------\n";
//        std::cout << decompressed.cast<int16_t>();
    }

    SECTION("bitpack") {
        Vec_u8 compressed(sz);
//        for (uint8_t nbits = 1; nbits <= 1; nbits++) {
        for (uint8_t nbits = 1; nbits <= 8; nbits++) {
            raw.setRandom();
            raw /= (1 << (8 - nbits));
        
            auto len = compress8b_bitpack(raw.data(), sz, compressed.data(), nbits);
            REQUIRE(len == (sz / 8) * nbits);
            
//            std::cout << "raw: " << raw.cast<uint16_t>();
//            std::cout << "comp: " << compressed.cast<uint16_t>();
            
            auto len2 = decompress8b_bitpack(compressed.data(), len, decompressed.data(), nbits);
            REQUIRE(len2 == sz);
//
//            std::cout << "raw: " << raw.cast<uint16_t>();
//            std::cout << "decomp: " << decompressed.cast<uint16_t>();
            
            REQUIRE(ar::all_eq(raw, decompressed));
        }
    }
}

TEST_CASE("profile_bitpack_u8", "[profile][bitpack]") {
    uint64_t sz = 256 * 1024 * 1024;
    Vec_u8 raw_orig(sz);
    Vec_u8 raw(sz);
    raw.setRandom(); // random vals in [0, 255]
    
    Vec_u8 compressed(sz);
    Vec_u8 decompressed(sz);
    for (uint8_t nbits = 1; nbits <= 8; nbits++) {
        std::cout << "---- nbits: " << (uint16_t)nbits << "\n";
//        raw.setRandom();
        raw = raw_orig / (1 << (8 - nbits));
        
        
        uint64_t len = 0, len2 = 0;
        {
//            cputime_t _tstart(timeNow());
            volatile PrintTimer t("compress");
            len = compress8b_bitpack(raw.data(), sz, compressed.data(), nbits);
//            auto elapsed = durationMs(_tstart, timeNow());
//            std::cout << "compress time:\t" << elapsed << "\tms\n";
        }
        if (len != (sz / 8) * nbits) { std::cout << "compression error!\n"; }
//        REQUIRE(len == (sz / 8) * nbits);
        
        {
            volatile PrintTimer t("decompress");
            len2 = decompress8b_bitpack(compressed.data(), len, decompressed.data(), nbits);
        }
        if (len2 != sz) { std::cout << "decompresion error!\n"; }
//        REQUIRE(len2 == sz);
    }
}

//TEST_CASE("naiveDelta", "[sanity]") {
//    uint16_t sz = 256;
//    Vec_u8 raw(sz);
//    raw.setRandom();
//    raw *= 127;
//    Vec_i8 compressed(sz);
//    Vec_u8 decompressed(sz);
//
//    auto len = compress8b_naiveDelta(raw.data(), sz, compressed.data());
//    REQUIRE(len == sz);
//    auto len2 = decompress8b_naiveDelta(compressed.data(), sz, decompressed.data());
//    REQUIRE(len2 == sz);
//
//    REQUIRE(ar::all_eq(raw, decompressed));
//}
