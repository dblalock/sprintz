//
//  test_online.cpp
//  Compress
//
//  Created by DB on 4/2/20.
//  Copyright © 2020 D Blalock. All rights reserved.
//

#include "online.hpp"

#include "catch.hpp"
#include "compress_testing.hpp"
#include "testing_utils.hpp"

//#include "debug_utils.hpp" // TODO rm

template<class PredictorT> void _test_predictive_encoder() {
    auto comp = [](const uint16_t* src, size_t len, int16_t* dest) {
        return predictive_encode<PredictorT>(src, (len_t)len, dest);
    };
    auto decomp = [](const int16_t* src, size_t len, uint16_t* dest) {
        return predictive_decode<PredictorT>(src, (len_t)len, dest);
    };
    test_codec<sizeof(typename PredictorT::data_t)>(comp, decomp);
}

template<class PredictorT> void _debug_predictive_encoder(int len=2) {
    auto comp = [](const uint16_t* src, size_t len, int16_t* dest) {
        return predictive_encode<PredictorT>(src, (len_t)len, dest);
    };
    auto decomp = [](const int16_t* src, size_t len, uint16_t* dest) {
        return predictive_decode<PredictorT>(src, (len_t)len, dest);
    };
    test_squares_input<sizeof(typename PredictorT::data_t)>(len, comp, decomp);
}

TEST_CASE("online codecs (no compression) invertible", "[online][preproc]") {
    SECTION("16b") {
        SECTION("delta predictor") {
            _test_predictive_encoder<DeltaPredictor_u16>();
            // _debug_predictive_encoder<DeltaPredictor_u16>();
        }
        SECTION("double delta predictor") {
            _test_predictive_encoder<DoubleDeltaPredictor_u16>();
        }
        SECTION("triple delta predictor") {
            _test_predictive_encoder<TripleDeltaPredictor_u16>();
        }
        SECTION("moving avg predictor") {
            _test_predictive_encoder<MovingAvgPredictor_u16>();
        }
    }
}

TEST_CASE("sanity check online codecs", "[online][preproc]") {
    SECTION("16b") {
        int sz = 42;
        Vec_u16 raw(sz);
        Vec_i16 encoded(sz);
        int16_t val = 7;
        SECTION("constant input") {
            for (int i = 0; i < sz; i++) { raw(i) = val; }
            SECTION("delta predictor actually delta codes") {
                predictive_encode<DeltaPredictor_u16>(
                    raw.data(), sz, encoded.data());
                REQUIRE(encoded(0) == val);
                encoded(0) = 0;
                REQUIRE(encoded.maxCoeff() == 0);
            }
            SECTION("double delta predictor delta codes a const input") {
                predictive_encode<DoubleDeltaPredictor_u16>(
                    raw.data(), sz, encoded.data());
                REQUIRE(encoded(0) == val);
                encoded(0) = 0;
                REQUIRE(encoded.maxCoeff() == 0);
            }
        }
        SECTION("constantly increasing input") {
            for (int i = 0; i < sz; i++) { raw(i) = val + i * 3; }
            SECTION("double delta") {
                predictive_encode<DoubleDeltaPredictor_u16>(
                    raw.data(), sz, encoded.data());
                REQUIRE(encoded(0) == val);
                REQUIRE(encoded(1) == (int16_t)(raw(1) - raw(0)));
                encoded(0) = 0;
                encoded(1) = 0;
                REQUIRE(encoded.maxCoeff() == 0);
            }
            SECTION("triple delta") {
                predictive_encode<TripleDeltaPredictor_u16>(
                    raw.data(), sz, encoded.data());
                REQUIRE(encoded(0) == val);
                REQUIRE(encoded(1) == (int16_t)(raw(1) - raw(0)));
                // ar::print(raw.data(), MIN(sz, 4));
                // ar::print(encoded.data(), MIN(sz, 4));
                auto predicted = raw(1) + 2 * (raw(1) - raw(0));
                REQUIRE(encoded(2) == (int16_t)(raw(2) - predicted));
                encoded(0) = 0;
                encoded(1) = 0;
                encoded(2) = 0;
                REQUIRE(encoded.maxCoeff() == 0);
            }
        }
        SECTION("constantly decreasing input") {
            for (int i = 0; i < sz; i++) { raw(i) = val - i * 11; }
            SECTION("double delta") {
                predictive_encode<DoubleDeltaPredictor_u16>(
                raw.data(), sz, encoded.data());
                REQUIRE(encoded(0) == val);
                REQUIRE(encoded(1) == (int16_t)(raw(1) - raw(0)));
                encoded(0) = 0;
                encoded(1) = 0;
                REQUIRE(encoded.maxCoeff() == 0);
            }
            SECTION("triple delta") {
                predictive_encode<TripleDeltaPredictor_u16>(
                    raw.data(), sz, encoded.data());
                REQUIRE(encoded(0) == val);
                REQUIRE(encoded(1) == (int16_t)(raw(1) - raw(0)));
                auto predicted = raw(1) + 2 * (raw(1) - raw(0));
                REQUIRE(encoded(2) == (int16_t)(raw(2) - predicted));
                encoded(0) = 0;
                encoded(1) = 0;
                encoded(2) = 0;
                REQUIRE(encoded.maxCoeff() == 0);
            }
        }
    }
}


// template<class PredictorT> void _debug_predictive_encoder(int len=2) {
//     auto comp = [](const uint16_t* src, size_t len, int16_t* dest) {
//         return predictive_encode<PredictorT>(src, (len_t)len, dest);
//     };
//     auto decomp = [](const int16_t* src, size_t len, uint16_t* dest) {
//         return predictive_decode<PredictorT>(src, (len_t)len, dest);
//     };
//     test_squares_input<sizeof(typename PredictorT::data_t)>(len, comp, decomp);
// }

TEST_CASE("dynamic delta coding", "[online][preproc][current]") {
    Vec_u8 choices_buff_vec(1000*1000); // TODO use length it says it needs
    auto choices_buff = choices_buff_vec.data();
    int len = 128;
    SECTION("16b") {
        SECTION("length and choices buff kept separate") {
            int loss = Losses::MaxAbs;
            auto comp = [choices_buff, loss](const uint16_t* src, size_t len,
                                             int16_t* dest)
            {
                return dynamic_delta_zigzag_encode_u16(
                    src, (len_t)len, dest, choices_buff, loss);
            };
            auto decomp = [choices_buff](const int16_t* src, size_t len,
                                         uint16_t* dest)
            {
                return dynamic_delta_zigzag_decode_u16(
                    src, (len_t)len, dest, choices_buff);
            };
            // test_squares_input<2>(len, comp, decomp);
            // test_squares_input<2>(len, comp, decomp);
            test_codec<2>(comp, decomp);
        }
        SECTION("length and choices written in buff") {
            test_codec<2>(dynamic_delta_pack_u16, dynamic_delta_unpack_u16);
        }

        // _debug_predictive_encoder()

        // TODO pick up here




    }
}
