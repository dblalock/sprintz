//
//  signal.hpp
//  Compress
//
//  Created by DB on 4/29/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef signal_h
#define signal_h

#include <chrono>
#include <stdint.h>
#include <tuple>
#include <vector>

#include "timing_utils.hpp"

using std::tuple;
using std::vector;

using timestamp_t = int32_t;


//static const union {
//    unsigned char bytes[4];
//    uint32_t value;
//    
//    enum {
//        __ORDER_LITTLE_ENDIAN__ = 0x03020100ul,
//        __ORDER_BIG_ENDIAN__ = 0x00010203ul,
//        __ORDER_PDP_ENDIAN__ = 0x01000302ul
//    };
//} host_order =
//{ { 0, 1, 2, 3 } };
//
//#define __BYTE_ORDER__ (host_order.value)


timestamp_t now() {
    auto ms = std::chrono::duration_cast< std::chrono::milliseconds >(
                                                                      std::chrono::system_clock::now().time_since_epoch()).count();
    return static_cast<timestamp_t>(ms);
}

static inline bool is_big_endian() {
    #define IS_BIG_ENDIAN ((*(uint16_t *)"\0\xff") < 0x100)
    return IS_BIG_ENDIAN;
}

// TODO codecs to have:
//  -uint32 start timestamp, uint32 delta in timestamps, raw dump of values
//  -raw dump of timestamps, then raw dump of values
//  -raw dump of timestamps, (count, value) pairs
//  -uint32 start timestamp, {delta, double delta} timestamps, {raw, delta, double delta} values
//
//  -fancier ones (learned FIR, lemire, etc) later
//  -also option to lz4 the whole buff after everything else
//  -and encrypt the compressed buff
//  -at some point, have LUT for all strings and/or tokens we've seen based on
//  relative frequencies
//  -prolly also store min/mean/max in header so these queries are crazy fast
//  -would be really nice to have ability to resample at uniform interval so
//  we don't have to store raw timestamps (this is lossy, but good enough)
//      -needs some way to handle missingness though
//
//  -so really, we mostly want {raw, delta, double delta} dump/load funcs



class SignalHeader {
//    bool is_big_endian;
    uint8_t encoding_algo;
    uint8_t num_signals;
    uint32_t length;
};

template<class T>
class Signal {
    using DataT = std::tuple<T, timestamp_t>;
    vector<DataT> _data;

public:

    explicit Signal() = default;

    // deserialize
    explicit Signal(const char* buff, char* scratch=nullptr) {
//        buff[0] = is_big_endian();
        auto& header_ptr = *reinterpret_cast<const SignalHeader*>(buff);
        auto in_ptr = buff + sizeof(SignalHeader);
    }

    // serialize
    void serialize(char* out) {

    }

    // deserialize


    // add in data
    void put(const T& x, timestamp_t t) {
        _data.push_back(std::make_tuple(x, t));
    }
    void put(const T& x) { put(x, now()); }



};


#endif /* signal_h */
