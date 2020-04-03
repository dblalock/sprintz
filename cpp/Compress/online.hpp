//
//  online.hpp
//  Compress
//
//  Created by DB on 3/31/20.
//  Copyright © 2020 D Blalock. All rights reserved.
//

#ifndef online_h
#define online_h

#include <stdint.h>

#include "bitpack.h" // for nbits cost arrays
#include "macros.h"  // just for restrict

// NOTE: a lot of this could be cleaner / more robust with more cpp features
// (e.g., checking that Predictor scalar type matches OnlineCoder scalar type),
// but we're writing it such that it will be easy to port to C

// ================================================================ types

// handle buffers longer than 2^32; signed so differences work correctly
// typedef int64_t idx_t;

typedef uint32_t len_t;

// =============================================================== helper funcs

// inline uint8_t _add_with_wraparound_i8(uint8_t a, int8_t b) {
//     // return a + b;  // technically undefined behavior, but usually works

//     // force 2s complement add
//     #ifdef __arm__
//         uint8_t ret;
//         asm("add %[c], %[a], %[b]" : [c] "=g" (ret) : [a] "g" (a), [b] "g" (b));
//         return ret;
//     #else  // assumes x86
//         asm("add %[a], %[b]" : [a] "+%g" (a) : [b] "g" (b));
//         return a;
//     #endif
// }

// inline int8_t _sub_with_wraparound_i8(uint8_t a, uint8_t b) {
//     // force 2s complement add
//     #ifdef __arm__
//         uint8_t ret;
//         asm("sub %[c], %[a], %[b]" : [c] "=g" (ret) : [a] "g" (a), [b] "g" (b));
//         return ret;
//     #else  // assumes x86
//         asm("sub %[a], %[b]" : [a] "+%g" (a) : [b] "g" (b));
//         return a;
//     #endif
// }

// inline uint16_t _add_with_wraparound_i16(uint16_t a, int16_t b) {
//     // return a + b;  // technically undefined behavior, but usually works

//     // force 2s complement add
//     #ifdef __arm__
//         uint16_t ret;
//         asm("add %[c], %[a], %[b]" : [c] "=g" (ret) : [a] "g" (a), [b] "g" (b));
//         return ret;
//     #else  // assumes x86
//         asm("add %[a], %[b]" : [a] "+%g" (a) : [b] "g" (b));
//         return a;
//     #endif
// }

// inline int16_t _sub_with_wraparound_i16(uint16_t a, int16_t b) {
//     // force 2s complement sub
//     #ifdef __arm__
//         int16_t ret;
//         asm("sub %[c], %[a], %[b]" : [c] "=g" (ret) : [a] "g" (a), [b] "g" (b));
//         return ret;
//     #else  // assumes x86
//         asm("sub %[a], %[b]" : [a] "+%g" (a) : [b] "g" (b));
//         return a;
//     #endif
// }

inline uint16_t _add_with_wraparound(uint16_t a, int16_t b) {
    // return a + b;  // technically undefined behavior, but usually works

    // force 2s complement add
    #ifdef __arm__
        uint16_t ret;
        // asm("add %[c], %[a], %[b]" : [c] "=g" (ret) : [a] "g" (a), [b] "g" (b));
        asm("add %[a], %[b], %[c]" : [c] "=g" (ret) : [a] "g" (a), [b] "g" (b));
        return ret;
    #else  // assumes x86
        // asm("add %[a], %[b]" : [a] "+g" (a) : [b] "g" (b));
        // asm("add %[b], %[a]" : [a] "+%g" (a) : [b] "g" (b));
        asm("add %[a], %[b]" : [b] "+g" (b) : [a] "g" (a));
        // return a;
        return b;
    #endif
}

inline int16_t _sub_with_wraparound(uint16_t a, int16_t b) {
    // force 2s complement sub
    #ifdef __arm__
        int16_t ret;
        // asm("sub %[c], %[a], %[b]" : [c] "=g" (ret) : [a] "g" (a), [b] "g" (b));
        asm("sub %[a], %[b], %[c]" : [c] "=g" (ret) : [a] "g" (a), [b] "g" (b));
        return ret;
    #else  // assumes x86
        // asm("sub %[a], %[b]" : [a] "+%g" (a) : [b] "g" (b));
        // return a;
        asm("sub %[b], %[a]" : [a] "+%g" (a) : [b] "g" (b));
        return a;
    #endif
}


// =========================================== predictive coding state machines

struct DeltaPredictor_u16 {
    using data_t = uint16_t;
    using err_t = int16_t;

    void init(data_t initial_val) {
        _prev = initial_val;
    }

    data_t predict() const {
        return _prev;
    }

    void train(err_t err, data_t true_val) {
        _prev = true_val;
    }

private:
    data_t _prev;
};

struct DoubleDeltaPredictor_u16 {
    using data_t = uint16_t;
    using err_t = int16_t;

    // TODO implement method to let it skip to end of block for dynamic
    // predictor encoding
    // void full_init(data_t prev0, data_t prev1, data_t prev2, data_t prev3) {
    //     _prev0 = prev0;
    //     _prev1 = prev1;
    // }

    void init(data_t initial_val) {
        // full_init(initial_val, initial_val, 0, 0);
        _prev_val = initial_val;
        _prev_diff = 0;
        // _prev1 = initial_val; // make it equivalent to delta coding at first
    }

    data_t predict() const {
        // like `_prev + (prev0 - prev1)`, but avoids undefined behavior
        // err_t diff = _sub_with_wraparound(_prev0, _prev1);
        // return _add_with_wraparound(_prev0, diff);
        return _add_with_wraparound(_prev_val, _prev_diff);
    }

    void train(err_t err, data_t true_val) {
        _prev_diff = _sub_with_wraparound(true_val, _prev_val);
        _prev_val = true_val;
        // _prev1 = _prev0;
        // _prev0 = true_val;
    }

private:
    // data_t _prev0;
    // data_t _prev1;
    data_t _prev_val;
    err_t _prev_diff;
};


struct TripleDeltaPredictor_u16 {
    using data_t = uint16_t;
    using err_t = int16_t;

    void init(data_t initial_val) {
        _prev_val = initial_val;
        // make it equivalent to delta coding at first
        _prev_diff = 0;
        _prev_ddiff = 0;

        // _prev0 = initial_val;
        // // make it equivalent to delta coding at first
        // _prev1 = initial_val;
        // _prev2 = initial_val;
    }

    data_t predict() const {
        err_t predicted_diff = _add_with_wraparound(_prev_diff, _prev_ddiff);
        // printf("trip prev val:   %d\n", _prev_val);
        // printf("trip prev diff:  %d\n", _prev_diff);
        // printf("trip prev ddiff: %d\n", _prev_ddiff);
        // printf("trip predicted diff: %d\n", predicted_diff);
        return _add_with_wraparound(_prev_val, predicted_diff);

        // err_t diff0 = _sub_with_wraparound(_prev0, _prev1);
        // printf("trip diff0: %d\n", diff0);
        // err_t diff1 = _sub_with_wraparound(_prev1, _prev0);
        // printf("trip diff1: %d\n", diff1);
        // err_t ddiff = _sub_with_wraparound(diff0, diff1);
        // printf("trip ddiff: %d\n", ddiff);
        // data_t linear_prediciton = _add_with_wraparound(_prev0, diff0);
        // printf("trip linpred: %d\n", linear_prediciton);
        // return _add_with_wraparound(linear_prediciton, ddiff);
    }

    void train(err_t err, data_t true_val) {
        err_t diff = _sub_with_wraparound(true_val, _prev_val);
        _prev_ddiff = _sub_with_wraparound(diff, _prev_diff);
        _prev_diff = diff;
        _prev_val = true_val;

        // _prev2 = _prev1;
        // _prev1 = _prev0;
        // _prev0 = true_val;
    }

private:
    // data_t _prev0;
    // data_t _prev1;
    // data_t _prev2;
    data_t _prev_val;
    err_t _prev_diff;
    err_t _prev_ddiff;
};

struct MovingAvgPredictor_u16 {
    using data_t = uint16_t;
    using err_t = int16_t;

    static const uint8_t _shift = 2;

    void init(data_t initial_val) {
        _accumulator = initial_val << _shift;
    }

    data_t predict() const {
        return _accumulator >> _shift;
    }

    void train(err_t err, data_t true_val) {
        // this implements a moving average in a nice way that uses extremely
        // few instructions and also keeps the low bits; derivation:
        //
        // a(t+1) = .25 * x(t) + .75 * a(t)
        // 4a(t+1) = x(t) + 3a(t)
        // 4a(t+1) = a(t) + err(t) + 3a(t)
        // a(t+1) = a(t) + .25 * err(t)
        //
        // and of course, could use different shift amounts to get different
        // convex combinations
        _accumulator += err;
    }

private:
    int32_t _accumulator;
};

template<class predictor_type>
struct PredictiveCoder {
    using data_t = typename predictor_type::data_t;
    using err_t = typename predictor_type::err_t;

    void init(data_t initial_val) {
        _predictor.init(initial_val);
    }

    err_t encode_next(data_t val) {
        data_t prediction = _predictor.predict();
        // printf("enc prediction: %d\n", prediction);
        // err_t err = val - prediction;
        err_t err = _sub_with_wraparound(val, prediction);
        // printf("enc err: %d\n", err);
        // printf("---- enc val: %d\n", val);
        _predictor.train(err, val);
        return err;
    }

    data_t decode_next(err_t err) {
        data_t prediction = _predictor.predict();
        // printf("dec prediction: %d\n", prediction);
        data_t val = (data_t)_add_with_wraparound(prediction, err);
        // data_t val = prediction + err;
        // printf("dec err: %d\n", err);
        // printf("dec val: %d\n", val);
        // data_t val = prediction + err;
        _predictor.train(err, val);
        return val;
    }

private:
    predictor_type _predictor;
};

template<class predictor_type, class int_t0, class int_t1>
len_t _predictive_code(const int_t0* data_in, int_t1* data_out,
                      len_t length, bool encode)
{
    if (length <= 0) { return length; }
    // always copy the first value
    *data_out = *data_in;
    if (length == 1) { return length; }

    // create predictive coder and either call encode or decode on each scalar
    PredictiveCoder<predictor_type> coder;
    coder.init(data_in[0]);
    if (encode) {
        for (uint32_t i = 1; i < length; i++) {
            data_out[i] = coder.encode_next(data_in[i]);
            // printf("i = %d, enc wrote val: %d\n", i, data_out[i]);
        }
    } else {
        for (uint32_t i = 1; i < length; i++) {
            // data_out[i] = coder.decode_next(data_in[i]);
            auto tmp = coder.decode_next(data_in[i]);
            data_out[i] = tmp;
            // printf("decode next returned: %d\n", tmp);
            // printf("i = %d, dec wrote val: %d\n", i, data_out[i]);
        }
    }
    return length;
}

template<class predictor_type, class uint_t, class int_t>
len_t predictive_encode(const uint_t* input, len_t length, int_t* output) {
    return _predictive_code<predictor_type>(
        input, output, length, true /* encode */);
}

template<class predictor_type, class uint_t, class int_t>
len_t predictive_decode(const uint_t* input, len_t length, int_t* output) {
    return _predictive_code<predictor_type>(
        input, output, length, false /* decode */);
}

// =================================================== dynamic predictor choice

typedef int32_t loss_t;
// static const int kMinPossibleLoss = -(1 << (8 * sizeof(loss_t) - 1));

struct Losses {  // just wrapping the enum so it has some reasonable scoping
    enum { MaxAbs, SumLogAbs };
};

template<int LossFunc, int BlockSz=8>
loss_t _compute_loss_u16(const uint16_t* buff) {
    typedef uint16_t int_t; // typedef so easy to adapt code for other types
    static_assert(LossFunc == Losses::MaxAbs || LossFunc == Losses::SumLogAbs,
                  "Invalid loss function!");
    static_assert(BlockSz >= 1, "Blocks must contain at least 1 element!");

    // uint8_t sign_bit_pos = 8 * sizeof(int_t) - 1;

    if (LossFunc == Losses::MaxAbs) {
        // int_t val = *buff;
        // val ^= val >> sign_bit_pos;  // flip bits if negative
        // loss_t loss = ZIGZAG_ENCODE_SCALAR(*buff);
        loss_t loss = *buff;
        for (int i = 1; i < BlockSz; i++) {
            loss = MAX(loss, buff[i]);
        }
        return loss;
    } else if (LossFunc == Losses::SumLogAbs) {
        loss_t loss = 0;
        for (int i = 0; i < BlockSz; i++) {
            uint8_t nleading_zeros = __builtin_clz(buff[i]);
            uint8_t logabs = sizeof(int_t) * 8 - nleading_zeros;
            loss += logabs;
        }
        return loss;
    }
    return -1;
}

template<int LossFunc=Losses::SumLogAbs, int BlockSz=8>
void dynamic_delta_zigzag_encode_u16(
    const uint16_t* data_in, uint16_t* data_out, uint8_t* choices_out,
    len_t length)
{
    typedef uint16_t uint_t;

    if (length == 0) { return; }
    // always copy the first value
    *data_out = *data_in;
    if (length == 1) { return; }

    length -= 1;  // effective length is 1 less, since 1st elem already copied
    len_t nblocks = length / BlockSz;
    len_t full_blocks_len = nblocks * BlockSz;
    len_t tail_len = length - full_blocks_len;

    // allocate buffers for codecs we consider
    uint_t tmp0[BlockSz];
    uint_t tmp1[BlockSz];

    // create encoders
    PredictiveCoder<DeltaPredictor_u16> enc0;
    PredictiveCoder<DoubleDeltaPredictor_u16> enc1;
    enc0.init(data_in[0]);
    enc1.init(data_in[0]);

    // zero out choices buffer; it's a bitfield so length/8 bytes, rounded up
    for (int i = 0; i < (length + 7) / 8; i++) {
        choices_out[i] = 0;
    }

    const uint_t* in_ptr = data_in + 1;
    uint_t* out_ptr = data_out + 1;
    for (len_t b = 0; b < nblocks; b++) {
        for (int bb = 0; bb < BlockSz; bb++) {
            tmp0[bb] = ZIGZAG_ENCODE_SCALAR(enc0.encode_next(*in_ptr));
            tmp1[bb] = ZIGZAG_ENCODE_SCALAR(enc1.encode_next(*in_ptr));
            in_ptr++;
        }
        loss_t loss0 = _compute_loss_u16<LossFunc, BlockSz>(tmp0);
        loss_t loss1 = _compute_loss_u16<LossFunc, BlockSz>(tmp1);

        uint_t* copy_from;
        uint8_t choice;
        if (loss0 <= loss1) {
            copy_from = tmp0;
            choice = 0;
        } else {
            copy_from = tmp1;
            choice = 1;
        }
        // write out prediction errs and choice of predictor
        for (int bb = 0; bb < BlockSz; bb++) {
            *out_ptr++ = copy_from[bb];
        }
        len_t choices_byte_offset = b / 8;
        uint8_t choices_bit_offset = (uint8_t)(b % 8);
        choices_out[choices_byte_offset] |= choice << choices_bit_offset;
    }
    // just delta code the tail
    for (int i = 0; i < tail_len; i++) {
        *out_ptr++ = enc0.encode_next(*in_ptr++);
    }
}

len_t dynamic_delta_choices_size(len_t length, int blocksz=8);

void dynamic_delta_zigzag_encode_u16(
    const uint16_t* data_in, uint16_t* data_out, uint8_t* choices_out,
    len_t length, int loss);

void dynamic_delta_zigzag_decode_u16(
    const uint16_t* data_in, uint16_t* data_out, uint8_t* choices_in,
    len_t length);

void dynamic_delta_zigzag_decode_u16(
    const uint16_t* data_in, uint16_t* data_out, uint8_t* choices_in);

#endif /* online_h */
