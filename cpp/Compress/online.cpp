//
//  online.cpp
//  Compress
//
//  Created by DB on 3/31/20.
//  Copyright © 2020 D Blalock. All rights reserved.
//

#include "online.hpp"

#include "array_utils.hpp"
#include "debug_utils.hpp"

// =================================================== dynamic predictor choice

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
len_t dynamic_delta_zigzag_encode_u16(
    const uint16_t* data_in, int16_t* data_out, uint8_t* choices_out,
    len_t length)
{
    typedef int16_t int_t;
    typedef uint16_t uint_t;

    if (length <= 0) { return length; }
    // always copy the first value
    *data_out = *data_in;
    if (length == 1) { return length; }

    // ar::print(data_in, "input")

    length -= 1;  // effective length is 1 less, since 1st elem already copied
    len_t nblocks = length / BlockSz;
    len_t full_blocks_len = nblocks * BlockSz;
    len_t tail_len = length - full_blocks_len;

    // allocate buffers for codecs we consider
    uint_t tmp0[BlockSz];
    uint_t tmp1[BlockSz];

    // create encoders
    PredictiveCoder<DeltaPredictor_u16> enc0;
    // PredictiveCoder<DeltaPredictor_u16> enc1;
    PredictiveCoder<DoubleDeltaPredictor_u16> enc1;
    enc0.init(data_in[0]);
    enc1.init(data_in[0]);

    // zero out choices buffer; it's a bitfield so length/8 bytes, rounded up
    for (int i = 0; i < (length + 7) / 8; i++) {
        choices_out[i] = 0;
    }

    const uint_t* in_ptr = data_in + 1;
    int_t* out_ptr = data_out + 1;
    for (len_t b = 0; b < nblocks; b++) {
        for (int bb = 0; bb < BlockSz; bb++) {
            uint16_t val = *in_ptr++;
            tmp0[bb] = zigzag_encode_16b(enc0.encode_next(val));
            tmp1[bb] = zigzag_encode_16b(enc1.encode_next(val));
            // if (val == 1) {
            //     printf("bb = %d, val = %d\n", bb, val);
            // }
            // if (bb == 7) { printf("enc next prediction: %d\n", enc0._predictor.predict()); }
            // auto err = enc0.encode_next(val);
            // printf("bb = %d, val = %d, err = %d, zz(err) = %d\n", bb, val, err, zigzag_encode_16b(err));
            // tmp0[bb] = zigzag_encode_16b(err);
            // TODO uncomment above
            // tmp0[bb] = enc0.encode_next(*in_ptr);
            // auto val1 = enc1.encode_next(*in_ptr);
            // tmp1[bb] = ZIGZAG_ENCODE_SCALAR(val1);
            // printf("elem, zigzagged elem = %d, %d\n", val1, ZIGZAG_ENCODE_SCALAR(val1));
        }
        // ar::print(&tmp0[0], BlockSz, "tmp0");
        // ar::print(&tmp1[0], BlockSz, "tmp1");
        loss_t loss0 = _compute_loss_u16<LossFunc, BlockSz>(tmp0);
        loss_t loss1 = _compute_loss_u16<LossFunc, BlockSz>(tmp1);

        uint_t* copy_from;
        uint8_t choice;
        if (loss0 <= loss1) {
        // if (true) { // TODO rm  // works
        // if (false) { // TODO rm
            copy_from = tmp0;
            choice = 0;
        } else {
            copy_from = tmp1;
            choice = 1;
        }
        // ar::print(&tmp0[0], BlockSz, "tmp0");
        // ar::print(&tmp1[0], BlockSz, "tmp1");
        // printf("enc choosing codec %d\n", choice);
        // write out prediction errs and choice of predictor
        for (int bb = 0; bb < BlockSz; bb++) {
            // printf("enc writing val: %d (%d)\n", copy_from[bb], ZIGZAG_DECODE_SCALAR(copy_from[bb]));
            // printf("tmp0[bb], tmp1[bb] = %d, %d\n", tmp0[bb], tmp1[bb]);
            *out_ptr++ = copy_from[bb];
        }
        len_t choices_byte_offset = b / 8;
        uint8_t choices_bit_offset = (uint8_t)(b % 8);
        choices_out[choices_byte_offset] |= choice << choices_bit_offset;
    }
    // just delta code the tail
    // printf("enc enc0 next prediction: %d\n", enc0._predictor.predict());
    // printf("enc *in_ptr, *(in_ptr+1): %d, %d\n", *in_ptr, *(in_ptr + 1));
    for (int i = 0; i < tail_len; i++) {
        // auto val = enc0.encode_next(*in_ptr++);
        // printf("enc writing tail val: %d\n", val);
        // *out_ptr++ = val;
        *out_ptr++ = enc0.encode_next(*in_ptr++);
    }

    // ar::print(data_out, length + 1, "compressed\t\t\t");
    // ar::print(ar::map(zigzag_decode_16b, data_out, length+1).get(), length+1, "compressed nozigzag\t");

    return length + 1; // original length
}

len_t dynamic_delta_zigzag_decode_u16(
    const int16_t* data_in, len_t length, uint16_t* data_out,
    uint8_t* choices_in)
{
    typedef uint16_t uint_t;
    typedef int16_t int_t;
    static const int BlockSz = 8; // TODO allow other block sizes?
    static_assert(BlockSz >= 3,
        "Predictor jump() can read past beginning with small blocks");

    if (length <= 0) { return length; }
    // always copy the first value
    *data_out = *data_in;
    if (length == 1) { return length; }

    length -= 1;  // effective length is 1 less, since 1st elem already copied
    len_t nblocks = length / BlockSz;
    len_t full_blocks_len = nblocks * BlockSz;
    len_t tail_len = length - full_blocks_len;

    // create encoders
    PredictiveCoder<DeltaPredictor_u16> enc0;
    PredictiveCoder<DoubleDeltaPredictor_u16> enc1;
    enc0.init(data_in[0]);
    enc1.init(data_in[0]);

    const int_t* in_ptr = data_in + 1;
    uint_t* out_ptr = data_out + 1;
    for (int b = 0; b < nblocks; b++) {
        // read the next predictor choice
        len_t choices_byte_offset = b / 8;
        uint8_t choices_bit_offset = (uint8_t)(b % 8);
        uint8_t choice = (
            choices_in[choices_byte_offset] >> choices_bit_offset) & 0x01;
        // printf("dec choosing codec %d\n", choice);
        // use chosen codec to decode this block
        switch(choice) {
            case (0): // delta code
                // printf("dec using delta codec\n");
                for (int bb = 0; bb < BlockSz; bb++) {
                    int_t val = zigzag_decode_16b(*in_ptr++);


                    // SELF: issue when switching coders is that predictive
                    // coder we don't use never sees the right vals; it instead
                    // sees the vals that it would have reconstructed given
                    // the errs; two solutions:
                    //  1) add train() to the predictive coder to let us pass
                    //  in the correct values
                    //  2) add jump() method to let us skip feeding in all the
                    //  values
                    //
                    //  -might be nice to have both; moving avg can't jump
                    // but training on everything unnecessarily is a waste



                    // printf("dec bb = %d, val = %d\n", bb, val);
                    // if (bb == 7) { printf("dec next prediction: %d\n", enc0._predictor.predict()); }
                    *out_ptr++ = enc0.decode_next(val);
                    // enc1.decode_next(val);  // TODO allow skip to block end
                }
                enc1.jump(out_ptr[-1], out_ptr[-2], out_ptr[-3]);
                break;
            case (1): // double delta code
                // printf("dec using double delta codec\n");
                for (int bb = 0; bb < BlockSz; bb++) {
                    int_t val = zigzag_decode_16b(*in_ptr++);
                    // enc0.decode_next(val);
                    *out_ptr++ = enc1.decode_next(val);
                }
                enc0.jump(out_ptr[-1], out_ptr[-2], out_ptr[-3]);
                break;
            default:
                printf("ERROR: got codec choice %d!\n", choice);
                break; // can't happen
        }
    }
    // printf("dec enc0 next prediction: %d\n", enc0._predictor.predict());
    // printf("dec *in_ptr, *(in_ptr+1): %d, %d\n", *in_ptr, *(in_ptr + 1));
    // just delta code the tail
    for (int i = 0; i < tail_len; i++) {
        // auto val = enc0.decode_next(*in_ptr++);
        // printf("dec writing tail val: %d\n", val);
        // *out_ptr++ = val;
        *out_ptr++ = enc0.decode_next(*in_ptr++);
    }
    return length + 1; // original length
}

len_t dynamic_delta_choices_size(len_t length, int blocksz) {
    return (length + blocksz - 1) / blocksz;
}

len_t dynamic_delta_zigzag_encode_u16(
    const uint16_t* data_in, len_t length, int16_t* data_out,
    uint8_t* choices_out, int loss)
{
    switch(loss) {
        case (Losses::MaxAbs):
            return dynamic_delta_zigzag_encode_u16<Losses::MaxAbs>(
                data_in, data_out, choices_out, length);
        case (Losses::SumLogAbs):
            return dynamic_delta_zigzag_encode_u16<Losses::SumLogAbs>(
                data_in, data_out, choices_out, length);
        default: return -1;
    }
}

