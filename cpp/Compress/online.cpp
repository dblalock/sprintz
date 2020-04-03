//
//  online.cpp
//  Compress
//
//  Created by DB on 3/31/20.
//  Copyright © 2020 D Blalock. All rights reserved.
//

#include "online.hpp"

len_t dynamic_delta_choices_size(len_t length, int blocksz) {
    return (length + blocksz - 1) / blocksz;
}

void dynamic_delta_zigzag_encode_u16(
    const uint16_t* data_in, uint16_t* data_out, uint8_t* choices_out,
    len_t length, int loss)
{
    switch(loss) {
        case (Losses::MaxAbs):
            dynamic_delta_zigzag_encode_u16<Losses::MaxAbs>(
                data_in, data_out, choices_out, length);
            break;
        case (Losses::SumLogAbs):
            dynamic_delta_zigzag_encode_u16<Losses::SumLogAbs>(
                data_in, data_out, choices_out, length);
            break;
        default: break;
    }
}

void dynamic_delta_zigzag_decode_u16(
    const uint16_t* data_in, uint16_t* data_out, uint8_t* choices_in,
    len_t length)
{
    typedef uint16_t uint_t;
    typedef int16_t int_t;
    static const int BlockSz = 8; // TODO allow other block sizes?

    if (length == 0) { return; }
    // always copy the first value
    *data_out = *data_in;
    if (length == 1) { return; }

    length -= 1;  // effective length is 1 less, since 1st elem already copied
    len_t nblocks = length / BlockSz;
    len_t full_blocks_len = nblocks * BlockSz;
    len_t tail_len = length - full_blocks_len;

    // create encoders
    PredictiveCoder<DeltaPredictor_u16> enc0;
    PredictiveCoder<DoubleDeltaPredictor_u16> enc1;
    enc0.init(data_in[0]);
    enc1.init(data_in[0]);

    const uint_t* in_ptr = data_in + 1;
    uint_t* out_ptr = data_out + 1;
    for (int b = 0; b < nblocks; b++) {
        // read the next predictor choice
        len_t choices_byte_offset = b / 8;
        uint8_t choices_bit_offset = (uint8_t)(b % 8);
        uint8_t choice = choices_in[choices_byte_offset] >> choices_bit_offset;
        // use chosen codec to decode this block
        switch(choice) {
            case (0): // delta code
                for (int bb = 0; bb < BlockSz; bb++) {
                    *out_ptr++ = ZIGZAG_DECODE_SCALAR(
                        enc0.decode_next(*in_ptr));
                    enc1.decode_next(*in_ptr);  // TODO allow skip to block end
                    in_ptr++;
                }
                break;
            case (1): // double delta code
                for (int bb = 0; bb < BlockSz; bb++) {
                    enc0.decode_next(*in_ptr);
                    *out_ptr++ = ZIGZAG_DECODE_SCALAR(
                        enc1.decode_next(*in_ptr));
                    in_ptr++;
                }
                break;
        }
    }
    // just delta code the tail
    for (int i = 0; i < tail_len; i++) {
        *out_ptr++ = enc0.decode_next(*in_ptr++);
    }
}


