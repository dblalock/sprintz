//
//  delta.hpp
//  Compress
//
//  Created by DB on 11/1/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef delta_hpp
#define delta_hpp

#include <stdint.h>

uint32_t encode_delta_rowmajor(const uint8_t* src, uint32_t len, int8_t* dest,
                           uint16_t ndims, bool write_size=true);
uint32_t decode_delta_rowmajor(const int8_t* src, uint32_t len, uint8_t* dest,
                           uint16_t ndims);
uint32_t decode_delta_rowmajor_inplace(uint8_t* buff, uint32_t len,
                                       uint16_t ndims);
uint32_t decode_delta_rowmajor(const int8_t* src, uint8_t* dest);

#endif /* delta_hpp */
