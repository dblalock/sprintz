//
//  format.hpp
//  Compress
//
//  Created by DB on 2017-12-5.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef format_hpp
#define format_hpp

#include <stdint.h>

uint16_t write_metadata_rle(int8_t* dest, uint16_t ndims, uint32_t ngroups,
    uint16_t remaining_len);

uint16_t read_metadata_rle(const int8_t* src, uint16_t* p_ndims,
    uint64_t* p_ngroups, uint16_t* p_remaining_len);


#endif /* format_hpp */
