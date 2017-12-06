//
//  format.cpp
//  Compress
//
//  Created by DB on 12/5/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include "format.h"

static const uint16_t kMetaDataLenBytesRle = 8;
static const uint16_t kMetaDataLenBytesSimple = 6;

uint16_t write_metadata_rle(int8_t* orig_dest, uint16_t ndims,
    uint32_t ngroups, uint16_t remaining_len)
{
    *(uint32_t*)orig_dest = ngroups;
    *(uint16_t*)(orig_dest + 4) = (uint16_t)remaining_len;
    *(uint16_t*)(orig_dest + 6) = ndims;
    return kMetaDataLenBytesRle;
}

uint16_t read_metadata_rle(const int8_t* src, uint16_t* p_ndims,
    uint64_t* p_ngroups, uint16_t* p_remaining_len)
{
    static const uint32_t len_nbytes = 4;
    uint64_t one = 1; // make next line legible
    uint64_t len_mask = (one << (8 * len_nbytes)) - 1;
    *p_ngroups = (*(uint64_t*)src) & len_mask;
    *p_remaining_len = (*(uint16_t*)(src + len_nbytes));
    *p_ndims = (*(uint16_t*)(src + len_nbytes + 2));

    return kMetaDataLenBytesRle; // bytes taken up by metadata
}

uint16_t write_metadata_simple(void* dest, uint16_t ndims, uint32_t len) {
    uint8_t* _dest = (uint8_t*)dest;
    *(uint32_t*)_dest = len;
    *(uint16_t*)(_dest + 4) = ndims;
    return kMetaDataLenBytesSimple;
}

uint16_t read_metadata_simple(const void* src, uint16_t* p_ndims,
    uint32_t* p_len)
{
    uint8_t* _src = (uint8_t*)src;
    *p_len = *(uint32_t*)_src;
    *p_ndims = *(uint16_t*)(_src + 4);
    return kMetaDataLenBytesSimple;
}
