//
//  sprintz_delta_rle.cpp
//  Compress
//
//  Created by DB on 12/11/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#include "sprintz_delta.h"
#include "sprintz_delta_rle_query.hpp"

#include "format.h"
#include "query.hpp"


int64_t query_rowmajor_delta_rle_8b(const int8_t* src, uint8_t* dest,
                                    const QueryParams& qparams)
{
    uint16_t ndims;
    uint32_t ngroups;
    uint16_t remaining_len;
    src += read_metadata_rle(src, &ndims, &ngroups, &remaining_len);
    NoopQuery<uint8_t> q(ndims);
    return query_rowmajor_delta_rle(
        src, dest, ndims, ngroups, remaining_len, q);
}
int64_t query_rowmajor_delta_rle_16b(const int16_t* src, uint16_t* dest,
    const QueryParams& qparams)
{
    uint16_t ndims;
    uint32_t ngroups;
    uint16_t remaining_len;
    src += read_metadata_rle(src, &ndims, &ngroups, &remaining_len);
    NoopQuery<uint16_t> q(ndims);
    return query_rowmajor_delta_rle(
        src, dest, ndims, ngroups, remaining_len, q);
}
