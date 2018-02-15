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
#include "util.h"  // for DIV_ROUND_UP


template<bool Materialize, class IntT, class UintT>
int64_t call_appropriate_query_func(const IntT* src, UintT* dest,
    uint16_t ndims, uint32_t ngroups, uint16_t remaining_len, QueryParams qp)
{
    // ensure that the compiler doesn't optimize everything away
    #define DUMMY_READ_QUERY_RESULT(q)                              \
        do {                                                        \
            auto res = q.result();                                  \
            auto elemsz = sizeof(res[0]);                           \
            auto ptr = (uint8_t*)res.data();                        \
            volatile uint8_t max = 0;                               \
            for (int i = 0; i < res.size() * elemsz; i++) {         \
                if (ptr[i] > max) { max = ptr[i]; }                 \
            }                                                       \
        } while (0)

    MaxQuery<UintT> qMax(ndims);
    SumQuery<UintT> qSum(ndims);
    NoopQuery<UintT> qNoop(ndims);
    int64_t ret = -1;
    switch (qp.op) {
    case (REDUCE_MIN): break; // TODO
    case (REDUCE_MAX):
        ret = query_rowmajor_delta_rle<Materialize>(src, dest, ndims, ngroups,
            remaining_len, qMax);
        DUMMY_READ_QUERY_RESULT(qMax);
        break;
    case (REDUCE_SUM):
        ret = query_rowmajor_delta_rle<Materialize>(src, dest, ndims, ngroups,
            remaining_len, qSum);
        DUMMY_READ_QUERY_RESULT(qSum);
        break;
    default:
        ret = query_rowmajor_delta_rle<Materialize>(src, dest, ndims, ngroups,
            remaining_len, qNoop);
        break;
    }

    #undef DUMMY_READ_QUERY_RESULT

    return ret;
}

int64_t query_rowmajor_delta_rle_8b(const int8_t* src, uint8_t* dest,
                                    const QueryParams& qp)
{
    uint16_t ndims;
    uint32_t ngroups;
    uint16_t remaining_len;
    src += read_metadata_rle(src, &ndims, &ngroups, &remaining_len);
    if (qp.materialize) {
        return call_appropriate_query_func<true>(src, dest, ndims, ngroups,
            remaining_len, qp);
    } else {
        return call_appropriate_query_func<false>(src, dest, ndims, ngroups,
            remaining_len, qp);
    }
}
int64_t query_rowmajor_delta_rle_16b(const int16_t* src, uint16_t* dest,
    const QueryParams& qp)
{
    uint16_t ndims;
    uint32_t ngroups;
    uint16_t remaining_len;
    src += read_metadata_rle(src, &ndims, &ngroups, &remaining_len);
    // NoopQuery<uint16_t> q(ndims);

    if (qp.materialize) {
        return call_appropriate_query_func<true>(src, dest, ndims, ngroups,
            remaining_len, qp);
    } else {
        return call_appropriate_query_func<false>(src, dest, ndims, ngroups,
            remaining_len, qp);
    }
}
