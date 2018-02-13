//
//  query.hpp
//  Compress
//
//  Created by DB on 2/13/18.
//  Copyright © 2018 D Blalock. All rights reserved.
//

#ifndef query_h
#define query_h

// namespace query {

#include "immintrin.h" // XXX don't assume AVX2

enum Operation { REDUCE_MIN, REDUCE_MAX, REDUCE_SUM };

typedef struct QueryParams {
    Operation op;
} QueryParams;

// XXX don't assume AVX2
// TODO put this in a different file
template<typename DataT> struct data_traits {};
template<> struct data_traits<int8_t> { using vector_type = __m256i; };
template<> struct data_traits<uint8_t> { using vector_type = __m256i; };
template<> struct data_traits<int16_t> { using vector_type = __m256i; };
template<> struct data_traits<uint16_t> { using vector_type = __m256i; };

template<typename DataT>
class NoopQuery {
public:
    using vec_t = typename data_traits<DataT>::vector_type;
    explicit NoopQuery(int64_t ndims) {}
    void operator()(const vec_t& prev_vals, const vec_t& vals) { }
    int result() { return 0; }
};

// } // namespace query
#endif /* query_h */
