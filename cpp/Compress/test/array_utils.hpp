//
//  array_utils.hpp
//
//  Created By Davis Blalock on 1/14/14.
//  Copyright (c) 2014 Davis Blalock. All rights reserved.
//

#ifndef __ARRAY_UTILS_HPP
#define __ARRAY_UTILS_HPP

#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <algorithm>
#include <memory>
#include <sstream>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "macros.hpp"
#include "debug_utils.hpp"

using std::begin;
using std::end;
using std::log2;
using std::unique_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;

namespace ar {

typedef int64_t length_t;

static const double kDefaultNonzeroThresh = .00001;

// TODO support for negative strides / asserts where they can't be negative

// ================================================================
// Scalar funcs
// ================================================================

// ------------------------ less picky min/max/abs funcs than stl

template<class data_t>
static inline data_t abs(data_t x) {
	return x >= 0 ? x : -x;
}

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline auto max(data_t1 x, data_t2 y) -> decltype(x + y) {
	return x >= y ? x : y;
}

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline auto min(data_t1 x, data_t2 y) -> decltype(x + y) {
	return x <= y ? x : y;
}

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline auto dist_L1(data_t1 x, data_t2 y) -> decltype(x - y) {
	return x >= y ? x - y : y - x;
}

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline auto dist_sq(data_t1 x, data_t2 y) -> decltype(x - y) {
	return (x - y) * (x - y);
}

// ------------------------ logical ops

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline bool logical_and(data_t1 x, data_t2 y) {
	return static_cast<bool>(x) && static_cast<bool>(y);
}

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline bool logical_nand(data_t1 x, data_t2 y) {
	return !logical_and(x, y);
}

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline bool logical_or(data_t1 x, data_t2 y) {
	return static_cast<bool>(x) || static_cast<bool>(y);
}

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline bool logical_nor(data_t1 x, data_t2 y) {
	return !logical_or(x, y);
}

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline bool logical_xor(data_t1 x, data_t2 y) {
	return static_cast<bool>(x) != static_cast<bool>(y);
}

template<class data_t1, class data_t2,
	REQUIRE_NOT_PTR(data_t1), REQUIRE_NOT_PTR(data_t2)>
static inline bool logical_xnor(data_t1 x, data_t2 y) {
	return static_cast<bool>(x) == static_cast<bool>(y);
}

template<class data_t>
static inline bool logical_not(data_t x) {
	return !static_cast<bool>(x);
}

// ------------------------ bitwise ops

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline bool bitwise_and(data_t1 x, data_t2 y) {
	return static_cast<data_t1>(x & y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline bool bitwise_nand(data_t1 x, data_t2 y) {
	return ~bitwise_and(x, y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline bool bitwise_or(data_t1 x, data_t2 y) {
	return static_cast<data_t1>(x | y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline bool bitwise_nor(data_t1 x, data_t2 y) {
	return ~bitwise_or(x, y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline data_t1 bitwise_xor(data_t1 x, data_t2 y) {
	return static_cast<data_t1>(x ^ y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline data_t1 bitwise_xnor(data_t1 x, data_t2 y) {
	return ~bitwise_xor(x, y);
}

template<class data_t, REQUIRE_PRIMITIVE(data_t)>
static inline data_t bitwise_not(data_t x) {
	return ~x;
}

// ================================================================
// Functional Programming
// ================================================================

// ================================ Map
// throughout these funcs, we use our own for loop instead of
// std::tranform with a std::back_inserter so that we can use
// emplace_back(), instead of push_back()

// ------------------------------- 1 container version

template <class F, class data_t, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline void map(const F&& func, const data_t *RESTRICT data,
	len_t len, data_t2 *RESTRICT out, length_t inStride=1, length_t outStride=1)
{
	for (len_t i = 0; i < len; i++) {
		out[i * outStride] = static_cast<data_t2>(func(data[i * inStride]));
	}
}
template <class F, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void map_inplace(const F&& func, data_t* data, len_t len,
	length_t stride=1)
{
	for (len_t i = 0; i < len; i++) {
		data[i * stride] = static_cast<data_t>(func(data[i * stride]));
	}
}
template <class F, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline auto map(const F&& func, const data_t* data, len_t len,
	length_t stride=1)
	-> unique_ptr<decltype(func(data[0]))[]>
{
	unique_ptr<decltype(func(data[0]))[]> ret(new decltype(func(data[0]))[len]);
	for (len_t i = 0; i < len; i++) {
		ret[i] = func(data[i * stride]);
	}
	return ret;
}
/** Returns a new container holding the results of applying the function
 * to the given container */
template<class F, template <class...> class Container, class... Args>
static inline auto map(const F&& func, const Container<Args...>& container,
	length_t stride=1)
	-> Container<decltype(func(*begin(container)))>
{
	Container<decltype(func(*begin(container)))> ret;
	for (auto i = begin(container); i < end(container); i+= stride) {
		ret.emplace_back(func(*i));
	}
	return ret;
}

// ------------------------------- 2 container version

template <class F, class data_t1, class data_t2, class data_t3, class len_t,
	 REQUIRE_INT(len_t)>
static inline void map(const F&& func, const data_t1* x, const data_t2* y,
	len_t len, data_t3 *RESTRICT out, length_t xStride=1, length_t yStride=1,
	length_t outStride=1)
{
	for (len_t i = 0; i < len; i++) {
		out[i * outStride] = static_cast<data_t3>(
			func(x[i * xStride], y[i * yStride]));
	}
}
template <class F, class data_t1, class data_t2, class len_t,
	REQUIRE_INT(len_t)>
static inline auto map(const F&& func, const data_t1* x, const data_t2* y,
	len_t len,  length_t xStride=1, length_t yStride=1)
	-> unique_ptr<decltype(func(x[0], y[0]))[]>
{
	unique_ptr<decltype(func(x[0], y[0]))[]> ret(
		new decltype(func(x[0], y[0]))[len]);
	for (len_t i = 0; i < len; i++) {
		ret[i] = func(x[i * xStride], y[i * yStride]);
	}
	return ret;
}
template <class F, class data_t1, class data_t2, class len_t,
	REQUIRE_INT(len_t)>
static inline void map_inplace(const F&& func, data_t1* x, const data_t2* y,
					   len_t len, length_t xStride=1, length_t yStride=1)
{
	for (len_t i = 0; i < len; i++) {
		x[i * xStride] = static_cast<data_t1>(func(x[i * xStride],
												   y[i * yStride]));
	}
}

template<class F, template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
static inline auto map(const F&& func, const Container1<Args1...>& x,
	const Container2<Args2...>& y, length_t xStride=1, length_t yStride=1)
	-> Container1<decltype(func(*begin(x), *begin(y)))>
{
	assert(x.size() / xStride == y.size() / yStride);
	Container1<decltype(func(*begin(x), *begin(y)))> ret;
	auto ity = begin(y);
	for (auto itx = begin(x); itx < end(x); itx += xStride, ity += yStride) {
		ret.emplace_back(func(*itx, *ity));
	}
	return ret;
}

// ------------------------------- mapi, 1 container version

template <class F, class data_t, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline void mapi(const F&& func, const data_t *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, length_t inStride=1, length_t outStride=1)
{
	for (len_t i = 0; i < len; i++) {
		out[i * outStride] = static_cast<data_t2>(func(i, data[i * inStride]));
	}
}
template <class F, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void mapi_inplace(const F&& func, data_t* data, len_t len,
	length_t stride=1)
{
	for (len_t i = 0; i < len; i++) {
		data[i * stride] = (data_t) func(i, data[i * stride]);
	}
}
template <class F, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline auto mapi(const F&& func, const data_t* data, len_t len,
	length_t stride=1)
	-> unique_ptr<decltype(func(len, data[0]))[]>
{
	unique_ptr<decltype(func(len, data[0]))[]>
			ret(new decltype(func(len, data[0]))[len]);
	for (len_t i = 0; i < len; i++) {
		ret[i] = func(i, data[i * stride]);
	}
	return ret;
}
/** Returns a new container holding the results of applying the function
 * to the given container; the index within the array, as well as the
 * array element itself, are passed to func. */
template<class F, template <class...> class Container, class... Args>
static inline auto mapi(const F&& func, const Container<Args...>& container,
	length_t stride=1)
	-> Container<decltype(func(0, *begin(container)))>
{
	Container<decltype(func(container.size(), *begin(container)))> ret;
	length_t i = 0;
	for (auto it = begin(container); it < end(container); it += stride, i++) {
		ret.emplace_back(func(i, *it));
	}
	// size_t i = 0;
	// for (const auto& el : container) {
	// 	ret.emplace_back(func(i, el));
	// 	i++;
	// }
	return ret;
}

// ------------------------------- mapi, 2 container version
template <class F, class data_t1, class data_t2, class data_t3, class len_t,
	REQUIRE_INT(len_t)>
static inline void mapi(const F&& func, const data_t1* x, const data_t2* y,
	len_t len, data_t3 *RESTRICT out, length_t xStride=1, length_t yStride=1,
	length_t outStride=1)
{
	for (len_t i = 0; i < len; i++) {
		out[i * outStride] = static_cast<data_t3>(
			func(i, x[i * xStride], y[i * yStride]));
	}
}
template <class F, class data_t1, class data_t2, class len_t,
	REQUIRE_INT(len_t)>
static inline auto mapi(const F&& func, const data_t1* x,
	const data_t2* y, len_t len, length_t xStride=1, length_t yStride=1)
	-> unique_ptr<decltype(func(len, x[0], y[0]))[]>
{
	unique_ptr<decltype(func(len, x[0], y[0]))[]>
		ret(new decltype(func(len, x[0], y[0]))[len]);
	for (len_t i = 0; i < len; i++) {
		ret[i] = func(i, x[i * xStride], y[i * yStride]);
	}
	return ret;
}
template<class F, template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
static inline auto mapi(const F&& func, const Container1<Args1...>& x,
	const Container2<Args2...>& y, length_t xStride=1, length_t yStride=1)
	-> Container1<decltype(func(0, *begin(x), *begin(y)))>
{
	assert(x.size() / xStride == y.size() / yStride);
	Container1<decltype(func(x.size(), *begin(x), *begin(y)))> ret;
	auto ity = begin(y);
	length_t i = 0;
	for (auto itx = begin(x); itx < end(x); itx += xStride, ity += yStride, i++) {
		ret.emplace_back(func(i, *itx, *ity));
	}
	return ret;
}


// ================================ Filter

template<class F, template <class...> class Container, class... Args>
static inline Container<Args...> filter(const F&& func,
	const Container<Args...>& container)
{
	Container<Args...> ret;
	for (const auto& el : container) {
		if (func(el)) {
			ret.emplace_back(el);
		}
	}
	return ret;
}
/** Like filter(), but also passes the index within the container to func() as
 * a first argument */
template<class F, template <class...> class Container, class... Args>
static inline Container<Args...> filteri(const F&& func,
	const Container<Args...>& container)
{
	Container<Args...> ret;
	size_t i = 0;
	for (const auto& el : container) {
		if (func(i, el)) {
			ret.emplace_back(el);
		}
		i++;
	}
	return ret;
}

// ================================ Reduce

template<class F, class data_t, class len_t, REQUIRE_INT(len_t)>
static auto reduce(const F&& func, const data_t* data, len_t len)
	-> decltype(func(data[0], data[0]))
{
	if (len < 1) {
		return static_cast<decltype(func(data[0], data[0]))>(NULL);
	}
	if (len == 1) {
		// ideally we would just return the first element,
		// but it might not be the right type
		printf("WARNING: reduce(): called on array with 1 element; ");
		printf("reducing the first element with itself.\n");
		return func(data[0], data[0]);
	}

	auto total = func(data[0], data[1]);
	for (len_t i = 2; i < len; i++) {
		total = func(total, data[i]);
	}
	return total;
}
template<class F, template <class...> class Container, class... Args>
static auto reduce(const F&& func, const Container<Args...>& container)
	-> decltype(func(*begin(container), *end(container)))
{
	auto it = begin(container);
	if (it >= end(container)) {			// 0 elements
		return NULL;
	}
	if (it == end(container) - 1) {		// 1 element
		// ideally we would just return the first element,
		// but it might not be the right type
		printf("WARNING: reduce(): called on container with 1 element; ");
		printf("reducing the first element with itself.\n");
		return func(*it, *it);
	}

	// call func on idxs {0,1}, then on total({0,1}, 2)
	auto total = func(*it, *(it+1));
	for (it += 2; it < end(container); it++) {
		total = func(total, *it);
	}
	return total;
}

// ================================ Where

// ------------------------ raw arrays

template<class F, class data_t>
static inline vector<length_t> where(const F&& func, const data_t* data,
	length_t len)
{
	vector<length_t> ret;
	for (length_t i = 0; i < len; i++) {
		if (func(data[i])) {
			ret.emplace_back(i);
		}
	}
	return ret;
}
template<class F, class data_t>
static inline vector<length_t> wherei(const F&& func, const data_t* data,
	length_t len)
{
	vector<length_t> ret;
	for (length_t i = 0; i < len; i++) {
		if (func(i, data[i])) {
			ret.emplace_back(i);
		}
	}
	return ret;
}

// ------------------------ containers


template<class F, template <class...> class Container, class... Args>
static inline Container<length_t> where(const F&& func,
	const Container<Args...>& container)
{
	Container<length_t> ret;
	length_t i = 0;
	for (const auto& el : container) {
		if (func(el)) {
			ret.emplace_back(i);
		}
		i++;
	}
	return ret;
}

/** Like where(), but also passes the index within the container to func() as
 * a first argument */
template<class F, template <class...> class Container, class... Args>
static inline Container<length_t> wherei(const F&& func,
	const Container<Args...>& container)
{
	Container<length_t> ret;
	length_t i = 0;
	for (const auto& el : container) {
		if (func(i, el)) {
			ret.emplace_back(i);
		}
		i++;
	}
	return ret;
}

// ================================ Where for particular properties

#define WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(FUNC, NAME) \
\
template<class data_t> \
static inline vector<length_t> NAME(const data_t* data, length_t len) { \
	return where([](data_t x) { return FUNC(x); }, data, len); \
} \
template<template <class...> class Container, class data_t> \
static inline Container<length_t> NAME(const Container<data_t>& container) { \
	return where([](data_t x) { return FUNC(x); }, container); \
} \

WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(static_cast<bool>, where);
WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(!static_cast<bool>, where_false);
WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(isnan, where_nan);
WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(isfinite, where_finite);
WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(!isfinite, where_inf);

template<class data_t> static inline bool isnegative(data_t x) { return x < 0; }
//template<class data_t> static inline bool isnonnegative(data_t x) { return x >= 0; }
template<class data_t> static inline bool ispositive(data_t x) { return x > 0; }
//template<class data_t> static inline bool isnonpositive(data_t x) { return x <= 0; }

WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(isnegative, where_negative);
WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(!isnegative, where_nonnegative);
WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(ispositive, where_positive);
WRAP_WHERE_UNARY_BOOLEAN_FUNC_WITH_NAME(!ispositive, where_nonpositive);


// ================================ Nonzeros (special case of above)

template<class data_t, class float_t=double>
static inline vector<length_t> nonzeros(const data_t* data, length_t len,
	float_t thresh=kDefaultNonzeroThresh)
{
	return where([thresh](data_t x) { return abs(x) > thresh; }, data, len);
}
template<template <class...> class Container, class data_t>
static inline Container<bool> nonzeros(const Container<data_t>& container,
	float_t thresh=kDefaultNonzeroThresh)
{
	return where([thresh](data_t x) { return abs(x) > thresh; }, container);
}

// ================================ Find (All)

template<template <class...> class Container, class... Args,
class data_t>
static inline int32_t find(const Container<Args...>& container,
	data_t val) {
	int32_t i = 0;
	for (auto it = std::begin(container); it != std::end(container); ++it) {
		if ((*it) == val) {
			return i;
		}
		i++;
	}
	return -1;
}

template<template <class...> class Container, class... Args,
class data_t>
static inline length_t rfind(const Container<Args...>& container,
	data_t val) {
	length_t i = container.size() - 1;
	for (auto it = std::end(container)-1; it >= std::begin(container); --it) {
		if ((*it) == val) {
			return i;
		}
		i--;
	}
	return -1;
}

template<template <class...> class Container, class... Args,
class data_t>
static inline Container<length_t> findall(const Container<Args...>& container,
	data_t val) {
	return where([&val](data_t a) {return a == val;} );
}

// ================================ Contains

template<template <class...> class Container, class... Args,
class data_t>
static inline size_t contains(const Container<Args...>& container,
	data_t val) {
	auto idx = find(container, val);
	return idx >= 0;
}

// ================================ at_idxs

/** note that this requires that the container implement operator[] */
template<class data_t,
	template <class...> class Container2, class... Args2>
static inline vector<data_t> at_idxs(const data_t data[],
		const Container2<Args2...>& indices) {
	vector<data_t> ret;
	for (auto idx : indices) {
		auto val = data[idx];
		ret.push_back(val);
	}
	return ret;
}

/** note that this requires that the container implement operator[] */
template<template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
static inline Container1<Args1...> at_idxs(const Container1<Args1...>& container,
		const Container2<Args2...>& indices,
		bool boundsCheck=false) {
	Container1<Args1...> ret;

	if (boundsCheck) {
		auto len = container.size();
		for (auto idx : indices) {
			if (idx >= 0 && idx < len) {
				auto val = container[static_cast<size_t>(idx)];
				ret.push_back(val);
			}
		}
	} else {
		for (auto idx : indices) {
			auto val = container[static_cast<size_t>(idx)];
			ret.push_back(val);
		}
	}
	return ret;
}

// ================================================================
// Sequence creation
// ================================================================

// ================================ range

// ------------------------ start and stop

template <class data_t1, class data_t2, class step_t=int8_t>
static inline int32_t num_elements_in_range(data_t1 startVal,
											data_t2 stopVal, step_t step)
{
	if (startVal == stopVal) { return 0; }
	// assertf( (stopVal - startVal) / step > 0,
	// 		"ERROR: range: invalid args min=%.3f, max=%.3f, step=%.3f\n",
	// 		(double)startVal, (double)stopVal, (double)step);
	return ceil((stopVal - startVal) / step);
}

template <class data_t0, class data_t1, class data_t2, class step_t=int8_t>
static inline void range_inplace(data_t0* data, data_t1 startVal,
								 data_t2 stopVal, step_t step=1)
{
	int32_t len = num_elements_in_range(startVal, stopVal, step);
	if (len < 1) { return; }

	data[0] = startVal;
	for (int32_t i = 1; i < len; i++) {
		data[i] = data[i-1] + step;
	}
}

/** Create an array containing a sequence of values; equivalent to Python
 * range(startVal, stopVal, step), or MATLAB startVal:step:stopVal */
template <class data_t1, class data_t2, class step_t=int8_t>
static inline auto range_ar(data_t1 startVal, data_t2 stopVal, step_t step=1)
	-> unique_ptr<decltype(stopVal - startVal + step)[]>
{
	typedef decltype(stopVal - startVal + step) out_t;
	int32_t len = num_elements_in_range(startVal, stopVal, step);
	// len = max(len, 1);
	if (len < 1) { return nullptr; }

	unique_ptr<out_t[]> data(new out_t[len]);
	range_inplace(data, startVal, stopVal, step);
	return data;
}
/** Create an array containing a sequence of values; equivalent to Python
 * range(startVal, stopVal, step), or MATLAB startVal:step:stopVal */
template <class data_t1, class data_t2, class step_t=int8_t>
static inline auto range(data_t1 startVal, data_t2 stopVal, step_t step=1)
	-> vector<decltype(stopVal - startVal + step)>
{
	using Scalar = decltype(stopVal - startVal + step);
	int32_t len = num_elements_in_range(startVal, stopVal, step);
	if (len < 1) { return vector<Scalar>{}; }

	vector<Scalar> data(len);
	range_inplace(data.data(), startVal, stopVal, step);
	return data;
}

// ------------------------ stop only
// note that this is different from python range in that range(-k) is
// equivalent range(0, -k, -1), rather than returning an empty collection

template <class data_t0, class data_t1>
static inline void range_inplace(data_t0* data, data_t1 stopVal) {
	if (stopVal == 0) { return; }
	int8_t step = stopVal > 0 ? 1 : -1;
	return range_inplace(data, 0, stopVal, step);
}

/** Create an array containing a sequence of values; equivalent to Python
 * range(startVal, stopVal, step), or MATLAB startVal:step:stopVal */
template <class data_t>
static inline unique_ptr<data_t> range_ar(data_t stopVal) {
	if (stopVal == 0) { return nullptr; }
	int8_t step = stopVal > 0 ? 1 : -1;
	return range_ar(0, stopVal, step);
}
/** Create an array containing a sequence of values; equivalent to Python
 * range(stopVal) */
template <class data_t>
static inline vector<data_t> range(data_t stopVal) {
	if (stopVal == 0) { return vector<data_t>{}; }
	int8_t step = stopVal > 0 ? 1 : -1;
	return range(0, stopVal, step);
}


// ================================ exprange

template <class data_t1, class data_t2, class step_t=int8_t>
static inline int32_t num_elements_in_exprange(data_t1 startVal,
											   data_t2 stopVal, step_t step)
{
	assertf(startVal != 0, "exprange(): start value == 0!");
	assertf(stopVal != 0, "exprange(): end value == 0!");
	assertf(step != 0, "exprange(): step == 0!");

	auto absStartVal = abs(startVal);
	auto absStopVal = abs(stopVal);
	auto absStep = abs(step);

	if (absStartVal > absStopVal) {
		assertf(absStep < 1,
				"exprange(): |startVal| %.3g > |stopVal| %.3g, but |step| %.3g >= 1",
				(double)absStartVal, (double)absStopVal, (double)absStep);
	} else if (absStartVal < absStopVal) {
		assertf(absStep > 1,
				"exprange(): |startVal| %.3g > |stopVal| %.3g, but |step| %.3g <= 1",
				(double)absStartVal, (double)absStopVal, (double)absStep);
	} else {
		return 1; // startVal == stopVal
	}

	double ratio = static_cast<double>(absStopVal) / absStartVal;
	double logBaseStep = log2(ratio) / log2(absStep);
	return 1 + floor(logBaseStep);
}
template <class data_t0, class data_t1, class data_t2, class step_t=int8_t>
static inline void exprange_inplace(data_t0* data, data_t1 startVal,
								 data_t2 stopVal, step_t step=2)
{
	int32_t len = num_elements_in_exprange(startVal, stopVal, step);
	data[0] = startVal;
	for (int32_t i = 1; i < len; i++) {
		data[i] = data[i-1] * step;
	}
}

// step is int so it can be powers of 2 by default
template <class data_t, class step_t=int8_t>
static inline auto exprange_ar(data_t startVal, data_t stopVal, step_t step=2)
	-> unique_ptr<decltype(stopVal * startVal * step)[]>
{
	int32_t len = num_elements_in_exprange(startVal, stopVal, step);
	unique_ptr<decltype(stopVal * startVal * step)[]> data(
		new decltype(stopVal * startVal * step)[len]);
	exprange_inplace(data, startVal, stopVal, step);
	return data;
}

template <class data_t1, class data_t2, class step_t=int8_t>
static inline auto exprange(data_t1 startVal, data_t2 stopVal, step_t step=2)
-> vector<decltype(stopVal * startVal * step)>
{
	int32_t len = num_elements_in_exprange(startVal, stopVal, step);
	vector<decltype(stopVal * startVal * step)> data(len);
	exprange_inplace(data.data(), startVal, stopVal, step);
	return data;
}

// ================================ Create constant array

/** Sets each element of the array to the value specified */
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline void constant_inplace(data_t1 *x, len_t len, data_t2 value) {
	for (len_t i = 0; i < len; i++) {
		x[i] = static_cast<data_t1>(value);
	}
}
template<template <class...> class Container,
	class data_t1, class data_t2>
static inline void constant_inplace(Container<data_t1>& data, data_t2 value) {
	constant_inplace(data.data(), value, data.size());
}

/** Returns an array of length len with all elements equal to value */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> constant_ar(len_t len, data_t value) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	constant_inplace(ret, value, len);
	return ret;
}
/** Returns an array of length len with all elements equal to value */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline vector<data_t> constant(len_t len, data_t value) {
	vector<data_t> ret(len, value);
	return ret;
}

// ================================================================
// Reshaping
// ================================================================

// reads in a 1D array and returns an array of ND arrays
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static data_t** split(const data_t* data, len_t len, len_t newNumDims) {
	size_t newArraysLen = len / newNumDims;
	assertf(newArraysLen * newNumDims == len,
		"reshape(): newNumDims %d is not factor of array length %d",
		newNumDims, len);

	size_t sample,dimension,readFrom=0;
	//initialize each array ptr and the array containing them; note
	//that the arrays are allocated as one contiguous block of memory
	data_t** arrays = new data_t*[newNumDims];
	data_t* arrayContents = new data_t[len];
	for (dimension = 0; dimension < newNumDims; dimension++) {
		arrays[dimension] = arrayContents + dimension*newArraysLen;
	}

	//copy the values from the 1D array to be reshaped
	for (sample = 0; sample < newArraysLen; sample++) {
		for (dimension = 0; dimension < newNumDims; dimension++, readFrom++) {
			arrays[dimension][sample] = data[readFrom];
		}
	}

	return arrays;
}

// ================================================================
// Statistics (V -> R)
// ================================================================

// ================================ Max

/** Returns the maximum value in data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline data_t max(const data_t *data, len_t len) {
	data_t max = std::numeric_limits<data_t>::min();
	for (len_t i = 0; i < len; i++) {
		if (data[i] > max) {
			max = data[i];
		}
	}
	return max;
}
/** Returns the maximum value in data[0..len-1] */
template<template <class...> class Container, class data_t>
static inline data_t max(const Container<data_t>& data) {
	return max(data.data(), data.size());
}

// ================================ Min

/** Returns the minimum value in data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline data_t min(const data_t *data, len_t len) {
	data_t min = std::numeric_limits<data_t>::max();
	for (len_t i = 0; i < len; i++) {
		if (data[i] < min) {
			min = data[i];
		}
	}
	return min;
}
/** Finds the minimum of the elements in data */
template<template <class...> class Container, class data_t>
static inline data_t min(const Container<data_t>& data) {
	return min(data.data(), data.size());
}

// ================================ Argmax

template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline len_t argmax(const data_t *data, len_t len) {
	data_t max = std::numeric_limits<data_t>::min();
	len_t bestIdx = 0;
	for (len_t i = 0; i < len; i++) {
		if (data[i] > max) {
			max = data[i];
			bestIdx = i;
		}
	}
	return bestIdx;
}
template<template <class...> class Container, class data_t>
static inline size_t argmax(const Container<data_t>& data) {
	return argmax(data.data(), data.size());
}

// ================================ Argmin

template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline len_t argmin(const data_t *data, len_t len) {
	data_t min = std::numeric_limits<data_t>::max();
	len_t bestIdx = 0;
	for (len_t i = 0; i < len; i++) {
		if (data[i] < min) {
			min = data[i];
			bestIdx = i;
		}
	}
	return bestIdx;
}
template<template <class...> class Container, class data_t>
static inline size_t argmin(const Container<data_t>& data) {
	return argmin(data.data(), data.size());
}

// ================================ Sum

/** Computes the sum of data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline data_t sum(const data_t *data, len_t len) {
	return reduce([](data_t x, data_t y){ return x+y;}, data, len);
}
/** Computes the sum of the elements in data */
// template <class data_t>
// data_t sum(const vector<data_t>& data) {
template<template <class...> class Container, class data_t>
static inline data_t sum(const Container<data_t>& data) {
	return reduce([](data_t x, data_t y){ return x+y;}, data);
}

// ================================ Sum of Squares

/** Computes the sum of data[i]^2 for i = [0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline data_t sumsquares(const data_t *data, len_t len) {
	data_t sum = 0;
	for (len_t i=0; i < len; i++) {
		sum += data[i]*data[i];
	}
	return sum;
}
/** Computes the sum of data[i]^2 for i = [0..len-1] */
template<template <class...> class Container, class data_t>
static inline data_t sumsquares(const Container<data_t>& data) {
	return sumsquares(data.data(), data.size());
}

// ================================ Mean

/** Computes the arithmetic mean of data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline double mean(const data_t* data, len_t len) {
	return sum(data, len) / ((double) len);
}
/** Computes the arithmetic mean of data[0..len-1] */
// template <class data_t>
// data_t mean(const vector<data_t>& data) {
template<template <class...> class Container, class data_t>
static inline double mean(const Container<data_t>& data) {
	return sum(data) / ((double) data.size());
}

// ================================ Variance

// Knuth's numerically stable algorithm for online mean + variance. See:
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
template<class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void mean_and_variance(const data_t* data, len_t len,
	double& mean, double& variance)
{
	mean = data[0];
	double sse = 0;
	double delta;
	for (len_t i=1; i < len; i++) {
		delta = data[i] - mean;
		mean += delta / (i+1);
		sse += delta * (data[i] - mean);
	}
	variance = sse / len;
}

// template<class data_t, class len_t, REQUIRE_INT(len_t)>
// static inline void mean_and_variance(const data_t* data, len_t len,
// 	double& mean, double& variance) {
// 	knuth_sse_stats(data, len, mean, variance);
// 	variance /= len;
// }

/** Computes the population variance of data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline double variance(const data_t *data, len_t len) {
	assert(len > 0);
	if (len == 1) {
		return 0;
	}
	double mean, variance;
	mean_and_variance(data, len, mean, variance);
	return variance;
}
template<template <class...> class Container, class data_t>
static inline double variance(const Container<data_t>& data) {
	return variance(data.data(), data.size());
}

// ================================ Standard deviation

/** Computes the population standard deviation of data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline double stdev(const data_t *data, len_t len) {
	return sqrt(variance(data,len));
}

/** Computes the population standard deviation of data[0..len-1] */
template<template <class...> class Container, class data_t>
static inline double stdev(const Container<data_t>& data) {
	return sqrt(variance(data));
}

// ================================ LP norm

/** Computes the arithmetic mean of data[0..len-1] */
template <class data_t, class len_t, class pow_t=int>
static inline double norm(const data_t* data, len_t len, pow_t p=2) {
	data_t sum = 0;
	for(len_t i = 0; i < len; i++) {
		sum += std::pow(data[i], p);
	}
	return std::pow(sum, 1.0 / p);
}
/** Computes the arithmetic mean of data[0..len-1] */
template <int P, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline double norm(const data_t* data, len_t len) {
	return norm(data, len, P);
}

template<int P, template <class...> class Container, class data_t>
static inline double norm(const Container<data_t>& data) {
	return norm<P>(data.data(), data.size());
}
template<template <class...> class Container, class data_t, class pow_t=int>
static inline double norm(const Container<data_t>& data, pow_t p=2) {
	return norm(data.data(), data.size(), p);
}

// ================================ L1 norm

template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline data_t norm_L1(const data_t* data, len_t len) {
	return sum(data, len);
}
template<template <class...> class Container, class data_t>
static inline data_t norm_L1(const Container<data_t>& data) {
	return sum(data);
}

// ================================ L2 norm

template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline data_t norm_L2(const data_t* data, len_t len) {
	return sqrt(sumsquares(data, len));
}
template<template <class...> class Container, class data_t>
static inline data_t norm_L2(const Container<data_t>& data) {
	return sqrt(sumsquares(data));
}

// ================================================================
// V x V -> R
// ================================================================

// ================================ Dot Product
/** Returns the the dot product of x and y */
template<class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline auto dot(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] * y[0])
{
	decltype(x[0] * y[0]) sum = 0; // get type of sum correct
	for (len_t i = 0; i < len; i++) {
		sum += x[i] * y[i];
	}
	return sum;
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline double dot(const Container1<data_t1>& x, const Container2<data_t2>& y) {
	assert(x.size() == y.size());
	return dot(x.data(), y.data(), x.size());
}

// ================================ L1 Distance

template<class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline auto dist_L1(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] - y[0])
{
	decltype(x[0] - y[0]) sum = 0; // get type of sum correct
	for (len_t i = 0; i < len; i++) {
		sum += dist_L1(x[i], y[i]);
	}
	return sum;
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline double dist_L1(const Container1<data_t1>& x,
	const Container2<data_t2>& y)
{
	assert(x.size() == y.size());
	return dist_L1(x.data(), y.data(), x.size());
}

// ================================ L2^2 Distance

template<class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline auto dist_sq(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] - y[0])
{
	decltype(x[0] - y[0]) sum = 0; // get type of sum correct
	for (len_t i = 0; i < len; i++) {
		auto diff = x[i] - y[i];
		sum += diff * diff;
	}
	return sum;
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline double dist_sq(const Container1<data_t1>& x,
	const Container2<data_t2>& y)
{
	assert(x.size() == y.size());
	return dist_sq(x.data(), y.data(), x.size());
}

// ================================ L2 Distance

template<class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline auto dist_L2(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] - y[0])
{
	return sqrt(dist_sq(x, y, len));
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline double dist_L2(const Container1<data_t1>& x,
	const Container2<data_t2>& y)
{
	assert(x.size() == y.size());
	return sqrt(dist_sq(x.data(), y.data(), x.size()));
}

// ================================================================
// Cumulative Statistics (V[1:i] -> R[i])
// ================================================================

// ================================ Cumulative Sum

/** Cumulative sum of elements in src, storing the result in dest */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void cumsum(const data_t* src, data_t* dest, len_t len) {
	dest[0] = src[0];
	for (len_t i=1; i < len; i++) {
		dest[i] = src[i] + dest[i-1];
	}
}
/** Returns a new array composed of the cumulative sum of the data */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> cumsum(data_t *data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	cumsum(data, ret, len);
	return ret;
}
/** Returns a new array composed of the cumulative sum of the data */
template<template <class...> class Container, class data_t>
static inline Container<data_t> cumsum(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	cumsum(data.data(),ret.data(),data.size());
	return ret;
}

// ================================ Cumulative Mean

/** Cumulative mean of elements in src, storing the result in dest */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void cummean(const data_t* src, data_t* dest, len_t len) {
	double sum = 0;
	for (len_t i=0; i < len; i++) {
		sum += src[i];
		dest[i] = sum / (i+1);
	}
}
/** Returns a new array composed of the cumulative mean of the data */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> cummean(data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_cummean(data, ret, len);
	return ret;
}
/** Returns a new array composed of the cumulative mean of the data */
template<template <class...> class Container, class data_t>
static inline Container<data_t> cummean(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	array_cummean(data.data(), ret.data(), data.size());
	return ret;
}

// ================================ Cumulative SSE

/** Cumulative SSE of elements in src, storing the result in dest */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void cumsxx(const data_t* src, len_t len, data_t* dest) {
	assert(len > 0);
	dest[0] = 0;
	if (len == 1) {
		return;
	}

	//use Knuth's numerically stable algorithm
	double mean = src[0];
	double sse = 0;
	double delta;
	for (len_t i=1; i < len; i++) {
		delta = src[i] - mean;
		mean += delta / (i+1);
		sse += delta * (src[i] - mean);
		dest[i] = sse;
	}
}
/** Returns the sum of squared differences from the mean of data[0..i] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> cumsxx(data_t *data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_cumsxx(data, len, ret);
	return ret;
}
/** Returns the sum of squared differences from the mean of data[0..i] */
template<template <class...> class Container, class data_t>
static inline Container<data_t> cumsxx(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	array_cumsxx(data.data(), data.size(), ret.data());
	return ret;
}

// ================================================================
// V x V -> V
// ================================================================

	// for (len_t i = 0; i < len; i++) { \
	// 	x[i] = static_cast<data_t1>(x[i] OP y[i]); \
	// } \

// TODO cleaner impl is to wrap arithmetic ops in binary funcs and then
// just use binary func wrapping code

#define WRAP_VECTOR_VECTOR_OP_WITH_NAME(OP, NAME) \
template <class data_t1, class data_t2, class len_t, class data_t3, \
	REQUIRE_INT(len_t)> \
static inline void NAME(const data_t1* x, const data_t2* y, len_t len, \
	data_t3* out) \
{ \
	return map([](data_t1 a, data_t2 b){ return a OP b;}, x, y, len, out); \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t)> \
static inline void NAME ## _inplace(data_t1 *RESTRICT x, \
	const data_t2 *RESTRICT y, len_t len) \
{ \
	map_inplace([](data_t1 a, data_t2 b) { return a OP b; }, x, y, len); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t1* x, const data_t2* y, len_t len) \
	-> unique_ptr<decltype(x[0] OP y[0])[]> \
{ \
	return map([](data_t1 a, data_t2 b){ return a OP b;}, x, y, len); \
} \
template<template <class...> class Container1, class data_t1, \
template <class...> class Container2, class data_t2> \
static inline auto NAME(const Container1<data_t1>& x, \
	const Container2<data_t2>& y) -> Container1<decltype(x[0] OP y[0])> \
{ \
	return map([](data_t1 a, data_t2 b){ return a OP b;}, x, y); \
} \

// TODO do +=, etc, also via map_inplace
WRAP_VECTOR_VECTOR_OP_WITH_NAME(+, add)
WRAP_VECTOR_VECTOR_OP_WITH_NAME(-, sub)
WRAP_VECTOR_VECTOR_OP_WITH_NAME(*, mul)
// WRAP_VECTOR_VECTOR_OP_WITH_NAME(/ (double), div) // cast denominator to double
WRAP_VECTOR_VECTOR_OP_WITH_NAME(/, div)


#define WRAP_VECTOR_VECTOR_FUNC_WITH_NAME(FUNC, NAME) \
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t)> \
static inline void NAME(const data_t1* x, const data_t2* y, data_t3* out, \
	len_t len) \
{ \
	for (len_t i = 0; i < len; i++) { \
		out[i] = static_cast<data_t3>(FUNC(x[i], y[i])); \
	} \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t)> \
static inline void NAME ## _inplace(data_t1 *RESTRICT x, \
	const data_t2 *RESTRICT y, len_t len) \
{ \
	for (len_t i = 0; i < len; i++) { \
		x[i] = static_cast<data_t1>(FUNC(x[i], y[i])); \
	} \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t1* x, const data_t2* y, \
	len_t len) -> unique_ptr<decltype(FUNC(x[0], y[0]))[]> \
{ \
	return map([](data_t1 a, data_t2 b){ return FUNC(a, b);}, x, y, len); \
} \
template<template <class...> class Container1, class data_t1, \
	template <class...> class Container2, class data_t2> \
static inline auto NAME(const Container1<data_t1>& x, \
	const Container2<data_t2>& y) -> Container1<decltype(FUNC(x[0], y[0]))> \
{ \
	return map([](data_t1 a, data_t2 b){ return FUNC(a, b);}, x, y); \
} \

#define WRAP_VECTOR_VECTOR_FUNC(FUNC) \
	WRAP_VECTOR_VECTOR_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_VECTOR_VECTOR_STD_FUNC_WITH_NAME(FUNC, NAME) \
	using std::FUNC; \
	WRAP_VECTOR_VECTOR_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_VECTOR_VECTOR_STD_FUNC(FUNC) \
	WRAP_VECTOR_VECTOR_STD_FUNC_WITH_NAME(FUNC, FUNC)

// WRAP_VECTOR_VECTOR_FUNC(max);
// WRAP_VECTOR_VECTOR_FUNC(min);
// WRAP_VECTOR_VECTOR_FUNC(logical_and);
// WRAP_VECTOR_VECTOR_FUNC(logical_nand);
// WRAP_VECTOR_VECTOR_FUNC(logical_or);
// WRAP_VECTOR_VECTOR_FUNC(logical_nor);
// WRAP_VECTOR_VECTOR_FUNC(logical_xor);
// WRAP_VECTOR_VECTOR_FUNC(logical_xnor);
// WRAP_VECTOR_VECTOR_FUNC(logical_not);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_and);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_nand);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_or);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_nor);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_xor);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_xnor);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_not);
// WRAP_VECTOR_VECTOR_STD_FUNC(pow);

// ================================ Concatenate

template <class data_t, class len_t1, class len_t2>
static inline unique_ptr<data_t[]> concat(const data_t* x, const data_t* y,
	len_t1 len1, len_t2 len2)
{
	auto combinedLen = len1 + len2;
	unique_ptr<data_t[]> ret(new data_t[combinedLen]);
	size_t i = 0;
	for( ; i < len1; i++) {
		ret[i] = x[i];
	}
	for( ; i < combinedLen; i++) {
		ret[i] = y[i];
	}
	return ret;
}

template<template <class...> class Container1,
	template <class...> class Container2, class data_t>
static inline void concat_inplace(Container1<data_t>& x,
	const Container2<data_t>& y)
{
	std::copy(begin(y), end(y), std::back_inserter(x));
}

template<template <class...> class Container1,
	template <class...> class Container2, class data_t>
static inline Container1<data_t> concat(const Container1<data_t>& x,
	const Container2<data_t>& y)
{
	Container1<data_t> ret(x);
	concat_inplace(x, y);
	return ret;
}

// ================================================================
// V x R -> V
// ================================================================

// ================================ wrap V x R (and R x V) operators

#define WRAP_VECTOR_SCALAR_OP_WITH_NAME(OP, NAME) \
\
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t)> \
static inline void NAME(data_t1 *RESTRICT data, len_t len, data_t2 val, \
	data_t3 *RESTRICT out) \
{ \
	map([val](data_t1 x){return x OP val;}, data, len, out); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline void NAME ## _inplace(data_t1* data, len_t len, data_t2 val) { \
	map_inplace([val](data_t1 x){return x OP val;}, data, len); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t1* data, len_t len, data_t2 val) \
	-> unique_ptr<decltype(data[0] OP val)[]> \
{ \
	return map([val](data_t1 x) {return x OP val;}, data, len); \
} \
template<template <class...> class Container, class data_t1, class data_t2> \
static inline auto NAME(const Container<data_t1>& data, data_t2 val) \
	-> Container<decltype(*begin(data) OP val)> \
{ \
	return map([val](data_t1 x) {return x OP val;}, data); \
} \
template<template <class...> class Container, class data_t1, \
	class data_t2, REQUIRE_NOT_PTR(data_t2)> \
static inline void NAME ## _inplace(Container<data_t1>& data, data_t2 val) \
{ \
	map_inplace([val](data_t1 x) {return x OP val;}, \
		data.data(), data.size()); \
} \
\
\
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NOT_PTR(data_t2)> \
static inline void NAME(data_t2 val, data_t1 *RESTRICT data, len_t len, \
	data_t3 *RESTRICT out) \
{ \
	map([val](data_t1 x){return val OP x;}, data, len, out); \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NOT_PTR(data_t2)> \
static inline void NAME ## _inplace(data_t2 val, data_t1* data, len_t len) { \
	map_inplace([val](data_t1 x){return val OP x;}, data, len); \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NOT_PTR(data_t2)> \
static inline auto NAME(data_t2 val, const data_t1* data, len_t len) \
	-> unique_ptr<decltype(val OP data[0])[]> \
{ \
	return map([val](data_t1 x) {return val OP x;}, data, len); \
} \
template<template <class...> class Container, class data_t1, \
	class data_t2, REQUIRE_NOT_PTR(data_t2)> \
static inline auto NAME(data_t2 val, const Container<data_t1>& data) \
	-> Container<decltype(val OP *begin(data))> \
{ \
	return map([val](data_t1 x) {return val OP x;}, data); \
} \
template<template <class...> class Container, class data_t1, \
	class data_t2, REQUIRE_NOT_PTR(data_t2)> \
static inline void NAME ## _inplace(data_t2 val, Container<data_t1>& data) \
{ \
	map_inplace([val](data_t1 x) {return val OP x;}, \
		data.data(), data.size()); \
} \

// TODO do +=, etc, also via map_inplace
// note that we need a separate macro for operators because there's
// no associated function for them in C++
WRAP_VECTOR_SCALAR_OP_WITH_NAME(+, add)
WRAP_VECTOR_SCALAR_OP_WITH_NAME(-, sub)
WRAP_VECTOR_SCALAR_OP_WITH_NAME(*, mul)
WRAP_VECTOR_SCALAR_OP_WITH_NAME(/, div)

// ================================ wrap V x R (and R x V) funcs

#define WRAP_VECTOR_SCALAR_FUNC_WITH_NAME(FUNC, NAME) \
\
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t)> \
static inline void NAME(data_t1 *RESTRICT data, len_t len, data_t2 val, \
	data_t3 *RESTRICT out) \
{ \
	map([val](data_t1 x){return FUNC(x, val);}, data, len, out); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline void NAME ## _inplace(data_t1* data, len_t len, data_t2 val) { \
	map_inplace([val](data_t1 x){return FUNC(x, val);}, data, len); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t1* data, len_t len, data_t2 val) \
	-> unique_ptr<decltype(FUNC(data[0], val))[]> \
{ \
	return map([val](data_t1 x) {return FUNC(x, val);}, data, len); \
} \
template<template <class...> class Container, class data_t1, class data_t2> \
static inline auto NAME(const Container<data_t1>& data, data_t2 val) \
	-> Container<decltype(FUNC(*begin(data), val))> \
{ \
	return map([val](data_t1 x) {return FUNC(x, val);}, data); \
} \
\
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NOT_PTR(data_t2)> \
static inline void NAME(data_t2 val, data_t1 *RESTRICT data, len_t len, \
	data_t3 *RESTRICT out) \
{ \
	map([val](data_t1 x){return FUNC(val, x);}, data, len, out); \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NOT_PTR(data_t2)> \
static inline void NAME ## _inplace(data_t2 val, data_t1* data, len_t len) { \
	map_inplace([val](data_t1 x){return FUNC(val, x);}, data, len); \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NOT_PTR(data_t2)> \
static inline auto NAME(data_t2 val, const data_t1* data, len_t len) \
	-> unique_ptr<decltype(FUNC(val, data[0]))[]> \
{ \
	return map([val](data_t1 x) {return FUNC(val, x);}, data, len); \
} \
template<template <class...> class Container, class data_t1, \
	class data_t2, REQUIRE_NOT_PTR(data_t2)> \
static inline auto NAME(data_t2 val, const Container<data_t1>& data) \
	-> Container<decltype(FUNC(val, *begin(data)))> \
{ \
	return map([val](data_t1 x) {return FUNC(val, x);}, data); \
} \

#define WRAP_VECTOR_SCALAR_FUNC(FUNC) \
	WRAP_VECTOR_SCALAR_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_VECTOR_SCALAR_STD_FUNC_WITH_NAME(FUNC, NAME) \
	using std::FUNC; \
	WRAP_VECTOR_SCALAR_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_VECTOR_SCALAR_STD_FUNC(FUNC) \
	WRAP_VECTOR_SCALAR_STD_FUNC_WITH_NAME(FUNC, FUNC)

// ================================ wrap binary op

#define WRAP_BINARY_FUNC_WITH_NAME(FUNC, NAME) \
	WRAP_VECTOR_VECTOR_FUNC_WITH_NAME(FUNC, NAME) \
	WRAP_VECTOR_SCALAR_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_BINARY_FUNC(FUNC) \
	WRAP_BINARY_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_BINARY_FUNC_WITH_NAME_IN_NAMESPACE(FUNC, NAME, NAMESPACE) \
	using NAMESPACE::FUNC; \
	WRAP_BINARY_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_BINARY_STD_FUNC_WITH_NAME(FUNC, NAME) \
	WRAP_BINARY_FUNC_WITH_NAME_IN_NAMESPACE(FUNC, NAME, std)

#define WRAP_BINARY_STD_FUNC(FUNC) \
	WRAP_BINARY_STD_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_BINARY_FUNC(FUNC) \
	WRAP_BINARY_FUNC_WITH_NAME(FUNC, FUNC)

WRAP_BINARY_FUNC(max);
WRAP_BINARY_FUNC(min);
WRAP_BINARY_FUNC(logical_and);
WRAP_BINARY_FUNC(logical_nand);
WRAP_BINARY_FUNC(logical_or);
WRAP_BINARY_FUNC(logical_nor);
WRAP_BINARY_FUNC(logical_xor);
WRAP_BINARY_FUNC(logical_xnor);
WRAP_BINARY_FUNC(bitwise_and);
WRAP_BINARY_FUNC(bitwise_nand);
WRAP_BINARY_FUNC(bitwise_or);
WRAP_BINARY_FUNC(bitwise_nor);
WRAP_BINARY_FUNC(bitwise_xor);
WRAP_BINARY_FUNC(bitwise_xnor);
WRAP_BINARY_STD_FUNC(pow);

// ================================ Pow

// // ------------------------ elements to scalar power

// template <class data_t1, class data_t2, class data_t3, class len_t,
// 	REQUIRE_INT(len_t)>
// static inline void pow(data_t1 *RESTRICT data, len_t len, data_t2 val,
// 	data_t3 *RESTRICT out)
// {
// 	map([val](data_t1 x){return std::pow(x, val);}, data, len, out);
// }
// template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
// static inline void pows_inplace(data_t1* data, len_t len, data_t2 val) {
// 	map_inplace([val](data_t1 x){return std::pow(x, val);}, data, len);
// }
// template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
// static inline unique_ptr<double[]> pow(const data_t1* data, len_t len,
// 	data_t2 val)
// {
// 	return map([val](data_t1 x) {return std::pow(x, val);}, data, len);
// }
// template<template <class...> class Container, class data_t1, class data_t2>
// static inline Container<double> pow(const Container<data_t1>& data,
// 	data_t2 val)
// {
// 	return map([val](data_t1 x) {return std::pow(x, val);}, data);
// }

// // ------------------------ scalar to power of elements

// template <class data_t1, class data_t2, class data_t3, class len_t,
// 	REQUIRE_INT(len_t)>
// static inline void pow(data_t2 val, data_t1 *RESTRICT data, len_t len,
// 	data_t3 *RESTRICT out)
// {
// 	map([val](data_t1 x){return std::pow(val, x);}, data, len, out);
// }
// template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
// static inline void pow_inplace(data_t2 val, data_t1* data, len_t len) {
// 	map_inplace([val](data_t1 x){return std::pow(val, x);}, data, len);
// }
// template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
// static inline unique_ptr<double[]> pow(data_t2 val, const data_t1* data,
// 	len_t len)
// {
// 	return map([val](data_t1 x) {return std::pow(val, x);}, data, len);
// }
// template<template <class...> class Container, class data_t1, class data_t2>
// static inline Container<double> pow(data_t2 val,
// 	const Container<data_t1>& data)
// {
// 	return map([val](data_t1 x) {return std::pow(val, x);}, data);
// }

// ================================ Copy

/** Copies src[0..len-1] to dest[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void copy(const data_t* src, len_t len, data_t* dest) {
	std::copy(src, src+len, dest);
}
/** Returns a copy of the provided array */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> copy(const data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	std::copy(data, data+len, ret.get());
	return ret;
}
/** Returns a copy of the provided array */
template<template <class...> class Container, class data_t>
static inline Container<data_t> copy(const Container<data_t>& data) {
	Container<data_t> ret(data);
	return ret;
}

// ================================================================
// V -> V
// ================================================================

#define WRAP_UNARY_FUNC_WITH_NAME(FUNC, NAME) \
\
template <class data_t, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline void NAME(data_t *RESTRICT data, len_t len, \
	data_t2 *RESTRICT out) \
{ \
	map([](data_t x){return FUNC(x);}, data, len, out); \
} \
template <class data_t, class len_t, REQUIRE_INT(len_t)> \
static inline void NAME ## _inplace (data_t* data, len_t len) { \
	map_inplace([](data_t x){return FUNC(x);}, data, len); \
} \
template <class data_t, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t* data, len_t len) \
	-> unique_ptr<decltype(FUNC(data[0]))[]> \
{ \
	return map([](data_t x) {return FUNC(x);}, data, len); \
} \
template<template <class...> class Container, class data_t> \
static inline auto NAME(const Container<data_t>& data) \
	-> Container<decltype(FUNC(*std::begin(data)))> \
{ \
	return map([](data_t x) {return FUNC(x);}, data); \
} \

#define WRAP_UNARY_FUNC(FUNC) WRAP_UNARY_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_UNARY_STD_FUNC_WITH_NAME(FUNC, NAME) \
using std::FUNC; \
WRAP_UNARY_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_UNARY_STD_FUNC(FUNC) WRAP_UNARY_STD_FUNC_WITH_NAME(FUNC, FUNC)

// exponents and logs
WRAP_UNARY_FUNC(abs);
WRAP_UNARY_STD_FUNC(sqrt);
WRAP_UNARY_STD_FUNC(cbrt);
WRAP_UNARY_STD_FUNC(exp);
WRAP_UNARY_STD_FUNC(exp2);
WRAP_UNARY_STD_FUNC(log);
WRAP_UNARY_STD_FUNC(log2);
WRAP_UNARY_STD_FUNC(log10);

// trig
WRAP_UNARY_STD_FUNC(sin);
WRAP_UNARY_STD_FUNC(asin);
WRAP_UNARY_STD_FUNC(sinh);
WRAP_UNARY_STD_FUNC(cos);
WRAP_UNARY_STD_FUNC(acos);
WRAP_UNARY_STD_FUNC(cosh);
WRAP_UNARY_STD_FUNC(tan);
WRAP_UNARY_STD_FUNC(atan);
WRAP_UNARY_STD_FUNC(tanh);

// err and gamma
WRAP_UNARY_STD_FUNC(erf);
WRAP_UNARY_STD_FUNC(erfc);
WRAP_UNARY_STD_FUNC_WITH_NAME(lgamma, log_gamma);
WRAP_UNARY_STD_FUNC_WITH_NAME(tgamma, gamma);

// rounding
WRAP_UNARY_STD_FUNC(ceil);
WRAP_UNARY_STD_FUNC(floor);
WRAP_UNARY_STD_FUNC(round);

// boolean
WRAP_UNARY_FUNC(logical_not);
WRAP_UNARY_FUNC(bitwise_not);
WRAP_UNARY_STD_FUNC(isnan);
WRAP_UNARY_STD_FUNC(isinf);
WRAP_UNARY_STD_FUNC(isfinite);

// ================================ Reverse

template <class data_t, class len_t, REQUIRE_INT(len_t)> // TODO test this
static inline void reverse_inplace(const data_t* data, len_t len) {
	for (len_t i = 0; i < len / 2; i++) {
		std::swap(data[i], data[len-i-1]);
	}
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void reverse(const data_t *RESTRICT src, data_t *RESTRICT dest,
	len_t len)
{
	len_t j = len - 1;
	for (len_t i = 0; i < len; i++, j--) {
		dest[i] = src[j];
	}
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> reverse(const data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_reverse(data, ret, len);
	return ret;
}
template<template <class...> class Container, class data_t>
static inline Container<data_t> reverse(const Container<data_t>& data) {
	Container<data_t> ret(data.size());
	array_reverse(data, ret.data(), data.size());
	return ret;
}

// ================================ Resample

/** Writes the elements of src to dest such that
 * dest[i] = src[ floor(i*srcLen/destLen) ]; note that this function does no
 * filtering of any kind */
template <class data_t>
static inline void resample(const data_t *src, data_t *dest,
	length_t srcLen, length_t destLen)
{
	length_t srcIdx;
	data_t scaleFactor = ((double)srcLen) / destLen;
	for(length_t i = 0; i < destLen; i++) {
		srcIdx = i * scaleFactor;
		dest[i] = src[srcIdx];
	}
}
template <class data_t>
static inline unique_ptr<data_t[]> resample(const data_t* data,
	length_t currentLen, length_t newLen)
{
	unique_ptr<data_t[]> ret(new data_t[newLen]);
	resample(data, ret.get(), currentLen, newLen);
	return ret;
}
template<template <class...> class Container, class data_t>
static inline Container<data_t> resample(const Container<data_t>& data,
	length_t newLen)
{
	Container<data_t> ret(newLen);
	resample(data.data(), ret.data(), data.size(), newLen);
	return ret;
}

// ================================ Pad

enum PadType {PAD_ZERO, PAD_CONSTANT, PAD_WRAP, PAD_EDGE};

template<class data_t1, class data_t2, class data_t3=data_t1>
static void pad(const data_t1* data, length_t len,
	length_t leftPadLen, length_t rightPadLen, data_t2* out,
	PadType padType=PAD_ZERO, data_t3 val=0)
{
	leftPadLen = max(leftPadLen, 0);
	rightPadLen = max(rightPadLen, 0);
	length_t newLen = leftPadLen + len + rightPadLen;

	// handle cases where we append fixed values to start and end
	data_t3 initialVal = val;
	data_t3 finalVal = val;
	if (padType == PAD_ZERO) {
		initialVal = 0;
		finalVal = 0;
	} else if (padType == PAD_EDGE) {
		initialVal = data[0];
		finalVal = data[len-1];
	}
	if (padType == PAD_ZERO || padType == PAD_CONSTANT || padType == PAD_EDGE) {
		for (length_t i = 0; i < leftPadLen; i++) {
			out[i] = initialVal;
		}
		copy(data, len, out+leftPadLen);
		for (length_t i = leftPadLen + len; i < newLen; i++) {
			out[i] = finalVal;
		}
	}
	return;

	// handle cases where we append different values to start and end
	const data_t1* initialAr = nullptr;
	const data_t1* finalAr = nullptr;
	if (padType == PAD_WRAP) {
		initialAr = &data[len - rightPadLen];
		finalAr = data;
	}

	assert(padType == PAD_WRAP); // no other pad types supported at present

	for (length_t i = 0; i < leftPadLen; i++) {
		out[i] = initialAr[i];
	}
	copy(data, len, out+leftPadLen);
	for (length_t i = 0; i < rightPadLen; i++) {
		out[leftPadLen + len + i] = finalAr[i];
	}
}

template<class data_t1, class data_t3=data_t1>
static unique_ptr<data_t1> pad(const data_t1* data, length_t len,
	length_t leftPadLen, length_t rightPadLen, PadType padType=PAD_ZERO,
	data_t3 val=0)
{
	leftPadLen = max(leftPadLen, 0);
	rightPadLen = max(rightPadLen, 0);
	auto newLen = leftPadLen + len + rightPadLen;
	unique_ptr<data_t1> ret(new data_t1[newLen]);
	pad(data, len, leftPadLen, rightPadLen, ret, padType, val);
	return ret;
}

template<template <class...> class Container1, class data_t1,
	class data_t3=data_t1>
static Container1<data_t1> pad(const Container1<data_t1>& data,
	length_t leftPadLen, length_t rightPadLen, PadType padType=PAD_ZERO,
	data_t3 val=0)
{
	leftPadLen = max(leftPadLen, 0);
	rightPadLen = max(rightPadLen, 0);
	auto newLen = leftPadLen + data.size() + rightPadLen;
	Container1<data_t1> ret(newLen);
	pad(data.data(), data.size(), leftPadLen, rightPadLen,
		ret.data(), padType, val);
	return ret;
}


// ================================ Equality

// TODO binary scalar op eq(a, b[, thresh])

/** Returns true if elements 0..(len-1) of x and y are equal, else false */
template <class data_t1, class data_t2, class float_t=double>
static inline bool all_eq(const data_t1 *x, const data_t2 *y, int64_t len,
	float_t thresh=kDefaultNonzeroThresh)
{
	for (int64_t i = 0; i < len; i++) {
		if (abs(x[i] - y[i]) > thresh) return false;
	}
	return true;
}
//template<template <class...> class Container1, class... Args1,
//	template <class...> class Container2, class... Args2, class float_t=double>
//static inline bool all_eq(const Container1<Args1>& x,
//	const Container2<Args2>& y, float_t thresh=kDefaultNonzeroThresh) {
//	if (x.size() != y.size()) return 0;
//	return all_eq(x.data(), y.data(), x.size());
//}
template<class Container1, class Container2, class float_t=double,
    REQUIRE_NOT_PTR(Container1), REQUIRE_NOT_PTR(Container2)>
static inline bool all_eq(const Container1& x, const Container2& y,
                          float_t thresh=kDefaultNonzeroThresh)
{
    if (x.size() != y.size()) return 0;
    return all_eq(x.data(), y.data(), x.size());
}

// ================================ All

/** Returns true iff func(x[i]) is true for all i */
template <class F, class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool all(const F&& func, const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (!func(x[i])) return false;
	}
	return true;
}
/** Returns true iff func(i, x[i]) is true for all i */
template <class F, class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool alli(const F&& func, const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (!func(i, x[i])) return false;
	}
	return true;
}
/** Returns true iff x[i] is true for all i */
template <class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool all(const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (!x[i]) return false;
	}
	return true;
}

template<class F, template <class...> class Container1, class data_t1>
static inline bool all(const F&& func, const Container1<data_t1>& x) {
//	return all(std::forward<F>(func), x.data(), x.size());
	for (auto it = begin(x); it < end(x); it++) {
		if (!func(*it)) return false;
	}
	return true;
}

template<class F, template <class...> class Container1, class data_t1>
static inline bool alli(const F&& func, const Container1<data_t1>& x) {
	length_t i = 0;
	for (auto it = begin(x); it < end(x); it++) {
		if (!func(i, *it)) return false;
		i++;
	}
	return true;
}

/** Returns true iff x[i] is true for all i */
template<template <class...> class Container1, class data_t1>
static inline bool all(const Container1<data_t1>& x) {
	return all(x.data(), x.size());
}

// ================================ Any

/** Returns true iff func(x[i]) is true for any i */
template <class F, class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool any(const F&& func, const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (func(x[i])) return true;
	}
	return false;
}
/** Returns true iff func(i, x[i]) is true for any i */
template <class F, class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool anyi(const F&& func, const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (func(i, x[i])) return true;
	}
	return false;
}
/** Returns true iff x[i] is true for any i */
template <class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool any(const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (x[i]) return true;
	}
	return false;
}

template<class F, template <class...> class Container1, class data_t1>
static inline bool any(const F&& func, const Container1<data_t1>& x) {
	for (auto it = begin(x); it < end(x); it++) {
		if (func(*it)) return true;
	}
	return false;
}
template<class F, template <class...> class Container1, class data_t1>
static inline bool anyi(const F&& func, const Container1<data_t1>& x) {
	length_t i = 0;
	for (auto it = begin(x); it < end(x); it++) {
		if (func(i, *it)) return true;
		i++;
	}
	return false;
}
/** Returns true iff x[i] is true for any i */
template<template <class...> class Container1, class data_t1>
static inline bool any(const Container1<data_t1>& x) {
	return any(x.data(), x.size());
}

// ================================ Nonnegativity
/** Returns true if elements 0..(len-1) of x are >= 0, else false */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline bool all_nonnegative(const data_t* data, len_t len) {
	return all([](data_t x){ return x >= 0; }, data, len);
}
template<template <class...> class Container, class data_t>
static inline bool all_nonnegative(const Container<data_t>& data) {
	return all([](data_t x){ return x >= 0; }, data);
}

// ================================ Positivity
/** Returns true if elements 0..(len-1) of x are > 0, else false */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline bool all_positive(const data_t *data, len_t len) {
	return all([](data_t x){ return x >= 0; }, data, len);
}
template<template <class...> class Container1, class data_t>
static inline bool all_positive(const Container1<data_t>& data) {
	return all([](data_t x){ return x > 0; }, data);
}

// ================================ Finite
// ------------------------ actually compute truth for float arrays
template <class data_t, REQUIRE_FLOAT(data_t), class len_t, REQUIRE_INT(len_t)>
static inline bool all_finite(const data_t *data, len_t len) {
	return all([](data_t x){ return isfinite(x); }, data, len);
}
template<template <class...> class Container1, class data_t,
	REQUIRE_FLOAT(data_t)>
static inline bool all_finite(const Container1<data_t>& data) {
	return all([](data_t x){ return isfinite(x); }, data);
}
// // ------------------------ ints are always finite
// template <class data_t, REQUIRE_INT(data_t), class len_t, REQUIRE_INT(len_t)>
// static inline bool all_finite(const data_t *data, len_t len) {
// 	return true;
// }
// template<template <class...> class Container1, class data_t,
// 	REQUIRE_INT(data_t)>
// static inline bool all_finite(const Container1<data_t>& data) {
// 	return true;
// }

// ================================ Unique
template<template <class...> class Container, class data_t>
static inline Container<data_t> unique(const Container<data_t>& data) {
	Container<data_t> sorted(data);
	auto begin = std::begin(sorted);
	auto end = std::end(sorted);
	std::sort(begin, end);

	Container<data_t> ret;
	std::unique_copy(begin, end, std::back_inserter(ret));
	return ret;
}
template<class data_t>
static inline vector<data_t> unique(const data_t *data, length_t len) {
	vector<data_t> tmp(data, data + len);
	return unique(tmp);
}

// ================================================================
// Normalizing
// ================================================================

#define WRAP_NORMALIZE_FUNC(FUNC) \
template <class data_t, class len_t, class float_t=double, REQUIRE_INT(len_t), \
	REQUIRE_NUM(float_t)> \
static inline unique_ptr<data_t[]> FUNC(const data_t* data, len_t len, \
	float_t nonzeroThresh=kDefaultNonzeroThresh) \
{ \
	unique_ptr<data_t[]> ret(new data_t[len]); \
	FUNC(data, len, ret.get(), nonzeroThresh); \
	return ret; \
} \
template <template <class...> class Container, class data_t, \
	class float_t=double, REQUIRE_NUM(float_t)> \
static inline Container<data_t> FUNC(Container<data_t> data, \
	float_t nonzeroThresh=kDefaultNonzeroThresh) \
{ \
	Container<data_t> ret(data.size()); \
	FUNC(data.data(), data.size(), ret.data(), nonzeroThresh); \
	return ret; \
} \

// ------------------------ znormalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool znormalize(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	double mean, variance;
	mean_and_variance(data, len, mean, variance);
	double std = std::sqrt(variance);
	if (std < nonzeroThresh) {
		return false;
	}
	for (len_t i = 0; i < len; i++) {
		out[i] = static_cast<data_t2>((data[i] - mean) / std);
	}
	return true;
}
template<class data_t1, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool znormalize_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	double mean, variance;
	mean_and_variance(data, len, mean, variance);
	double std = std::sqrt(variance);
	if (std < nonzeroThresh) {
		return false;
	}
	for (len_t i = 0; i < len; i++) {
		data[i] = static_cast<data_t1>((data[i] - mean) / std);
	}
	return true;
}
WRAP_NORMALIZE_FUNC(znormalize)

// ------------------------ mean normalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_mean(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto avg = mean(data, len);
	sub(data, len, avg, out);
	return true;
}
template<class data_t1, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_mean_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto avg = mean(data, len);
	sub_inplace(data, len, avg);
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_mean)

// ------------------------ std normalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_stdev(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto std = stdev(data, len);
	if (std < nonzeroThresh) {
		return false;
	}
	div(data, len, std, out);
	return true;
}
template<class data_t1, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_stdev_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto std = stdev(data, len);
	if (std < nonzeroThresh) {
		return false;
	}
	div_inplace(data, len, std);
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_stdev)

// ------------------------ L1 normalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_L1(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto norm = sum(data, len);
	if (norm < nonzeroThresh) {
		return false;
	}
	div(data, len, norm, out);
	return true;
}
template<class data_t1, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_L1_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto norm = sum(data, len);
	if (norm < nonzeroThresh) {
		return false;
	}
	div_inplace(data, len, norm);
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_L1)

// ------------------------ L2 normalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_L2(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto norm = sqrt(sumsquares(data, len));
	if (norm < nonzeroThresh) {
		return false;
	}
	div(data, len, norm, out);
	return true;
}
template<class data_t1, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_L2_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto norm = sqrt(sumsquares(data, len));
	if (norm < nonzeroThresh) {
		return false;
	}
	div_inplace(data, len, norm);
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_L2)

// ------------------------ Linf norm (max norm)

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_max(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	data_t1 norm = max(data, len);
	if (norm < nonzeroThresh) {
		return false;
	}
	div(data, len, norm, out);
	return true;
}
template<class data_t1, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_max_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	data_t1 norm = max(data, len);
	if (norm < nonzeroThresh) {
		return false;
	}
	div_inplace(data, len, norm);
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_max)

// ------------------------ 0-1 normalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_zero_one(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto minVal = min(data, len);
	auto maxVal = max(data, len);
	double range = maxVal - minVal;
	if (range < nonzeroThresh) {
		return false;
	}
	for (len_t i = 0; i < len; i++) {
		out[i] = static_cast<data_t2>((data[i] - minVal) / range);
	}
	return true;
}
template<class data_t1, class len_t, class float_t=double,
	REQUIRE_INT(len_t), REQUIRE_NUM(float_t)>
static inline bool normalize_zero_one_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto minVal = min(data, len);
	auto maxVal = max(data, len);
	double range = maxVal - minVal;
	if (range < nonzeroThresh) {
		return false;
	}
	for (len_t i = 0; i < len; i++) {
		data[i] = static_cast<data_t1>((data[i] - minVal) / range);
	}
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_zero_one)

// ================================================================
// Stringification / IO
// ================================================================

// ================================ Stringification

// ------------------------ with name

template <class data_t, class len_t, class cast_to=data_t, REQUIRE_INT(len_t)>
static std::string to_string(const data_t *x, len_t len, const char* name="")
{
	std::ostringstream os;
	os.precision(3);
	if (name && name[0] != '\0') {
		os << name << ": ";
	}
	// printf("received length %d\n", len);
	// printf("cast to is int? %d\n", static_cast<cast_to>(x[0]));
	os << "[";
	for (len_t i = 0; i < len; i++) {
		os << std::to_string(static_cast<cast_to>(x[i])) << " ";
	}
	os << "]";
	return os.str();
}
template <class data_t, class len_t, class cast_to=data_t, REQUIRE_INT(len_t)>
static std::string to_string(const data_t *x, len_t len, std::string name)
{
	return to_string<data_t, len_t, cast_to>(x, len, name.c_str());
}
template<template <class...> class Container, class data_t, class cast_to=data_t>
static inline std::string to_string(const Container<data_t>& data,
	const char* name="")
{
	return to_string<data_t, decltype(data.size()), cast_to>(
		data.data(), data.size(), name);
}
template<template <class...> class Container, class data_t, class cast_to=data_t>
static inline std::string to_string(const Container<data_t>& data,
	std::string name)
{
	return to_string<data_t, decltype(data.size()), cast_to>(
		data.data(), data.size(), name);
}

// ================================ Printing
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void print(const data_t *x, len_t len, const char* name="") {
	printf("%s\n", to_string(x, len, name).c_str());
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void print(const data_t *x, len_t len, std::string name) {
	printf("%s\n", to_string(x, len, name).c_str());
}

template<template <class...> class Container, class data_t>
static inline void print(const Container<data_t>& data, const char* name="") {
	print(data.data(), data.size(), name);
}
template<template <class...> class Container, class data_t>
static inline void print(const Container<data_t>& data, std::string name) {
	print(data.data(), data.size(), name);
}

// ================================================================
// Randomness
// ================================================================

// TODO sample without replacement using:
// http://lemire.me/blog/2013/08/16/picking-n-distinct-numbers-at-random-how-to-do-it-fast/
// TODO use PCG for random number generation
// https://github.com/imneme/pcg-cpp

// ================================ Random Number Generation

// utility func for rand_ints
template<template <class...> class Container, typename K, typename V>
inline V map_get(Container<K, V> map, K key, V defaultVal) {
	if (map.count(key)) {
		return map[key];
	}
	return defaultVal;
}

// ------------------------ rand_ints

static inline vector<int64_t> rand_ints(int64_t minVal, int64_t maxVal,
	uint64_t howMany, bool replace=false)
{
	vector<int64_t> ret;
	int64_t numPossibleVals = maxVal - minVal + 1;

//	assertf(numPossibleVals >= 1, "rand_ints(): no values between"
//			"min %lld and max %lld", minVal, maxVal);
//
//	assertf(replace || (numPossibleVals >= howMany),
//		"rand_ints(): can't sample %llu values without replacement between "
//		"min %lld and max %lld", howMany, minVal, maxVal);

	if (replace) {
		for (size_t i = 0; i < howMany; i++) {
			int64_t val = (rand() % numPossibleVals) + minVal;
			ret.push_back(val);
		}
		return ret;
	}

	// if returning all possible values, or within a constant factor thereof,
	// just shuffle the set of possible values and take the first howMany
	if (howMany >  numPossibleVals / 2) {
		ret = range(minVal, maxVal + 1);
		std::random_shuffle(std::begin(ret), std::end(ret));
		ret.resize(howMany);
		return ret;
	}

	// sample without replacement; each returned int is unique
	unordered_map<int64_t, int64_t> possibleIdxs;
	int64_t idx;
	for (size_t i = 0; i < howMany; i++) {
		idx = (rand() % (numPossibleVals - i)) + minVal;

		// next value to add to array; just the idx, unless we've picked this
		// idx before, in which case it's whatever the highest unused value
		// was the last time we picked it
		auto val = map_get(possibleIdxs, idx, idx);

		// move highest unused idx into this idx; the result is that the
		// first numPossibleVals-i idxs are all available idxs
		int64_t highestUnusedIdx = maxVal - i;
		possibleIdxs[idx] = map_get(possibleIdxs, highestUnusedIdx,
									highestUnusedIdx);

		ret.push_back(val);
	}
	return ret;
}

template<class float_t>
static inline vector<int64_t> rand_ints(int64_t minVal, int64_t maxVal,
	uint64_t howMany, bool replace=false, const float_t* probs=nullptr)
{
	vector<int64_t> ret;
	int64_t numPossibleVals = maxVal - minVal + 1;

	bool willReturnEverything = (numPossibleVals == howMany) && !replace;
	if ( (!probs) || willReturnEverything) {
		return rand_ints(minVal, maxVal, howMany, replace);
	}
//	assertf(numPossibleVals >= 1, "rand_ints(): no values between"
//			"min %lld and max %lld", minVal, maxVal);
//
//	assertf(replace || (numPossibleVals >= howMany),
//			"rand_ints(): can't sample %llu values without replacement"
//			"between min %lld and max %lld", howMany, minVal, maxVal);
//
//	assertf(all_finite(probs, numPossibleVals),
//			"Probabilities must be finite!");
//
//	assertf(all_nonnegative(probs, numPossibleVals),
//		"Probabilities must be nonnegative!");
	auto totalProb = sum(probs, numPossibleVals);
	assertf(totalProb > 0, "rand_ints(): probabilities sum to a value <= 0");

    // init random distro object
	std::random_device rd;
    std::mt19937 gen(rd());
	auto possibleIdxsAr = range(minVal, maxVal+2); // end range at max+1
	std::piecewise_constant_distribution<float_t> distro(
		std::begin(possibleIdxsAr), std::end(possibleIdxsAr), probs);

	if (replace) {
		for (size_t i = 0; i < howMany; i++) {
			ret.push_back(static_cast<int64_t>(distro(gen)));
		}
		return ret;
	}

	// if we would be selecting most of the values, it will take a long time
	// to get this many distinct values randomly; instead, select the values
	// not to return randomly and then return everything else
	if (howMany > numPossibleVals / 2) {
		auto notThese = rand_ints(minVal, maxVal, numPossibleVals - howMany,
			replace, probs);

		// check whether idx exists in notThese; if not, insert it; fast
		// checks by sorting notThese first
		std::sort(std::begin(notThese), std::end(notThese));
		int notIdx = 0;
		for (int64_t idx = minVal; idx <= maxVal; idx++) {
			// invariant: notThese[notIdx] >= idx or undefined
			while (idx > notThese[notIdx] && notIdx < notThese.size()) {
				notIdx++;
			}
			if (idx != notThese[notIdx]) {
				ret.push_back(idx);
			}
		}
		return ret;
	}

	auto howManyPossible = nonzeros(probs, maxVal - minVal + 1).size();
	if (howManyPossible < howMany) {
		printf("WARNING: rand_ints(): only %lu ints have nonzero probability,"
			"but %lld were requested; returning the smaller value\n",
			howManyPossible, howMany);
	}
	howMany = min(howMany, howManyPossible);


	// sample without replacement; each returned int is unique
	unordered_set<int64_t> usedIdxs;
	usedIdxs.reserve(howMany);
	int64_t abandonAfter = howMany * howMany * howMany;
	int watchdog = 0;
	while(usedIdxs.size() < howMany && watchdog < abandonAfter) {
		int64_t idx = static_cast<int64_t>(distro(gen));
		if (! usedIdxs.count(idx)) { // if new idx
			usedIdxs.insert(idx);
			ret.push_back(idx);
		}
		watchdog++;
	}
	if (watchdog == abandonAfter) {
		printf("WARNING: rand_ints(): only sampled %lu ints"
			"without replacement; check for very small probabilities\n",
			ret.size());
	}

	return ret;
}

template<template <class...> class Container, class float_t>
static inline vector<int64_t> rand_ints(int64_t minVal, int64_t maxVal,
	uint64_t howMany, bool replace, const Container<float_t>& probs)
{
	int64_t numPossibleVals = maxVal - minVal + 1;
	assertf(probs.size() == numPossibleVals,
		"Number of probabilities %llu doesn't match number of possible values %lld",
		probs.size(), numPossibleVals);
	return rand_ints(minVal, maxVal, howMany, replace, probs.data());
}

// ------------------------ rand_idxs

template<class float_t=float>
static inline vector<int64_t> rand_idxs(uint64_t len, uint64_t howMany,
	bool replace=false, const float_t* probs=nullptr)
{
	return rand_ints(0, len-1, howMany, replace, probs);
}
template<template <class...> class Container, class float_t>
static inline vector<int64_t> rand_idxs(uint64_t len, uint64_t howMany,
	bool replace, const Container<float_t>& probs)
{
	return rand_ints(0, len-1, howMany, replace, probs);
}

// ================================ Random Sampling

template<template <class...> class Container, class data_t>
static inline vector<data_t> rand_choice(const Container<data_t>& data,
	size_t howMany, bool replace=false)
{
	// auto maxIdx = data.size() - 1;
	// auto idxs = rand_ints(0, maxIdx, howMany, replace);
	auto idxs = rand_idxs(data.size(), howMany, replace);
	return at_idxs(data, idxs);
}

template<template <class...> class Container1, class data_t,
	template <class...> class Container2, class float_t>
static inline vector<data_t> rand_choice(const Container1<data_t>& data,
	size_t howMany, bool replace, const Container2<float_t>& probs)
{
	// auto maxIdx = data.size() - 1;
	// auto idxs = rand_ints(0, maxIdx, howMany, replace, probs);
	auto idxs = rand_idxs(data.size(), howMany, replace, probs);
	return at_idxs(data, idxs);
}

// ================================ Random Data Generation

// ------------------------ iid gaussians

template <class data_t, class len_t, REQUIRE_INT(len_t), class float_t=double>
static inline void randn_inplace(const data_t* data, len_t len,
	float_t mean=0., float_t std=1) {
	assert(len > 0);

	// create normal distro object
	std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, std);

	for (len_t i = 0; i < len; i++) {
		data[i] = static_cast<data_t>(d(gen));
	}
}
// note that this must be called as, e.g., randn<double>(...)
template <class data_t, class len_t, REQUIRE_INT(len_t), class float_t=double>
static inline unique_ptr<data_t[]> randn(len_t len,
	float_t mean=0., float_t std=1)
{
	assert(len > 0);
	unique_ptr<data_t[]> ret(new data_t[len]);
	randn_inplace(ret, len, mean, std);
	return ret;
}

// ------------------------ gaussian random walk

template <class data_t, class len_t, REQUIRE_INT(len_t), class float_t=double>
static inline void randwalk_inplace(const data_t* data, len_t len, float_t std=1) {
	assert(len > 0);

	// create normal distro object
	std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0., std);

    data[0] = static_cast<data_t>(d(gen));
	for (len_t i = 1; i < len; i++) {
		data[i] = data[i-1] + static_cast<data_t>(d(gen));
	}
}
// note that this must be called as, e.g., randwalk<double>(...)
template <class data_t, class len_t, REQUIRE_INT(len_t), class float_t=double>
static inline unique_ptr<data_t[]> randwalk(len_t len, float_t std=1)
{
	assert(len > 0);
	unique_ptr<data_t[]> ret(new data_t[len]);
	randwalk_inplace(ret, len, std);
	return ret;
}

// ================================================================
// Miscellaneous
// ================================================================

// ================================ Sorting

template<template <class...> class Container, class data_t>
static inline void sort_inplace(Container<data_t>& data) {
	std::sort(std::begin(data), std::end(data));
}

template<template <class...> class Container, class data_t>
static inline Container<data_t> sort(const Container<data_t>& data) {
	Container<data_t> ret(data);
	sort_inplace(ret);
	return ret;
}

// ================================ Argsort

template<class data_t, class len_t, REQUIRE_INT(len_t)>
static void argsort(const data_t* data, length_t len, len_t* out,
	bool ascending=true)
{
	typedef std::pair<length_t, data_t> pair;

	// pair idx with value
	auto pairs = mapi([](length_t i, data_t x) {
		return pair(i, x);
	}, data, len);
	// sort pairs by value
	if (ascending) {
		std::sort(pairs.get(), pairs.get() + len,
		[](const pair& lhs, const pair& rhs) {
			return lhs.second < rhs.second;
		});
	} else {
		std::sort(pairs.get(), pairs.get() + len,
		[](const pair& lhs, const pair& rhs) {
			return lhs.second > rhs.second;
		});
	}
	// return idxs that yield values in sorted order
	map([](const pair& p) { return p.first; }, pairs.get(), len, out);
}
template<class data_t>
static inline unique_ptr<length_t[]> argsort(const data_t* data, length_t len,
	bool ascending=true)
{
	unique_ptr<length_t[]> ret(new length_t[len]);
	argsort(data, len, ret, ascending);
	return ret;
}

template<template <class...> class Container, class data_t>
static inline Container<length_t> argsort(const Container<data_t>& data,
	bool ascending=true)
{
	Container<length_t> ret(data.size());
	argsort(data.data(), data.size(), ret.data(), ascending);
	return ret;
}


} // namespace ar
#endif











