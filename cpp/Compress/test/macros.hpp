//
//  macros.hpp
//
//  Created By Davis Blalock on 3/15/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __MACROS_HPP
#define __MACROS_HPP

// ------------------------ restrict keyword
// adapted from http://stackoverflow.com/a/5948101/1153180

#if defined(__GNUC__) && ((__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
	#define RESTRICT __restrict__

	#define LIKELY(x)    __builtin_expect (!!(x), 1)
	#define UNLIKELY(x)  __builtin_expect (!!(x), 0)

	#define PREFETCH_WITH_STICKINESS(ADDR, INT) \
		__builtin_prefetch(ADDR, 0, INT)
	#define PREFETCH_RW_WITH_STICKINESS(ADDR, INT) \
		__builtin_prefetch(ADDR, 0, INT)
	#define PREFETCH_TRANSIENT(ADDR) PREFETCH_WITH_STICKINESS(ADDR, 0)
	#define PREFETCH_PERSISTENT(ADDR) PREFETCH_WITH_STICKINESS(ADDR, 3)
	#define PREFETCH_RW_TRANSIENT(ADDR) PREFETCH_RW_WITH_STICKINESS(ADDR, 0)
	#define PREFETCH_RW_PERSISTENT(ADDR) PREFETCH_RW_WITH_STICKINESS(ADDR, 3)
#elif defined(__clang__)
	#define RESTRICT __restrict__

	#define LIKELY(x)    __builtin_expect (!!(x), 1)
	#define UNLIKELY(x)  __builtin_expect (!!(x), 0)

	#define PREFETCH_WITH_STICKINESS(ADDR, INT) \
		__builtin_prefetch(ADDR, 0, INT)
	#define PREFETCH_RW_WITH_STICKINESS(ADDR, INT) \
		__builtin_prefetch(ADDR, 1, INT)
	#define PREFETCH_TRANSIENT(ADDR) PREFETCH_WITH_STICKINESS(ADDR, 0)
	#define PREFETCH_PERSISTENT(ADDR) PREFETCH_WITH_STICKINESS(ADDR, 3)
	#define PREFETCH_RW_TRANSIENT(ADDR) PREFETCH_RW_WITH_STICKINESS(ADDR, 0)
	#define PREFETCH_RW_PERSISTENT(ADDR) PREFETCH_RW_WITH_STICKINESS(ADDR, 3)
#else
	#if defined(_MSC_VER) && _MSC_VER >= 1400
		#define RESTRICT
	#else
		#define RESTRICT __restrict
	#endif

	#define LIKELY(x)    (x)
	#define UNLIKELY(x)  (x)

	#define PREFETCH_WITH_STICKINESS(ADDR, INT)
	#define PREFETCH_RW_WITH_STICKINESS(ADDR, INT)
	#define PREFETCH_TRANSIENT(ADDR) PREFETCH_WITH_STICKINESS(ADDR, 0)
	#define PREFETCH_PERSISTENT(ADDR) PREFETCH_WITH_STICKINESS(ADDR, 3)
	#define PREFETCH_RW_TRANSIENT(ADDR) PREFETCH_RW_WITH_STICKINESS(ADDR, 0)
	#define PREFETCH_RW_PERSISTENT(ADDR) PREFETCH_RW_WITH_STICKINESS(ADDR, 3)

#endif

// count the number of arguments in a varargs list
#define VA_NUM_ARGS(...) _VA_NUM_ARGS_IMPL(__VA_ARGS__, \
	16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define _VA_NUM_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, \
	 _13, _14, _15, _16, N, ...) N

#ifdef __cplusplus
// uncomment below if using C++14
//    // ------------------------ type aliases // TODO macros file not ideal place
//	#include <experimental/optional>
//	// will be std::optional in c++17
//	template<class T> using optional = std::experimental::optional<T>;
//	static constexpr auto nullopt = std::experimental::nullopt;

    // ------------------------ type traits macros
	#include <type_traits>

	// put these in function bodies to statically assert that appropriate types
	// have been passed in as template params; prefer using the below type
	// constraint macros, however
	#define ASSERT_TRAIT(TRAIT, T, MSG) static_assert(std::TRAIT<T>::value, MSG)
	#define ASSERT_INTEGRAL(T) ASSERT_TRAIT(is_integral, T, "Type not integral!")

	// put these as extra template params to enforce constraints
	// on previous template params; e.g.:
	//
	// template<class T, REQUIRE_INT(T)> T foo(T arg) { return arg + 1; }
	//

	// ------------------------ require that some constexpr be true

	#define REQ(EXPR) \
		typename = typename std::enable_if<EXPR, void>::type

	// have to wrap EXPR in a local template param for enable_if to work on
	// a class method where EXPR is a class template param
	#define _METHOD_REQ(EXPR, NAME) \
		bool NAME = EXPR, typename = typename std::enable_if<NAME, void>::type

	#define METHOD_REQ(EXPR) \
		_METHOD_REQ(EXPR, __expr__)

	#define REQUIRE_TRAIT(TRAIT, T) \
		typename = typename std::enable_if<std::TRAIT<T>::value, T>::type

	#define REQUIRE_NOT_TRAIT(TRAIT, T) \
		typename = typename std::enable_if<!std::TRAIT<T>::value, T>::type

	#define REQUIRE_IS_A(BASE, T) \
		typename = typename std::enable_if<std::is_base_of<BASE, T>::value, T>::type

	#define REQUIRE_IS_NOT_A(BASE, T) \
		typename = typename std::enable_if<!std::is_base_of<BASE, T>::value, T>::type

    #define REQUIRE_SIGNED_INT(T) REQUIRE_TRAIT(is_integral, T)
	#define REQUIRE_INT(T)                                                  \
        typename = typename std::enable_if<(std::is_integral<T>::value ||   \
            !std::is_signed<T>::value), T>::type
	#define REQUIRE_NUM(T) REQUIRE_TRAIT(is_arithmetic, T)
	#define REQUIRE_FLOAT(T) REQUIRE_TRAIT(is_floating_point, T)
	#define REQUIRE_PRIMITIVE(T) REQUIRE_TRAIT(is_arithmetic, T)
	#define REQUIRE_NOT_PTR(T) REQUIRE_NOT_TRAIT(is_pointer, T)

	#define REQ_HAS_ATTR(T, METHOD_INVOCATION) \
		typename=decltype(std::declval<T>() . METHOD_INVOCATION)

	// ------------------------ is_valid; requires C++14
	// inspired by https://gist.github.com/Jiwan/7a586c739a30dd90d259

	template <typename T> struct _valid_helper {
	private:
	    template <typename Param> constexpr auto _is_valid(int _) const
		    // type returned by decltype is last type in the list (here,
			// std::true_type), but previous types must be valid
	    	-> decltype(std::declval<T>()(std::declval<Param>()),
	    		std::true_type())
	    {
	        return std::true_type();
	    }

	    template <typename Param> constexpr std::false_type _is_valid(...) const
        {
	        return std::false_type();
	    }

	public:
	    template <typename Param> constexpr auto operator()(const Param& p) const
            -> decltype(_is_valid<Param>(int(0)))
        {
	        return _is_valid<Param>(int(0));
	    }
	};

	template <typename T> constexpr auto is_valid(const T& t)
        -> decltype(_valid_helper<T>())
    {
	    return _valid_helper<T>();
	}

	#define CREATE_TEST(OBJNAME, EXPR) \
		is_valid([](auto&& OBJNAME) -> decltype(EXPR) { })

	#define CREATE_TEST_X(EXPR) \
		is_valid([](auto&& x) -> decltype(EXPR) { })

	#define TEST_FOR_METHOD(INVOCATION) \
		is_valid([](auto&& x) -> decltype(x. INVOCATION) { })

	#define PASSES_TEST(OBJ, TEST) \
		decltype(TEST(OBJ))::value

	#define TYPE_PASSES_TEST(T, TEST) \
		PASSES_TEST(std::declval<T>(), TEST)

	#define REQ_TYPE_PASSES(T, TEST) \
		REQ(TYPE_PASSES_TEST(T, TEST))

	#define ENABLE_IF(EXPR, T) \
		typename std::enable_if<EXPR, T>::type


    // ------------------------ TYPES(...) convenience macro for template args
    #define TYPES_1(A) template<typename A>

    #define TYPES_2(A, B) \
        template<typename A, typename B>

    #define TYPES_3(A, B, C) \
        template<typename A, typename B, typename C>

    #define TYPES_4(A, B, C, D) \
        template<typename A, typename B, typename C, typename D>

    #define TYPES_5(A, B, C, D, E) \
        template<typename A, typename B, typename C, typename D, typename E>

    #define TYPES_6(A, B, C, D, E, F) \
        template<typename A, typename B, typename C, typename D, typename E, \
            typename F>

    #define TYPES_7(A, B, C, D, E, F, G) \
        template<typename A, typename B, typename C, typename D, typename E, \
            typename F, typename G>

    #define TYPES_8(A, B, C, D, E, F, G, H) \
        template<typename A, typename B, typename C, typename D, typename E, \
            typename F, typename G, typename H>

    #define TYPES_9(A, B, C, D, E, F, G, H, I) \
        template<typename A, typename B, typename C, typename D, typename E, \
            typename F, typename G, typename H, typename I>

    #define TYPES_10(A, B, C, D, E, F, G, H, I, J) \
        template<typename A, typename B, typename C, typename D, typename E, \
            typename F, typename G, typename H, typename I, typename J

    #define _WRAP_VARIADIC_IMPL2(name, count, ...) \
            name ## count (__VA_ARGS__)
    #define _WRAP_VARIADIC_IMPL(name, count, ...) \
            _WRAP_VARIADIC_IMPL2(name, count, __VA_ARGS__)
    #define WRAP_VARIADIC(name, ...) \
            _WRAP_VARIADIC_IMPL(name, VA_NUM_ARGS(__VA_ARGS__), __VA_ARGS__)

    #define TYPES(...) WRAP_VARIADIC(TYPES, __VA_ARGS__)

    #undef _WRAP_VARIADIC_IMPL
    #undef _WRAP_VARIADIC_IMPL2
    #undef VA_NUM_ARGS
    #undef _VA_NUM_ARGS_IMPL

    // ------------------------ static size assertions from Eigen

    #define _PREDICATE_SAME_MATRIX_SIZE(TYPE0,TYPE1) \
        ( \
            (int(TYPE0::SizeAtCompileTime)==0 && int(TYPE1::SizeAtCompileTime)==0) \
        || (\
              (int(TYPE0::RowsAtCompileTime)==Eigen::Dynamic \
            || int(TYPE1::RowsAtCompileTime)==Eigen::Dynamic \
            || int(TYPE0::RowsAtCompileTime)==int(TYPE1::RowsAtCompileTime)) \
          &&  (int(TYPE0::ColsAtCompileTime)==Eigen::Dynamic \
            || int(TYPE1::ColsAtCompileTime)==Eigen::Dynamic \
            || int(TYPE0::ColsAtCompileTime)==int(TYPE1::ColsAtCompileTime))\
           ) \
        )

    #define STATIC_ASSERT_SAME_SHAPE(TYPE0, TYPE1) \
        static_assert(_PREDICATE_SAME_MATRIX_SIZE(TYPE0, TYPE1), \
            YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES)

    #undef _PREDICATE_SAME_MATRIX_SIZE


    #define PRINT_STATIC_TYPE(X) \
        static_assert(decltype(X)::__debug__, #X);

#endif // __cplusplus




#endif // __MACROS_HPP
