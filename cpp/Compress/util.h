//
//  util.h
//  Compress
//
//  Created by DB on 11/4/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#ifndef util_h
#define util_h

#ifndef MAX
    #define MAX(x, y) ( ((x) > (y)) ? (x) : (y) )
#endif
#ifndef MIN
    #define MIN(x, y) ( ((x) < (y)) ? (x) : (y) )
#endif

#if __cpp_constexpr >= 201304
    #define CONSTEXPR constexpr
#else
    #define CONSTEXPR
#endif

template<typename T, typename T2>
static CONSTEXPR inline T round_up_to_multiple(T x, T2 multipleof) {
    T remainder = x % multipleof;
    return remainder ? (x + multipleof - remainder) : x;
}

#endif /* util_h */
