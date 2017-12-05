//
//  macros.h
//  Compress
//
//  Created by DB on 12/4/17.
//  Copyright © 2017 D Blalock. All rights reserved.
//

#if defined(__GNUC__)
    #define SPRINTZ_FORCE_INLINE __attribute__((always_inline))
#elif defined(__clang__)
    #define SPRINTZ_FORCE_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define SPRINTZ_FORCE_INLINE __forceinline
#else
    // this should still work if compiled as C99 without static keyword; see
    // https://stackoverflow.com/a/25623448
    #define SPRINTZ_FORCE_INLINE inline
#endif
