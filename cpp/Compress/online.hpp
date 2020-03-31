//
//  online.hpp
//  Compress
//
//  Created by DB on 3/31/20.
//  Copyright © 2020 D Blalock. All rights reserved.
//

#ifndef online_h
#define online_h

#include <stdint.h>

// NOTE: a lot of this would be cleaner with templates, but we're writing it
// such that it will be easy to port to C

struct DeltaForecaster_16 {
    using scalar_type = uint16_t;
    
    void init(scalar_type initial_val) {
        _prev = initial_val;
    }
    
    scalar_type predict() const {
        return _prev;
    }
    
    void train(scalar_type err) const {
        _prev = _prev + err;  // overflow is fine; just wraps around
    }
    
private:
    scalar_type _prev;
};

struct DoubleDeltaForecaster_16 {
    using scalar_type = uint16_t;
    
    void init(scalar_type initial_val) {
        _prev0 = initial_val;
        _prev1 = initial_val; // same val; extrapolate const line at first
    }
    
    scalar_type predict() const {
        return _prev + (prev0 - prev1);
    }
    
    void train(scalar_type err) const {
        _prev = _prev + err;
    }
    
private:
    scalar_type _prev;
};

struct DeltaCoder_u16 {
    using scalar_type = uint16_t;
  
    
    
    
private:
    scalar_type _prev;
};


#endif /* online_h */
