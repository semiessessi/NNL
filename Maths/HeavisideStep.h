// Copyright (c) 2015-2016 Cranium Software

#ifndef HEAVISIDE_STEP_H
#define HEAVISIDE_STEP_H

namespace NNL
{

static inline float HeavisideStep( const float fX )
{
    return ( fX > 0.0f ) ? 1.0f : 0.0f;
}

// class to use for template metaprogramming stuff
class HeavisideStepSummingFunction
{

public:

    static float Evaluate( const float fSum )
    {
        return HeavisideStep( fSum );
    }
    
    static float Deriviative( const float fSum )
    {
        return 0.0f;
    }

};

}

#endif
