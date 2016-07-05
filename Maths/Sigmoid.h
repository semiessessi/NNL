// Copyright (c) 2015-2016 Cranium Software

#ifndef SIGMOID_H
#define SIGMOID_H

#include <math.h>

namespace NNL
{

static inline float Sigmoid( const float fX )
{
    return 1.0f / ( 1.0f + expf( -fX ) );
}

static inline float SigmoidDerivative( const float fX )
{
    const float fSigmoid = Sigmoid( fX );
    return fSigmoid * ( 1.0f - fSigmoid );
}

// class to use for template metaprogramming stuff
class SigmoidSummingFunction
{

public:

    static float Evaluate( const float fSum )
    {
        return Sigmoid( fSum );
    }
    
    static float Derivative( const float fSum )
    {
        return SigmoidDerivative( fSum );
    }

};

}

#endif
