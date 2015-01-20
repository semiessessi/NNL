// Copyright (c) 2015 Cranium Software

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
    const float fSquareRootDenominator = expf( fX ) + 1.0f;
    return expf( fX ) / ( fSquareRootDenominator * fSquareRootDenominator );
}

}

#endif
