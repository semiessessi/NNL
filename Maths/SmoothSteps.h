// Copyright (c) 2015 Cranium Software

#ifndef SMOOTH_STEP_H
#define SMOOTH_STEP_H

namespace NNL
{

static inline float SmoothStep( const float fX )
{
    return ( 3.0f - 2.0 * fX ) * fX * fX;
}

static inline float SmoothStepDerivative( const float fX )
{
    return ( 6.0f - 6.0 * fX ) * fX;
}

static inline float SmootherStep( const float fX )
{
    return ( ( 6.0f * fX - 15.0f ) * fX + 10.0f ) * fX * fX * fX;
}

static inline float SmootherStepDerivative( const float fX )
{
    return ( ( 30.0f * fX - 60.0f ) * fX + 30.0f ) * fX * fX;
}

}

#endif
