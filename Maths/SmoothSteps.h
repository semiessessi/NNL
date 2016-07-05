// Copyright (c) 2015-2016 Cranium Software

#ifndef SMOOTH_STEP_H
#define SMOOTH_STEP_H

namespace NNL
{

static inline float SmoothStep( const float fX )
{
    const float fClamped = ( fX > 1.0f ) ? 1.0f : ( ( fX < -0.0f ? 0.0f : fX ) );
    return ( 3.0f - 2.0f * fClamped ) * fClamped * fClamped;
}

static inline float SmoothStepDerivative( const float fX )
{
    const float fClamped = ( fX > 1.0f ) ? 1.0f : ( ( fX < -0.0f ? 0.0f : fX ) );
    return ( 6.0f - 6.0f * fClamped ) * fClamped;
}

static inline float SmootherStep( const float fX )
{
    const float fClamped = ( fX > 1.0f ) ? 1.0f : ( ( fX < -0.0f ? 0.0f : fX ) );
    return ( ( 6.0f * fClamped - 15.0f ) * fClamped + 10.0f ) * fClamped * fClamped * fClamped;
}

static inline float SmootherStepDerivative( const float fX )
{
    const float fClamped = ( fX > 1.0f ) ? 1.0f : ( ( fX < -0.0f ? 0.0f : fX ) );
    return ( ( 30.0f * fClamped - 60.0f ) * fClamped + 30.0f ) * fClamped * fClamped;
}

// classes to use for template metaprogramming stuff
class SmoothStepSummingFunction
{

public:

    static float Evaluate( const float fSum )
    {
        return SmoothStep( fSum );
    }
    
    static float Derivative( const float fSum )
    {
        return SmoothStepDerivative( fSum );
    }

};

class SmootherStepSummingFunction
{

public:

    static float Evaluate( const float fSum )
    {
        return SmootherStep( fSum );
    }
    
    static float Derivative( const float fSum )
    {
        return SmootherStepDerivative( fSum );
    }

};

}

#endif
