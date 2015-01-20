// Copyright (c) 2015 Cranium Software

#ifndef HEAVISIDE_STEP_H
#define HEAVISIDE_STEP_H

namespace NNL
{

static inline float HeavisideStep( const float fX )
{
    return ( fX > 0.0f ) ? 1.0f : 0.0f;
}

}

#endif
