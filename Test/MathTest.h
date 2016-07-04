#ifndef NNL_MATH_TEST_H
#define NNL_MATH_TEST_H

#include "../Maths/HeavisideStep.h"
#include "../Maths/Sigmoid.h"
#include "../Maths/SmoothSteps.h"

#include <cstdio>

template< typename Functor >
static bool CheckSummingFunctionRangeTest( const Functor& xSummingFunction, const char* const szName )
{
    for( float f = -1000.0f; f < 1000.0f; f += 0.1f )
    {
        const float fResult = xSummingFunction( f );
        if( ( fResult < -0.0f ) || ( fResult > 1.0f ) )
        {
            printf( "Result %f for %s( %f ) is out of range\n", fResult, szName, f );
            return false;
        }
    }

    printf( "%s passed range check\n", szName );
    return true;
}

bool CheckSummingFunctionRange()
{
    using namespace NNL;

    if( !CheckSummingFunctionRangeTest( HeavisideStep, "Heaviside Step" ) )
    {
        return false;
    }

    if( !CheckSummingFunctionRangeTest( SmoothStep, "Smooth Step" ) )
    {
        return false;
    }

    if( !CheckSummingFunctionRangeTest( SmootherStep, "Smoother Step" ) )
    {
        return false;
    }

    if( !CheckSummingFunctionRangeTest( Sigmoid, "Sigmoid" ) )
    {
        return false;
    }
    
    return true;
}

#endif
