// Copyright (c) 2016 Cranium Software

#include <cstdio>

#include "OptimisedNetworkTest.h"
#include "MathTest.h"

int main( const int, const char* const* const )
{
    puts( "NNL Tests..." );

    do
    {
        if( !CheckSummingFunctionRange() )
        {
            break;
        }
        
        if( !OptimisedNetworkTest() )
        {
            break;
        }

        puts( "Success!" );

        return 0;
    }
    while( false );

    puts( "Error: Test failed!" );
    return -1;
}
