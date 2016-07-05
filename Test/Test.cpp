#include <cstdio>

#include "FastNetworkTest.h"
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

        puts( "Success!" );

        return 0;
    }
    while( false );

    puts( "Error: Test failed!" );
    return -1;
}
