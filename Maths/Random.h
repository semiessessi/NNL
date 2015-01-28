// Copyright (c) 2015 Cranium Software

#ifndef RANDOM_H
#define RANDOM_H

#include <ctime>

namespace NNL
{

static inline unsigned long long WeakRandomInt()
{
    static unsigned long long lsaullSeeds[ 2 ] =
    {
        static_cast< unsigned long long >( time( 0 ) ),
        static_cast< unsigned long long >( time( 0 ) ^ 516527181 )
    };
    static unsigned long long ullXorShift = lsaullSeeds[ 0 ];
    static const unsigned long long kullInitialSeed = lsaullSeeds[ 1 ];
    lsaullSeeds[ 0 ] = kullInitialSeed;
    ullXorShift ^= ullXorShift << 23;
    ullXorShift ^= ullXorShift >> 17;
    ullXorShift ^= kullInitialSeed ^ ( kullInitialSeed >> 26 );
    lsaullSeeds[ 1 ] = ullXorShift;
    return ullXorShift + kullInitialSeed;
}
    
static inline float WeakRandom( const float fMin = -1.0f, const float fMax = 1.0f )
{
    return fMin + ( fMax - fMin ) * static_cast< float >( static_cast< double >( WeakRandomInt() ) / static_cast< double >( ~0ULL ) );
}

}

#endif
