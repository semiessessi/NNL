// Copyright (c) 2015 Cranium Software

#ifndef INPUT_H
#define INPUT_H

#include "Neuron.h"

namespace NNL
{

class Input
: public Neuron< 0, Input >
{

    friend class Neuron< 0, Input >;

public:

    Input( float* const pfValue = 0 )
    : Neuron< 0, Input >( false )
    , mpfValue( pfValue )
    {

    }
    
    void SetInput( float* const pfValue ) { mpfValue = pfValue; }

private:
    
    void BackPropagate( const float, const float )
    {
        // SE - NOTE: how can we do this?!?! nothing to learn here...
    }

    float SummingFunction( const float ) const { return *mpfValue; }

    float* mpfValue;

};

}

#endif