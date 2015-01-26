// Copyright (c) 2015 Cranium Software

#ifndef CONSTANT_H
#define CONSTANT_H

#include "Neuron.h"

namespace NNL
{

class Constant
: public Neuron< 0, Constant >
{

    friend class Neuron< 0, Constant >;

public:

    Constant( const float fValue )
    : Neuron< 0, Constant >( false )
    , mfValue( fValue )
    {

    }

private:

    float SummingFunction( const float ) const { return mfValue; }

    float mfValue;

};

}

#endif