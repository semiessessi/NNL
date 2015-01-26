// Copyright (c) 2015 Cranium Software

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "Neuron.h"

#include "Maths/HeavisideStep.h"

namespace NNL
{
template< int iInputCount >
class Perceptron
: public Neuron< iInputCount, Perceptron< iInputCount > >
{

    friend class Neuron< iInputCount, Perceptron< iInputCount > >;

public:

    Perceptron()
    : Neuron< iInputCount, Perceptron< iInputCount > >( )
    {

    }

private:

    static float SummingFunction( const float fSum ) { return HeavisideStep( fSum ); }

    void BackPropogate( const float fPotential, const float fLearningRate )
    {
        // adjust weights
        for( int i = 0; i < iInputCount; ++i )
        {
            mafWeights[ i ] += ( fPotential - mfAxonPotential ) * fLearningRate
                * ( 1.0f / ( static_cast< float >( iInputCount ) * *( mapfInputs[ i ] ) ) );
        }

        // adjust bias
        mfBias += ( fPotential - mfAxonPotential ) * fLearningRate
            * ( 1.0f / static_cast< float >( iInputCount ) );
    }
};

}

#endif
