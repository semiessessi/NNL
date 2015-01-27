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

    static float SummingFunction( const float fSum ) { return 2.0f * HeavisideStep( fSum ) - 1.0f; }

    void BackPropogate( const float fPotential, const float fLearningRate )
    {
        // adjust weights
        for( int i = 0; i < iInputCount; ++i )
        {
            if( mapxInputs[ i ]->GetResult() != 0.0f )
            {
                mafWeights[ i ] += ( fPotential - mfAxonPotential )
                    * fLearningRate / ( static_cast< float >( iInputCount ) * mapxInputs[ i ]->GetResult() );
            }

            if( mafWeights[ i ] != 0.0f )
            {
                const float fBetterInput = mapxInputs[ i ]->GetResult() + ( fPotential - mfAxonPotential )
                    * fLearningRate / ( static_cast< float >( iInputCount ) * mafWeights[ i ] );
                mapxInputs[ i ]->BackCycleVirtual( fBetterInput, fLearningRate );
            }
        }

        // adjust bias
        mfBias += ( fPotential - mfAxonPotential ) * fLearningRate
            * ( 1.0f / static_cast< float >( iInputCount ) );
    }
};

}

#endif
