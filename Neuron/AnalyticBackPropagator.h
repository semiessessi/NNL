// Copyright (c) 2015 Cranium Software

#ifndef ANALYTIC_BACK_PROPOGATOR_H
#define ANALYTIC_BACK_PROPOGATOR_H

#include "Neuron.h"

#include "Maths/HeavisideStep.h"

namespace NNL
{

template< int iInputCount, class Implementation >
class AnalyticBackPropagator
: public Neuron< iInputCount, Implementation >
{

    friend class Neuron< iInputCount, Implementation >;

public:

    AnalyticBackPropagator()
    : Neuron< iInputCount, Implementation >( )
    {

    }

private:

    static float SummingFunction( const float fSum ) { return fSum; }
    static float DerivativeSummingFunction( const float fSum ) { return 1.0f; }

    void BackPropagate( const float fPotential, const float fLearningRate )
    {
        const float fDiff = fPotential - mfAxonPotential;
        for( int i = 0; i < iInputCount; ++i )
        {
            // dP/dw[i] = dP/du du/dw[ i ] = S'( w[ i ] x[ i ] + c ) x[ i ]
            mafWeights[ i ] -= fLearningRate * static_cast< Implementation* >( this )->DerivativeSummingFunction( fPotential ) * mapxInputs[ i ]->GetResult();
        }

        // dP/db = dP/du du/db = S'( b + c )
        mfBias -= fLearningRate * static_cast< Implementation* >( this )->DerivativeSummingFunction( fPotential );

        for( int i = 0; i < iInputCount; ++i )
        {
            const float fBetterInput = mapxInputs[ i ]->GetResult() + fDiff / mafWeights[ i ];
            
            mapxInputs[ i ]->BackCycleVirtual( fBetterInput, fLearningRate );
        }
    }

};

}

#endif
