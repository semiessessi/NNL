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
        const float fDiff = fPotential - this->mfAxonPotential;
        const float fOriginalSum = this->EvaluateSum( this->mafWeights );
        const float fDerivative = static_cast< Implementation* >( this )->DerivativeSummingFunction( fOriginalSum );
        const float fErrorSignal = fDiff * fDerivative;
        
        // SE: so, something that mystifies me is multiplying by the result/weight
        // instead of dividing... but it does work in practice
        for( int i = 0; i < iInputCount; ++i )
        {
            // dP/dw[i] = dP/du du/dw[ i ] = S'( w[ i ] x[ i ] + c ) x[ i ]
            this->mafWeights[ i ] += fLearningRate * fErrorSignal * this->mapxInputs[ i ]->GetResult();
        }

        // dP/db = dP/du du/db = S'( b + c )
        this->mfBias += fLearningRate * fErrorSignal;

        for( int i = 0; i < iInputCount; ++i )
        {
            const float fBetterInput = this->mapxInputs[ i ]->GetResult()
                + fErrorSignal * this->mafWeights[ i ];
            
            this->mapxInputs[ i ]->BackCycleVirtual( fBetterInput, fLearningRate );
        }
    }

};

}

#endif
