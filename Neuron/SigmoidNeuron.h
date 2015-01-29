// Copyright (c) 2015 Cranium Software

#ifndef SIGMOID_NEURON_H
#define SIGMOID_NEURON_H

#include "AnalyticBackPropagator.h"

#include "Maths/Sigmoid.h"

namespace NNL
{

template< int iInputCount >
class SigmoidNeuron
: public AnalyticBackPropagator< iInputCount, SigmoidNeuron< iInputCount > >
{
    friend class Neuron< iInputCount, SigmoidNeuron< iInputCount > >;
    friend class AnalyticBackPropagator< iInputCount, SigmoidNeuron< iInputCount > >;

public:

private:

    static float SummingFunction( const float fSum )
    {
        return 2.0f * Sigmoid( fSum ) - 1.0f;
    }

    static float DerivativeSummingFunction( const float fValue )
    {
        return 2.0f * SigmoidDerivative( fValue );
    }

};

}

#endif