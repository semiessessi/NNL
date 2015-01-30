// Copyright (c) 2015 Cranium Software

#ifndef LINEAR_NEURON_H
#define LINEAR_NEURON_H

#include "AnalyticBackPropagator.h"

namespace NNL
{

// SE - TODO: use analytic back propagation...
// ... also atm this is rubbish.
template< int iInputCount >
class LinearNeuron
: public Neuron< iInputCount, LinearNeuron< iInputCount > >
//: public AnalyticBackPropagator< iInputCount, LinearNeuron< iInputCount > >
{
    friend class Neuron< iInputCount, LinearNeuron< iInputCount > >;
    //friend class AnalyticBackPropagator< iInputCount, LinearNeuron< iInputCount > >;

public:

    void BackPropagate( const float fPotential, const float fLearningRate )
    {
        this->LinearBackPropagator( fPotential, fLearningRate );
    }

};

}

#endif