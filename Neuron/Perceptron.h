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

    void BackPropagate( const float fPotential, const float fLearningRate )
    {
        this->LinearBackPropagator( fPotential, fLearningRate );
    }

};

}

#endif
