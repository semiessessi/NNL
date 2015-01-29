// Copyright (c) 2015 Cranium Software

#ifndef LAYER_H
#define LAYER_H

#include <vector>

#include "Neuron/Neuron.h"

namespace NNL
{

class Layer
{

public:

    Layer()
    {

    }

    void AddNeuron( NeuronBase& xNeuron ) { mapxNeurons.push_back( &xNeuron ); }
    
    template< class NeuronType >
    void AddNeurons( NeuronType* const pxNeurons, const int iCount )
    {
        for( int i = 0; i < iCount; ++i )
        {
            AddNeuron( pxNeurons[ i ] );
        }
    }

    void Cycle()
    {
        for( int i = 0; i < static_cast< int >( mapxNeurons.size() ); ++i )
        {
            mapxNeurons[ i ]->CycleVirtual();
        }
    }

    void BackCycle( const float fPotential, const float fLearningRate )
    {
        for( int i = 0; i < static_cast< int >( mapxNeurons.size() ); ++i )
        {
            mapxNeurons[ i ]->BackCycleVirtual( fPotential, fLearningRate );
        }
    }

    NeuronBase* GetNeuron( const int iIndex ) { return mapxNeurons[ iIndex ]; }
    int GetNeuronCount() const { return static_cast< int >( mapxNeurons.size() ); }

private:

    std::vector< NeuronBase* > mapxNeurons;

};

}

#endif
