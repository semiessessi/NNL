// Copyright (c) 2016 Cranium Software

#ifndef NNL_OPTIMISED_NETWORK_TEST_H
#define NNL_OPTIMISED_NETWORK_TEST_H

#include "Maths/Random.h"
#include "Maths/Sigmoid.h"
#include "Network/OptimisedNetwork.h"
#include "Neuron/Input.h"
#include "Neuron/SigmoidNeuron.h"

#include <cmath>
#include <cstdio>

bool OptimisedNetworkTest()
{
    using namespace NNL;
    
    OptimisedNetwork< LayerDescriptor< SigmoidSummingFunction, 2, 1 > > xNetwork;
    
    // test against single sigmoid neuron
    // SE - TODO: refactor all the 2.0f * fFoo - 1.0f out
    float afSingleNeuronInputs[ 2 ] = { 0.0f, 0.0f };
    SigmoidNeuron< 2 > xSigmoid;
    Input xInput0( afSingleNeuronInputs );
    Input xInput1( afSingleNeuronInputs + 1 );
    xSigmoid.SetInputVirtual( 0, &xInput0 );
    xSigmoid.SetInputVirtual( 1, &xInput1 );
    
    for( float x = -100.0f; x <= 100.0f; x += 0.1f )
    {
        for( float y = -100.0f; y <= 100.0f; y += 0.1f )
        {
            float* const pfSingleWeights = xSigmoid.GetWeightPointer();
            float* const pfNetworkWeights = xNetwork.GetWeights( 0 );
            
            pfSingleWeights[ 0 ] = pfNetworkWeights[ 0 ] = x;
            pfSingleWeights[ 1 ] = pfNetworkWeights[ 1 ] = y;
            
            const float fTestInputs[] = { WeakRandom(), WeakRandom() };
            for( int i = 0; i < 2; ++i )
            {
                afSingleNeuronInputs[ i ] = fTestInputs[ i ];
            }
            
            xInput0.Cycle();
            xInput1.Cycle();

            const float fTestBias = WeakRandom();
            xSigmoid.SetBias( fTestBias );
            xNetwork.GetBiases( 0 )[ 0 ] = fTestBias;

            xSigmoid.Cycle();
            xNetwork.Cycle( afSingleNeuronInputs );

            const float fNeuronResult = xSigmoid.GetResult();
            const float fNetworkResult = xNetwork.GetOutputValues()[ 0 ];
            const float fError = std::abs( fNeuronResult - fNetworkResult );
            if( fError > 0.005f )
            {
                printf( "Error: test failed since error value was too great (%f)\n", fError );
                return false;
            }
        }
    }
    
    puts( "Optimised network passed most basic of tests - one layer, one neuron, forward cycles." );
    
    // test back cycling
    for( float fTestValue = -100.0f; fTestValue <= 100.0f; fTestValue += 0.1f )
    {
        for( float fTestRate = 0.0005f; fTestRate < 1.0f; fTestRate *= 2.5f )
        {
            xSigmoid.BackCycleVirtual( fTestValue, fTestRate );
            xNetwork.BackCycle( &fTestValue, fTestRate );
            
            for( int i = 0; i < 2; ++i )
            {
                const float fNeuronWeight = xSigmoid.GetWeightPointer()[ i ];
                const float fNetworkWeight = xNetwork.GetWeights( 0 )[ i ];
                const float fError = std::abs( fNeuronWeight - fNetworkWeight );
                if( fError > 0.05f )
                {
                    printf( "Error: test failed since error value was too great (%f)\n", fError );
                    return false;
                }
            }
        }
    }

    puts( "Optimised network passed most basic of tests - one layer, one neuron, backward cycles." );
    
    return true;
}

#endif
