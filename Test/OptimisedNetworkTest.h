#ifndef NNL_OPTIMISED_NETWORK_TEST_H
#define NNL_OPTIMISED_NETWORK_TEST_H

#include "../Maths/Random.h"
#include "../Maths/Sigmoid.h"
#include "../Network/OptimisedNetwork.h"
#include "../Neuron/Input.h"
#include "../Neuron/SigmoidNeuron.h"

#include <cstdio>

bool OptimisedNetworkTest()
{
    using namespace NNL;
    
    OptimisedNetwork< LayerDescriptor< SigmoidSummingFunction, 2, 1 > > xSingleNeuronNetwork;
    
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
            float* const pfNetworkWeights = xSingleNeuronNetwork.GetWrights( 0 );
            
            pfSingleWeights[ 0 ] = pfNetworkWeights[ 0 ] = x;
            pfSingleWeights[ 1 ] = pfNetworkWeights[ 1 ] = y;
            
            const float fTestInputs[] = { WeakRandom(), WeakRandom() };
            for( int i = 0; i < 2; ++i )
            {
                afSingleNeuronInputs[ i ] = fTestInputs[ i ];
            }
            
            
        }
    }
    
    return true;
}

#endif
