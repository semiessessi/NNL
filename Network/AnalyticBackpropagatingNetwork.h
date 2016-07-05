// Copyright (c) 2015 Cranium Software

#ifndef ANALYTIC_BACKPROPAGATING_NETWORK
#define ANALYTIC_BACKPROPAGATING_NETWORK

#include "Maths/Random.h"

namespace NNL
{

template< int iLayerCount, int iInputCount >
class AnalyticBackpropagatingNetwork
{

public:

    AnalyticBackpropagatingNetwork( const int* const piLayerCounts )
    {
        for( int i = 0; i < iLayerCount; ++i )
        {
            maiLayerCounts[ i ] = piLayerCounts[ i ];
        }

        AllocateWeights();
        AllocatePotentials();
        AllocateErrorSignals();
    }

    ~AnalyticBackpropagatingNetwork()
    {
        for( int i = 0; i < iLayerCount; ++i )
        {
            delete[] mapfWeights[ i ];
            delete[] mapfBiases[ i ];
            delete[] mapfPotentials[ i ];
            delete[] mapfErrorSignals[ i ];
            delete[] mapfSums[ i ];
        }
    }

    const int GetLayerCount() const { return iLayerCount; }
    float GetOutput( const int iIndex ) const { return mapfPotentials[ iLayerCount - 1 ][ iIndex ]; }

    void Load( const char* const /*szPath*/ )
    {
        // SE - TODO: ...
    }

    void Save( const char* const /*szPath*/ )
    {
        // SE - TODO: ...
    }

    float* FeedForward( const float* const pfInputs, float ( * const pfnSummingFunction )( const float ) )
    {
        for( int i = 0; i < iInputCount; ++i )
        {
            mafInputs[ i ] = pfInputs[ i ];
        }
        // do the first layer first...
        FeedForwardLayer( 0, pfnSummingFunction );

        // then for each other layer
        for( int i = 1; i < iLayerCount; ++i )
        {
            FeedForwardLayer( i, pfnSummingFunction );
        }

        return mapfPotentials[ iLayerCount - 1 ];
    }

    void BackPropagate( const float* const pfCorrectOutputs, const float fLearningRate, float( *const pfnDerivativeSummingFunction )( const float ) )
    {
        // reset all of the error signals to zero
        const int iLastLayer = iLayerCount - 1;

        for( int i = 0; i < iLayerCount; ++i )
        {
            for( int j = 0; j < GetNeuronCount( i ); ++j )
            {
                mapfErrorSignals[ i ][ j ] = 0.0f;
            }
        }

        // fill out the error signals into the last layer...
        for( int i = 0; i < GetNeuronCount( iLastLayer ); ++i )
        {
            const float fDiff = pfCorrectOutputs[ i ] - mapfPotentials[ iLastLayer ][ i ];
            mapfErrorSignals[ iLastLayer ][ i ] = fDiff;
        }
        
        // then go back through the layers propagating it
        for( int i = iLastLayer; i >= 0; --i )
        {
            BackPropagateLayer( i, fLearningRate, pfnDerivativeSummingFunction );
        }
    }

protected:

    void FeedForwardLayer( const int iLayer, float( *const pfnSummingFunction )( const float ) )
    {
        for( int i = 0; i < GetNeuronCount( iLayer ); ++i )
        {
            // take each input, multiply it by a weight and accumulate it into a sum
            mapfSums[ iLayer ][ i ] = 0.0f;
            const int iLayerInputCount = GetInputCount( iLayer );
            for( int j = 0; j < iLayerInputCount; ++j )
            {
                // SE - NOTE: hopefully this conditional is optimised out
                const float fInput = ( iLayer == 0 ) ? mafInputs[ j ] : mapfPotentials[ iLayer - 1 ][ j ];
                mapfSums[ iLayer ][ i ] += mapfWeights[ iLayer ][ j + i * iLayerInputCount ] * fInput;
            }

            mapfPotentials[ iLayer ][ i ] = pfnSummingFunction( mapfSums[ iLayer ][ i ] );
        }
    }

    void BackPropagateLayer( const int iLayer, const float fLearningRate, float( *const pfnDerivativeSummingFunction )( const float ) )
    {
        for( int i = 0; i < GetNeuronCount( iLayer ); ++i )
        {
            const float fDiff = mapfErrorSignals[ iLayer ][ i ];
            const float fOriginalSum = mapfSums[ iLayer ][ i ];
            const float fDerivative = pfnDerivativeSummingFunction( fOriginalSum );
            const int iLayerInputCount = GetInputCount( iLayer );
            const float fErrorSignal = fDiff * fDerivative;
            for( int j = 0; j < iLayerInputCount; ++j )
            {
                // dP/dw[i] = dP/du du/dw[ i ] = S'( w[ i ] x[ i ] + c ) x[ i ]
                // SE - NOTE: hopefully this conditional is optimised out
                const float fInputPotential = ( iLayer == 0 ) ? mafInputs[ j ] : mapfPotentials[ iLayer - 1 ][ j ];
                mapfWeights[ iLayer ][ j + i * iLayerInputCount ] += fLearningRate * fErrorSignal * fInputPotential;

                // SE - NOTE: hopefully this conditional is optimised out
                if( iLayer != 0 )
                {
                    // accumulate some error signal
                    mapfErrorSignals[ iLayer - 1 ][ j ] += fErrorSignal * mapfWeights[ iLayer ][ j + i * iLayerInputCount ];
                }
            }

            // dP/db = dP/du du/db = S'( b + c )
            mapfBiases[ iLayer ][ i ] += fLearningRate * fErrorSignal;
        }
    }

    int GetCumulativeWeightCount( const int iLayer ) const
    {
        int iSum = GetWeightCount( 0 );
        for( int i = 0; i < iLayer; ++i )
        {
            iSum += GetWeightCount( i + 1 );
        }

        return iSum;
    }

    int GetWeightCount( const int iLayer ) const
    {
        const int iLayerInputCount = GetInputCount( iLayer );
        const int iNeuronCount = GetNeuronCount( iLayer );
        return iLayerInputCount * iNeuronCount;
    }

    int GetNeuronCount( const int iLayer ) const
    {
        return maiLayerCounts[ iLayer ];
    }

    int GetInputCount( const int iLayer ) const
    {
        return ( iLayer == 0 ) ? iInputCount : maiLayerCounts[ iLayer - 1 ];
    }

    void AllocateWeights()
    {
        for( int i = 0; i < iLayerCount; ++i )
        {
            const int iWeightCount = GetWeightCount( i );
            const int iNeuronCount = GetNeuronCount( i );
            mapfWeights[ i ] = new float[ iWeightCount ];
            mapfBiases[ i ] = new float[ iNeuronCount ];
            // initialise values. very important!
            for( int j = 0; j < iWeightCount; ++j )
            {
                mapfWeights[ i ][ j ] = WeakRandom();
            }

            for( int j = 0; j < iNeuronCount; ++j )
            {
                mapfBiases[ i ][ j ] = WeakRandom();
            }
        }
    }

    void AllocatePotentials()
    {
        for( int i = 0; i < iLayerCount; ++i )
        {
            mapfSums[ i ] = new float[ GetNeuronCount( i ) ];
            mapfPotentials[ i ] = new float[ GetNeuronCount( i ) ];
        }
    }

    void AllocateErrorSignals()
    {
        for( int i = 0; i < iLayerCount; ++i )
        {
            mapfErrorSignals[ i ] = new float[ GetNeuronCount( i ) ];
        }
    }

    int maiLayerCounts[ iLayerCount ];
    float mafInputs[ iInputCount ];
    float* mapfWeights[ iLayerCount ];
    float* mapfSums[ iLayerCount ];
    float* mapfBiases[ iLayerCount ];
    float* mapfPotentials[ iLayerCount ];
    float* mapfErrorSignals[ iLayerCount ];
};

}

#endif
