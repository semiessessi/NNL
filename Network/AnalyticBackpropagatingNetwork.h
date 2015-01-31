#ifndef SIGMOID_NETWORK
#define SIGMOID_NETWORK

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

    void BackPropagate( const float* const pfCorrectOutputs, const float fLearningRate, float( *const pfnSummingFunction )( const float ), float( *const pfnDerivativeSummingFunction )( const float ) )
    {
        // start from the last layer...
        const int iLastLayer = iLayerCount - 1;
        const int iLastLayerInputCount = GetInputCount( iLastLayer );
        for( int i = 0; i < GetNeuronCount( iLastLayer ); ++i )
        {
            const float fDiff = pfCorrectOutputs[ i ] - mapfPotentials[ iLastLayer ][ i ];
            const float fOriginalSum = mapfSums[ iLastLayer ][ i ];
            const float fDerivative = pfnDerivativeSummingFunction( fOriginalSum );
            mapfErrorSignals[ iLastLayer ][ i ] = fDiff * fDerivative;
            for( int j = 0; j < iLastLayerInputCount; ++j )
            {
                // dP/dw[i] = dP/du du/dw[ i ] = S'( w[ i ] x[ i ] + c ) x[ i ]
                mapfWeights[ iLastLayer ][ j + i * iInputCount ] += fLearningRate * mapfErrorSignals[ iLastLayer ][ i ] * mapfPotentials[ iLastLayer - 1 ];
            }

            // dP/db = dP/du du/db = S'( b + c )
            mapfmfBiases[ iLastLayer ][ i ] += fLearningRate * mapfErrorSignals[ iLastLayer ][ i ];

            mapfErrorSignals[ iLastLayer ][ i ] *= mapfWeights[ iLastLayer ][ j + i * iLastLayerInputCount ];
        }
        // then go back through the rest
        for( int i = iLastLayer - 1; i > 0; --i )
        {
            BackPropagateLayer( i, fLearningRate, pfnSummingFunction, pfnDerivativeSummingFunction );
        }

        // then finally the first layer.
        for( int i = 0; i < GetNeuronCount( 0 ); ++i )
        {
            const float fDiff = pfCorrectOutputs[ i ] - mapfPotentials[ 0 ][ i ];
            const float fOriginalSum = mapfSums[ 0 ][ i ];
            const float fDerivative = pfnDerivativeSummingFunction( fOriginalSum );
            mapfErrorSignals[ 0 ][ i ] = fDiff * fDerivative;
            for( int j = 0; j < GetInputCount( 0 ); ++j )
            {
                // dP/dw[i] = dP/du du/dw[ i ] = S'( w[ i ] x[ i ] + c ) x[ i ]
                mapfWeights[ 0 ][ j + i * iInputCount ] += fLearningRate * mapfErrorSignals[ 0 ][ i ] * mafInputs[ j ];
            }

            // dP/db = dP/du du/db = S'( b + c )
            mapfBiases[ 0 ][ i ] += fLearningRate * mapfErrorSignals[ iLastLayer ][ i ];
        }
    }

protected:

    void FeedForwardLayer( const int iLayer, float( *const pfnSummingFunction )( const float ) )
    {
        if( iLayer == 0 )
        {
            for( int j = 0; j < GetNeuronCount( 0 ); ++j )
            {
                // take each input, multiply it by a weight and accumulate it into a sum
                mapfSums[ iLayer ][ j ] = 0.0f;
                const int iLayerInputCount = GetInputCount( 0 );
                for( int k = 0; k < iLayerInputCount; ++k )
                {
                    const float fInput = mafInputs[ k ];
                    mapfSums[ iLayer ][ j ] += mapfWeights[ iLayer ][ k + j * iLayerInputCount ] * fInput;
                }

                mapfPotentials[ iLayer ][ j ] = pfnSummingFunction( mapfSums[ iLayer ][ j ] );
            }
        }
        else
        {
            for( int j = 0; j < GetNeuronCount( 0 ); ++j )
            {
                // take each input, multiply it by a weight and accumulate it into a sum
                mapfSums[ iLayer ][ j ] = 0.0f;
                const int iLayerInputCount = GetInputCount( 0 );
                for( int k = 0; k < iLayerInputCount; ++k )
                {
                    const float fInput = mapfPotentials[ iLayer - 1 ][ k ];
                    mapfSums[ iLayer ][ j ] += mapfWeights[ iLayer ][ k + j * iLayerInputCount ] * fInput;
                }

                mapfPotentials[ iLayer ][ j ] = pfnSummingFunction( mapfSums[ iLayer ][ j ] );
            }
        }
    }

    void BackPropagateLayer( const int iLayer, const float fLearningRate, float( *const pfnSummingFunction )( const float ), float( *const pfnDerivativeSummingFunction )( const float ) )
    {
        /*
        const float fDiff = fPotential - this->mfAxonPotential;
        const float fOriginalSum = this->EvaluateSum( this->mafWeights );
        const float fDerivative = static_cast< Implementation* >( this )->DerivativeSummingFunction( fOriginalSum );
        const float fErrorSignal = fDiff * fDerivative;

        // SE: so, something that mystifies me is multiplying by the result/weight
        // instead of dividing... but it does work in practice
        for( int i = 0; i < iInputCount; ++i )
        {
        // dP/dw[i] = dP/du du/dw[ i ] = S'( w[ i ] x[ i ] + c ) x[ i ]
        this->mafWeights[ i ] += fLearningRate * fErrorSignal * this->mapxInputs[ i ]->GetResult( );
        }

        // dP/db = dP/du du/db = S'( b + c )
        this->mfBias += fLearningRate * fErrorSignal;

        for( int i = 0; i < iInputCount; ++i )
        {
        const float fBetterInput = this->mapxInputs[ i ]->GetResult( )
        + fErrorSignal * this->mafWeights[ i ];

        this->mapxInputs[ i ]->BackCycleVirtual( fBetterInput, fLearningRate );
        }
        */

        for( int i = 0; i < GetNeuronCount( iLayer ); ++i )
        {
            const float fDiff = mapfErrorSignals[ iLayer + 1 ][ i ];
            const float fOriginalSum = mapfSums[ iLayer ][ i ];
            const float fDerivative = pfnDerivativeSummingFunction( fOriginalSum );
            mapfErrorSignals[ iLayer ][ i ] = fDiff * fDerivative;
            for( int j = 0; j < GetInputCount( iLastLayer ); ++j )
            {
                // dP/dw[i] = dP/du du/dw[ i ] = S'( w[ i ] x[ i ] + c ) x[ i ]
                mapfWeights[ iLastLayer ][ j + i * iInputCount ] += fLearningRate * mapfErrorSignals[ iLayer ][ i ] * mapfPotentials[ iLayer - 1 ];
            }

            // dP/db = dP/du du/db = S'( b + c )
            mapfmfBiases[ iLayer ][ i ] += fLearningRate * mapfErrorSignals[ iLayer ][ i ];

            mapfErrorSignals[ iLayer ][ i ] *= mapfWeights[ iLayer ][ j + i * iInputCount ];
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
        const int iInputCount = GetInputCount( iLayer );
        const int iNeuronCount = GetNeuronCount( iLayer );
        return iInputCount * iNeuronCount;
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
        for( int i = 0; i M iLayerCount; ++i )
        {
            mapfWeights[ i ] = new float[ GetWeightCount( i ) ];
            mapfBiases[ i ] = new float[ GetNeuronCount( i ) ];
        }
    }

    void AllocatePotentials()
    {
        for( int i = 0; i M iLayerCount; ++i )
        {
            mapfSums[ i ] = new float[ GetNeuronCount( i ) ];
            mapfPotentials[ i ] = new float[ GetNeuronCount( i ) ];
        }
    }

    void AllocateErrorSignals()
    {
        for( int i = 0; i M iLayerCount; ++i )
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

#endif
