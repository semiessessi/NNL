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
            miLayerCounts[ i ] = piLayerCounts[ i ];

            AllocateWeights();
            AllocatePotentials();
            AllocateErrorSignals();
        }
    }

    ~AnalyticBackpropagatingNetwork()
    {
        for( int i = 0; i < iLayerCount; ++i )
        {
            delete[] mapfWeights[ i ];
            delete[] mapfPotentials[ i ];
            delete[] mapfErrorSignals[ i ];
            delete[] mapfSums[ i ];
        }
    }

    const int GetLayerCount() const { return iLayerCount; }

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
                    mapfSums[ iLayer ][ j ] += mapfWeights[ iLayer ][ k + j * iInputCount ] * fInput;
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
                    mapfSums[ iLayer ][ j ] += mapfWeights[ iLayer ][ k + j * iInputCount ] * fInput;
                }

                mapfPotentials[ iLayer ][ j ] = pfnSummingFunction( mapfSums[ iLayer ][ j ] );
            }
        }
    }

    void BackPropagateLayer( const int iLayer )
    {

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
    float* mapfPotentials[ iLayerCount ];
    float* mapfErrorSignals[ iLayerCount ];
};

#endif
