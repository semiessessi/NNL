// Copyright (c) 2015 Cranium Software

#include "FeedForward.h"

#include "Neuron/SigmoidNeuron.h"

#include <cstdio>

namespace NNL
{

void FeedForwardNetwork::AddLayer( Layer& xLayer )
{
    if( mapxLayers.size() > 0 )
    {
        Layer& xPrevious = *( mapxLayers.back() );
        // connect all the neurons to each other
        for( int i = 0; i < xLayer.GetNeuronCount(); ++i )
        {
            for( int j = 0; j < xPrevious.GetNeuronCount(); ++j )
            {
                NeuronBase* pxNew = xLayer.GetNeuron( i );
                NeuronBase* pxOld = xPrevious.GetNeuron( j );
                pxNew->SetInputVirtual( j, pxOld );
            }
        }
    }
    mapxLayers.push_back( &xLayer );
}

void FeedForwardNetwork::Load()
{
    printf( "Loading...\r\n" );

    FILE* pxFile = fopen( "data.dat", "rb" );
    if( pxFile )
    {
        const unsigned char aucMagic[] = { 'C', 'N', 'N', 'D' };
        unsigned char aucRead[ 4 ];
        fread( aucRead, 1, 4, pxFile );

        for( int i = 0; i < 4; ++i )
        {
            if( aucMagic[ i ] != aucRead[ i ] )
            {
                fclose( pxFile );
                printf( "Unrecognised file format.\r\n" );
            }
        }

        int iRead = 0;
        fread( &iRead, sizeof( int ), 1, pxFile );
        //  SE - TODO:what type of neuron?
        fread( &iRead, sizeof( int ), 1, pxFile );
        const int iLayerCount = iRead;

        if( iLayerCount != static_cast< int >( mapxLayers.size() ) )
        {
            printf( "Wrong number of layers.\r\n" );
            fclose( pxFile );
            return;
        }

        // read in all of the weights and biases.
        for( int i = 0; i < iLayerCount; ++i )
        {
            const int iNeuronCount = mapxLayers[ i ]->GetNeuronCount();
            for( int j = 0; j < iNeuronCount; ++j )
            {
                const int iInputCount = mapxLayers[ i ]->GetNeuron( j )->GetInputCount();
                fread( mapxLayers[ i ]->GetNeuron( j )->GetWeightPointer( ), sizeof( float ), iInputCount + 1, pxFile );
            }
        }

        fclose( pxFile );
        printf( "Done.\r\n" );
    }
    else
    {
        printf( "Failed to open file.\r\n" );
    }
}

void FeedForwardNetwork::Save()
{
    printf( "Saving...\r\n" );

    FILE* pxFile = fopen( "data.dat", "wb" );
    if( !pxFile )
    {
        printf( "Failed to open file.\r\n" );
        return;
    }

    const unsigned char aucMagic[] = { 'C', 'N', 'N', 'D' };
    fwrite( aucMagic, 1, 4, pxFile );

    // SE - TODO: neuron type
    fwrite( aucMagic, sizeof( int ), 1, pxFile );

    const int iLayerCount = static_cast< int >( mapxLayers.size() );
    fwrite( &iLayerCount, sizeof( int ), 1, pxFile );

    // write out all of the weights and biases.
    for( int i = 0; i < iLayerCount; ++i )
    {
        const int iNeuronCount = mapxLayers[ i ]->GetNeuronCount();
        for( int j = 0; j < iNeuronCount; ++j )
        {
            const int iInputCount = mapxLayers[ i ]->GetNeuron( j )->GetInputCount();
            fwrite( mapxLayers[ i ]->GetNeuron( j )->GetWeightPointer(), sizeof( float ), iInputCount + 1, pxFile );
        }
    }

    fclose( pxFile );

    printf( "Done.\r\n" );
}

}
