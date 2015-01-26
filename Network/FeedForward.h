// Copyright (c) 2015 Cranium Software

#ifndef FEED_FORWARD_NETWORK_H
#define FEED_FORWARD_NETWORK_H

#include <vector>

#include "Layer/Layer.h"

namespace NNL
{

class FeedForwardNetwork
{

public:

    FeedForwardNetwork()
    {

    }

    void AddLayer( Layer& xLayer )
    {
        //if( mapxLayers.size() > 0 )
        //{
            //Layer& xPrevious = *( mapxLayers.back() );
            // connect all the neurons to each other
            //for( int i = 0;  )
        //}
        mapxLayers.push_back( &xLayer );
    }

    void Cycle()
    {
        for( int i = 0; i < static_cast< int >( mapxLayers.size() ); ++i )
        {
            mapxLayers[ i ]->Cycle();
        }
    }

    void BackCycle( const float fResult, const float fLearningRate )
    {
        for( int i = 0; i < static_cast< int >( mapxLayers.size() ); ++i )
        {
            mapxLayers[ i ]->BackCycle( fResult, fLearningRate );
        }
    }

private:

    std::vector< Layer* > mapxLayers;

};

}

#endif
