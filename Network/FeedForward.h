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

    ~FeedForwardNetwork()
    {
        DeleteOwnedLayers();
    }

    void AddLayer( Layer& xLayer );

    void Load();
    void Save();

    void Cycle()
    {
        for( int i = 0; i < static_cast< int >( mapxLayers.size() ); ++i )
        {
            mapxLayers[ i ]->Cycle();
        }
    }

    void BackCycle( const float fResult, const float fLearningRate )
    {
        mapxLayers.back()->BackCycle( fResult, fLearningRate );
    }

private:

    void DeleteOwnedLayers()
    {
        for( int i = 0; i < static_cast< int >( mapxOwnedLayers.size( ) ); ++i )
        {
            delete mapxOwnedLayers[ i ];
        }

        mapxOwnedLayers.clear();
    }

    std::vector< Layer* > mapxLayers;
    std::vector< Layer* > mapxOwnedLayers;
};

}

#endif
