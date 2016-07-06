// Copyright (c) 2016 Cranium Software

#ifndef NNL_OPTIMISED_NETWORK_H
#define NNL_OPTIMISED_NETWORK_H

#include <thread>

namespace NNL
{

template< typename SummingFunctionClass, int iInputCount, int iNeuronCount >
struct LayerDescriptor
{
    static const int kiNeuronCount = iNeuronCount;
    static const int kiInputCount = iInputCount;
    static const int kiTotalInputCount = iInputCount * iNeuronCount;
    typedef SummingFunctionClass SummingFunction;
};

template< typename Class >
struct IsLayerDescriptor
{
    static const bool Result = false;
};

template< typename SummingFunctionClass, int iInputCount, int iNeuronCount >
struct IsLayerDescriptor< LayerDescriptor< SummingFunctionClass, iInputCount, iNeuronCount > >
{
    static const bool Result = true;
};

// this will work recursively by using specialised versions of itself
// as its own members, and remove any run-time dynamism (virtuals etc.)
template< typename FirstLayerDescriptor, typename... OtherLayerDescriptors >
class OptimisedNetwork
{

public:

    void Cycle( const float* const pfInputs )
    {
        mxFirstLayerNetwork.Cycle( pfInputs );
        const float* const pfFirstLayerOutputs = mxFirstLayerNetwork.GetOutputValues();
        mxRemainingLayersNetwork.Cycle( pfFirstLayerOutputs );
    }

    void PrepareBackCycle( const float* const pfInputs )
    {
        mxFirstLayerNetwork.SetInputValues( pfInputs );
        mxRemainingLayersNetwork.PrepareBackCycle( mxFirstLayerNetwork.GetOutputValues() );
    }

    void BackCycle( const float* const pfExpectedOutputs, const float fLearningRate )
    {
        mxRemainingLayersNetwork.BackCycle( pfExpectedOutputs, fLearningRate );
        mxFirstLayerNetwork.BackCycle( mxRemainingLayersNetwork.GetInputValues(), fLearningRate );
    }
    
    const float* GetOutputValues() const
    {
        return mxRemainingLayersNetwork.GetOutputValues();
    }

    float* GetInputValues( const int iLayer = 0 )
    {
        if( iLayer == 0 )
        {
            return mxFirstLayerNetwork.GetInputValues();
        }

        return mxRemainingLayersNetwork.GetInputValues( iLayer - 1 );
    }

    void SetInputValues( const float* const pfSourceValues, const int iLayer = 0 )
    {
        if( iLayer == 0 )
        {
            mxFirstLayerNetwork.SetInputValues( pfSourceValues );
            return;
        }

        mxRemainingLayersNetwork.SetInputValues( pfSourceValues, iLayer - 1 );
    }
    
    // SE - TODO: will this optimise or does it require the template metaprogramming treatment?
    float* GetWeights( const int iLayer = 0 )
    {
        if( iLayer == 0 )
        {
            return mxFirstLayerNetwork.GetWeights();
        }
        
        return mxRemainingLayersNetwork.GetWeights( iLayer - 1 );
    }

    float* GetBiases( const int iLayer = 0 )
    {
        if( iLayer == 0 )
        {
            return mxFirstLayerNetwork.GetBiases();
        }

        return mxRemainingLayersNetwork.GetBiases( iLayer - 1 );
    }

private:

    // template recursion to build all the arrays etc.
    OptimisedNetwork< FirstLayerDescriptor > mxFirstLayerNetwork;
    OptimisedNetwork< OtherLayerDescriptors... >  mxRemainingLayersNetwork;

};

// hard code the case for one layer to be used recursively...
template< typename SummingFunctionClass, int iInputCount, int iNeuronCount >
class OptimisedNetwork< LayerDescriptor< SummingFunctionClass, iInputCount, iNeuronCount > >
{

    typedef OptimisedNetwork< LayerDescriptor< SummingFunctionClass, iInputCount, iNeuronCount > > Network;
    typedef LayerDescriptor< SummingFunctionClass, iInputCount, iNeuronCount > Layer;

    static const int kiThreadCount = 8;

public:

    void Cycle( const float* const pfInputs )
    {
        // SE - TODO: thread switch over magic number...
        //if( Layer::kiNeuronCount <= 32 )
        {
            CycleSequential( pfInputs );
        }
        /*
        else
        {
            CycleThreaded( pfInputs );
        }
        */
    }

    void PrepareBackCycle( const float* const pfInputs )
    {
        // SE - TODO: thread even this copy?
        SetInputValues( pfInputs );
    }

    void BackCycle( const float* const pfExpectedOutputs, const float fLearningRate )
    {
        // SE - TODO: thread switch over magic number...
        //if( Layer::kiNeuronCount <= 32 )
        {
            BackCycleSequential( pfExpectedOutputs, fLearningRate );
        }
        /*
        else
        {
            BackCycleThreaded( pfExpectedOutputs, fLearningRate );
        }
        */
    }

    const float* GetOutputValues() const { return mafAxonValues; }

    float* GetInputValues( const int iLayer = 0 )
    {
        if( iLayer == 0 )
        {
            return mafInputValues;
        }

        return mxRemainingLayersNetwork.GetInputValues( iLayer - 1 );
    }

    void SetInputValues( const float* const pfSourceValues, const int iLayer = 0 )
    {
        if( iLayer == 0 )
        {
            for( int i = 0; i < Layer::kiInputCount; ++i )
            {
                mafInputValues[ i ] = pfSourceValues[ i ];
            }
        }
    }

    float* GetWeights( const int iLayer = 0 )
    {
        if( iLayer == 0 )
        {
            return mafWeights;
        }

        return 0;
    }

    float* GetBiases( const int iLayer = 0 )
    {
        if( iLayer == 0 )
        {
            return mafBiases;
        }

        return 0;
    }

private:

    void CycleSequential( const float* const pfInputs )
    {
        // SE - TODO: split loop into chunks for threads
        // SE - TODO: SIMD optimisation for loop.

        // for each neuron
        int iWeightIndex = 0;
        for( int iNeuronIndex = 0; iNeuronIndex < iNeuronCount; ++iNeuronIndex )
        {
            // create a sum, adding with each weight, each input
            float fSum = 0.0f; // add bias last for precision reeasons.
            for( int iInputIndex = 0; iInputIndex < Layer::kiInputCount; ++iInputIndex )
            {
                fSum += pfInputs[ iInputIndex ] * mafWeights[ iWeightIndex ];
                ++iWeightIndex;
            }
            
            fSum += mafBiases[ iNeuronIndex ];
            
            mafAxonValues[ iNeuronIndex ] = Layer::SummingFunction::Evaluate( fSum );
        }
    }

    static void ThreadFunction( Network* const pxNetwork, const float* const pfInputs, const int iSegment )
    {
        pxNetwork->CycleThreadedSegment( pfInputs, iSegment );
    }

    void CycleThreaded( const float* const pfInputs )
    {
        std::thread sapxThreadPool[ kiThreadCount ];

        for( int i = 0; i < kiThreadCount; ++i )
        {
            new ( axThreadPool + 1 ) std::thread( ThreadFunction, this, pfInputs, i );
        }

        for( int i = 0; i < kiThreadCount; ++i )
        {
            axThreadPool[ i ].join();
        }
    }

    void CycleThreadedSegment( const float* const pfInputs, const int iSegment )
    {
        // SE - TODO: split loop into chunks for threads
        // SE - TODO: SIMD optimisation for loop.

        // for each neuron
        int iWeightIndex = 0;
        for( int iNeuronIndex = iSegment; iNeuronIndex < iNeuronCount; iNeuronIndex += kiThreadCount )
        {
            // create a sum, adding with each weight, each input
            float fSum = 0.0f; // add bias last for precision reeasons.
            for( int iInputIndex = 0; iInputIndex < Layer::kiInputCount; ++iInputIndex )
            {
                fSum += pfInputs[ iInputIndex ] * mafWeights[ iWeightIndex ];
                ++iWeightIndex;
            }

            fSum += mafBiases[ iNeuronIndex ];

            mafAxonValues[ iNeuronIndex ] = Layer::SummingFunction::Evaluate( fSum );
        }
    }

    void BackCycleSequential( const float* const pfExpectedOutputs, const float fLearningRate )
    {
        // for each neuron
        int iWeightOffset = 0;
        for( int iNeuronIndex = 0; iNeuronIndex < iNeuronCount; ++iNeuronIndex )
        {
            // for each output, back propogate...
            const float fScaledError = fLearningRate * ( pfExpectedOutputs[ iNeuronIndex ] - mafAxonValues[ iNeuronIndex ] );

            for( int iInputIndex = 0; iInputIndex < Layer::kiInputCount; ++iInputIndex )
            {
                
            }

            iWeightOffset += Layer::kiInputCount
        }
    }
    
private:

    float mafWeights[ Layer::kiTotalInputCount ];
    float mafBiases[ Layer::kiNeuronCount ];
    float mafAxonValues[ Layer::kiNeuronCount ];
    float mafInputValues[ Layer::kiInputCount ];
    
};

// SE - TODO: ... other types? for something recurrent?

}

#endif
