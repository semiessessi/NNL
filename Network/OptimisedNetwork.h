// Copyright (c) 2016 Cranium Software

#ifndef NNL_OPTIMISED_NETWORK_H
#define NNL_OPTIMISED_NETWORK_H

namespace NNL
{

template< typename Class >
struct IsLayerDescriptor
{
    static const bool Result = false;
};

template< typename SummingFunctionClass, int iNeuronCount, int iInputCount >
template<>
struct IsLayerDescriptor< LayerDescriptor< SummingFunctionClass, iNeuronCount, iInputCount > >
{
    static const bool Result = true;
};

template< typename SummingFunctionClass, int iNeuronCount, int iInputCount >
struct LayerDescriptor
{
    static const int kiNeuronCount = iNeuronCount;
    static const int kiInputCount = iInputCount;
    static const int kiTotalInputCount = iInputCount * iNeuronCount;
    typedef typename SummingFunctionClass SummingFunction;
};

// this will work recursively by using specialised versions of itself
// as its own members, and remove any run-time dynamism (virtuals etc.)
template< typename FirstLayerDescriptor, typename... OtherLayerDescriptors >
class OptimisedLayeredNetwork
{

public:

    void Cycle( const float* const pfInputs )
    {
        mxFirstLayerNetwork.Cycle( pfInputs );
        const float* const pfFirstLayerOutputs = mxFirstLayerNetwork.GetOutputValues();
        mxRemainingLayersNetwork.Cycle( pfFirstLayerOutputs );
    }
    
    const float* GetOutputValues() const
    {
        return mxRemainingLayersNetwork.GetOutputValues();
    }
    
    // SE - TODO: will this optimise or does it require the template metaprogramming treatment?
    float* GetWeights( const int iLayer )
    {
        if( iLayer == 0 )
        {
            return mxFirstLayerNetwork.GetWeights( iLayer );
        }
        
        return mxRemainingLayersNetwork.GetWeights( iLayer - 1 );
    }

private:

  // template recursion to build all the arrays etc.
  OptimisedNetwork< FirstLayerDescriptor > mxFirstLayerNetwork;
  OptimisedNetwork< OtherLayerDescriptors... >  mxRemainingLayersNetwork;

};

// hard code the case for one layer to be used recursively...
template< typename LayerDescriptor >
class OptimisedLayeredNetwork< LayerDescriptor >
{
  
public:

    void Cycle( const float* const pfInputs )
    {
        // for each neuron
        int iWeightIndex = 0;
        for( int iNeuronIndex = 0; iNeuronIndex < LayerDescriptor::kiNeuronCount; ++iNeuronIndex )
        {
            // create a sum, adding with each weight, each input
            float fSum = 0.0f; // add bias last for precision reeasons.
            for( int iInputIndex = 0; iInputIndex < LayerDescriptor::kiInputCount; ++iInputIndex )
            {
                fSum += pfInputs[ iInputIndex ] * mafWeights[ iWeightIndex ];
                ++iWeightIndex;
            }
            
            fSum += mafBiases[ iNeuronIndex ];
            
            mafAxonValues[ iNeuronIndex ] = LayerDescriptor::SummingFunction::Evaluate( fSum );
        }
    }
    
    const float* GetOutputValues() const { return mafAxonValues; }
    
    float* GetWeights( const int iLayer )
    {
        if( iLayer == 0 )
        {
            return mafWeights;
        }
        
        return 0;
    }
    
private:

    float mafWeights[ LayerDescriptor::kiTotalInputCount ];
    float mafBiases[ LayerDescriptor::kiNeuronCount ];
    float mafAxonValues[ LayerDescriptor::kiNeuronCount ];
    
};

// SE - TODO: ... other types? for something recurrent?

}

#endif
