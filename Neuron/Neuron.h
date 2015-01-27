// Copyright (c) 2015 Cranium Software

#ifndef NEURON_H
#define NEURON_H

namespace NNL
{

class NeuronBase
{

public:

    virtual void CycleVirtual() = 0;
    virtual void BackCycleVirtual( const float, const float ) = 0;
    virtual void SetInputVirtual( const int iIndex, NeuronBase* const pxInput ) = 0;

    float& GetResult() { return mfAxonPotential; }

protected:

    void BackPropogate( const float /*fPotential*/, const float /*fLearningRate*/ ) const {}

    //static float SummingFunction( const float fSum ) { return fSum; }
    static float InitialWeight( const int /*iInitialWeight*/ ) { return 0.5f; }
    static float InitialBias() { return 0.0f; }

    float mfAxonPotential;

};

template<
    int iInputCount,
    class Implementation >
class Neuron
: public NeuronBase
{

public:

    Neuron( const bool bInitialise = true )
    {
        if( bInitialise )
        {
            for( int i = 0; i < iInputCount; ++i )
            {
                mapxInputs[ i ] = 0;
                mafWeights[ i ] = Implementation::InitialWeight( i );
            }

            mfBias = Implementation::InitialBias();
        }
    }

    template< int iOtherInputCount, class OtherImplementation >
    void ConnectAtIndex( const int iIndex, Neuron< iOtherInputCount, OtherImplementation >& xInputNeuron )
    {
        mapxInputs[ iIndex ] = &xInputNeuron;
    }

    virtual void CycleVirtual() { Cycle(); }
    virtual void BackCycleVirtual( const float fPotential, const float fLearningRate ) { BackCycle( fPotential, fLearningRate ); }
    virtual void SetInputVirtual( const int iIndex, NeuronBase* const pxInput ) { mapxInputs[ iIndex ] = pxInput; }

    void Cycle()
    {
        float fSum = mfBias;
        for( int i = 0; i < iInputCount; ++i )
        {
            if( mapxInputs[ i ] )
            {
                fSum += mapxInputs[ i ]->GetResult() * mafWeights[ i ];
            }
        }

        mfAxonPotential = static_cast< Implementation* >( this )->SummingFunction( fSum );
    }

    void BackCycle( const float fPotential, const float fLearningRate )
    {
        static_cast< Implementation* >( this )->BackPropogate( fPotential, fLearningRate );
    }

protected:

    NeuronBase* mapxInputs[ iInputCount ? iInputCount : 1 ];
    float mafWeights[ iInputCount ? iInputCount : 1 ];
    float mfBias;

};

}

#endif
