// Copyright (c) 2015 Cranium Software

#ifndef NEURON_H
#define NEURON_H

#include "Maths/Random.h"

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

    void BackPropagate( const float /*fPotential*/, const float /*fLearningRate*/ ) const {}

    static float SummingFunction( const float fSum ) { return fSum; }
    static float InitialWeight( const int /*iInitialWeight*/ ) { return WeakRandom(); }
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
        mfAxonPotential = EvaluateAxon( mafWeights );
    }

    void BackCycle( const float fPotential, const float fLearningRate )
    {
        static_cast< Implementation* >( this )->BackPropagate( fPotential, fLearningRate );
    }

    void BackPropagate( const float fPotential, const float fLearningRate )
    {
        RandomBackPropagator( fPotential, fLearningRate );
    }

protected:

    float EvaluateAxon( const float* pfWeights )
    {
        float fSum = mfBias;
        for( int i = 0; i < iInputCount; ++i )
        {
            if( mapxInputs[ i ] )
            {
                fSum += mapxInputs[ i ]->GetResult() * pfWeights[ i ];
            }
        }

        return static_cast< const Implementation* >( this )->SummingFunction( fSum );
    }

    void RandomBackPropagator( const float fPotential, const float fLearningRate )
    {
        // pick a random set of weights
        float afWeights[ iInputCount ? iInputCount : 1 ];
        for( int i = 0; i < iInputCount; ++i )
        {
            afWeights[ i ] = mafWeights[ i ] + /*fLearningRate **/ WeakRandom();
        }

        // if the work better, keep them
        const float fNewPotential = EvaluateAxon( afWeights );
        const bool bRandomIsBetter = fabsf( fPotential - fNewPotential ) < fabsf( fPotential - mfAxonPotential );
        if( bRandomIsBetter )
        {
            for( int i = 0; i < iInputCount; ++i )
            {
                mafWeights[ i ] = afWeights[ i ];
            }
        }

        const float fMinPotentialDifference = fPotential - ( bRandomIsBetter ? fNewPotential : mfAxonPotential );
        for( int i = 0; i < iInputCount; ++i )
        {
            const float fBetterInput = fMinPotentialDifference / mafWeights[ i ];
            mapxInputs[ i ]->BackCycleVirtual( fBetterInput, fLearningRate );
        }
    }

    void LinearBackPropagator( const float fPotential, const float fLearningRate )
    {
        // adjust weights
        for( int i = 0; i < iInputCount; ++i )
        {
            if( mapxInputs[ i ]->GetResult( ) != 0.0f )
            {
                mafWeights[ i ] += ( fPotential - mfAxonPotential )
                    * fLearningRate / ( static_cast< float >( iInputCount )* mapxInputs[ i ]->GetResult( ) );
            }

            if( mafWeights[ i ] != 0.0f )
            {
                const float fBetterInput = mapxInputs[ i ]->GetResult( ) + ( fPotential - mfAxonPotential )
                    * fLearningRate / ( static_cast< float >( iInputCount )* mafWeights[ i ] );
                mapxInputs[ i ]->BackCycleVirtual( fBetterInput, fLearningRate );
            }
        }

        // adjust bias
        mfBias += ( fPotential - mfAxonPotential ) * fLearningRate
            * ( 1.0f / static_cast< float >( iInputCount ) );
    }

    NeuronBase* mapxInputs[ iInputCount ? iInputCount : 1 ];
    float mafWeights[ iInputCount ? iInputCount : 1 ];
    float mfBias;

};

}

#endif
