// Copyright (c) 2015 Cranium Software

#ifndef NEURON_H
#define NEURON_H

namespace NNL
{

template<
    int iInputCount,
    class Activation,
    class BackPropogator >
class Neuron
{

public:

    void Pulse()
    {
        mfAxonPotential = mfBias;
        for( int i = 0; i < iInputCount; ++i )
        {
            mfAxonPotential += mafInputs[ i ] * mafWeights[ i ];
        }
    }

    float GetAxonPotential() const { return mfAxonPotential; }
    void SetDendriteInput( const int iIndex, const float fValue ) { mafInputs[ iIndex ] = fValue; }

private:

    float mafInputs[ iInputCount ];
    float mafWeights[ iInputCount ];
    float mfBias;
    float mfAxonPotential;

};

}

#endif
