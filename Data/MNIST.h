// Copyright (c) 2015 Cranium Software

#ifndef MNIST_H
#define MNIST_H

namespace NNL
{
    
struct MNIST_Image
{
    unsigned char maaucPixels[ 28 ][ 28 ];
};

struct MNIST_Label
{
    unsigned char mucLabel;
};

MNIST_Image* LoadMNISTImages( const char* const szPath );
void FreeMNISTImages( MNIST_Image* pxImages );
    
MNIST_Label* LoadMNISTLabels( const char* const szPath );
void FreeMNISTLabels( MNIST_Label* pxLabels );
    
}

#endif
