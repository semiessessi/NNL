// Copyright (c) 2015 Cranium Software

#include "MNIST.h"

#include <cstdio>

#ifdef _WIN32
#include <WinSock2.h>

#pragma comment( lib, "ws2_32" )
#else
#include <arpa/inet.h>
#endif

namespace NNL
{
    MNIST_Image* LoadMNISTImages( const char* const szPath )
    {
        MNIST_Image* pxReturnValue = NULL;
        FILE* pxFile = fopen( szPath, "rb" );
        if( pxFile )
        {
            unsigned int uMagic = 0;
            fread( &uMagic, sizeof( unsigned int ), 1, pxFile );
            uMagic = ntohl( uMagic );
            if( uMagic != 2051 )
            {
                return pxReturnValue;
            }
            
            unsigned int uCount = 0;
            fread( &uCount, sizeof( unsigned int ), 1, pxFile );
            uCount = ntohl( uCount );
            pxReturnValue = new MNIST_Image[ uCount ];
            
            fread( &uMagic, sizeof( unsigned int ), 1, pxFile );
            uMagic = ntohl( uMagic );
            if( uMagic != 28 )
            {
                return pxReturnValue;
            }
            
            fread( &uMagic, sizeof( unsigned int ), 1, pxFile );
            uMagic = ntohl( uMagic );
            if( uMagic != 28 )
            {
                return pxReturnValue;
            }
            
            fread( pxReturnValue, sizeof( MNIST_Image ), uCount, pxFile );
            
            fclose( pxFile );
        }
        
        return pxReturnValue;
    }
    
    void FreeMNISTImages( MNIST_Image* pxImages )
    {
        delete[] pxImages;
    }
    
    MNIST_Label* LoadMNISTLabels( const char* const szPath )
    {
        MNIST_Label* pxReturnValue = NULL;
        FILE* pxFile = fopen( szPath, "rb" );
        if( pxFile )
        {
            unsigned int uMagic = 0;
            fread( &uMagic, sizeof( unsigned int ), 1, pxFile );
            uMagic = ntohl( uMagic );
            if( uMagic != 2049 )
            {
                return pxReturnValue;
            }
            
            unsigned int uCount = 0;
            fread( &uCount, sizeof( unsigned int ), 1, pxFile );
            uCount = ntohl( uCount );
            
            pxReturnValue = new MNIST_Label[ uCount ];
            fread( pxReturnValue, sizeof( MNIST_Label ), uCount, pxFile );
            
            fclose( pxFile );
        }
        
        return pxReturnValue;
    }
    
    void FreeMNISTLabels( MNIST_Label* pxLabels )
    {
        delete[] pxLabels;
    }
}
