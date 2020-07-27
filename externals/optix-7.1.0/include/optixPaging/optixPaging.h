//
// Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <cuda_runtime.h>

#if !defined(__CUDACC_RTC__)
#include <utility>
#include <stdio.h>
#endif

inline bool optixPagingCheckCudaError( cudaError_t err )
{
    if( err != cudaSuccess )
    {
        printf( "CUDA error: %d\n", err );
        return false;
    }
    return true;
}

#if !defined( OPTIX_PAGING_CHECK_CUDA_ERROR )
#define OPTIX_PAGING_CHECK_CUDA_ERROR( err ) optixPagingCheckCudaError( err )
#endif

const int MAX_WORKER_THREADS = 32;

template <typename T>
__host__ __device__ T minimum( T lhs, T rhs )
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
__host__ __device__ T maximum( T lhs, T rhs )
{
    return lhs > rhs ? lhs : rhs;
}

struct PageMapping
{
    unsigned int id;
    unsigned long long page;
};

struct OptixPagingSizes
{
    unsigned int pageTableSizeInBytes;  // only one for all workers
    unsigned int usageBitsSizeInBytes;  // per worker
};

struct OptixPagingOptions
{
    unsigned int maxVaSizeInPages;
    unsigned int initialVaSizeInPages;
};

struct OptixPagingContext
{
    unsigned int  maxVaSizeInPages;
    unsigned int* usageBits;      // also beginning of referenceBits. [ referenceBits | residencesBits ]
    unsigned int* residenceBits;  // located half way into usasgeBits.
    unsigned long long* pageTable;
};

#ifndef __CUDACC_RTC__
__host__ void optixPagingCreate( OptixPagingOptions* options, OptixPagingContext** context );
__host__ void optixPagingDestroy( OptixPagingContext* context );
__host__ void optixPagingCalculateSizes( unsigned int vaSizeInPages, OptixPagingSizes& sizes );
__host__ void optixPagingSetup( OptixPagingContext* context, const OptixPagingSizes& sizes, int numWorkers );
__host__ void optixPagingPullRequests( OptixPagingContext* context,
                                       unsigned int*       devRequestedPages,
                                       unsigned int        numRequestedPages,
                                       unsigned int*       devStalePages,
                                       unsigned int        numStalePages,
                                       unsigned int*       devEvictablePages,
                                       unsigned int        numEvictablePages,
                                       unsigned int*       devNumPagesReturned );
__host__ void optixPagingPushMappings( OptixPagingContext* context,
                                       PageMapping*            devFilledPages,
                                       int                 filledPageCount,
                                       unsigned int*       devInvalidatedPages,
                                       int                 invalidatedPageCount );
#endif

#if defined( __CUDACC__ ) || defined( OPTIX_PAGING_BIT_OPS )
__device__ inline void atomicSetBit( unsigned int bitIndex, unsigned int* bitVector )
{
    const unsigned int wordIndex = bitIndex >> 5;
    const unsigned int bitOffset = bitIndex % 32;
    const unsigned int mask      = 1U << bitOffset;
    atomicOr( bitVector + wordIndex, mask );
}

__device__ inline void atomicUnsetBit( int bitIndex, unsigned int* bitVector )
{
    const int wordIndex = bitIndex / 32;
    const int bitOffset = bitIndex % 32;

    const int mask = ~( 1U << bitOffset );
    atomicAnd( bitVector + wordIndex, mask );
}

__device__ inline bool checkBitSet( unsigned int bitIndex, const unsigned int* bitVector )
{
    const unsigned int wordIndex = bitIndex >> 5;
    const unsigned int bitOffset = bitIndex % 32;
    return ( bitVector[wordIndex] & ( 1U << bitOffset ) ) != 0;
}

__device__ inline unsigned long long optixPagingMapOrRequest( unsigned int* usageBits, unsigned int* residenceBits, unsigned long long* pageTable, unsigned int page, bool* valid )
{
    bool requested = checkBitSet( page, usageBits );
    if( !requested )
        atomicSetBit( page, usageBits );

    bool mapped = checkBitSet( page, residenceBits );
    *valid      = mapped;

    return mapped ? pageTable[page] : 0;
}
#endif
