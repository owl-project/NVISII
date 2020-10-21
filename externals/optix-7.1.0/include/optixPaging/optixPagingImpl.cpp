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

#include <optixPaging/optixPaging.h>

__device__ __forceinline__ unsigned int countSetBitsAndCalcIndex( const unsigned int laneId, const unsigned int pageBits, unsigned int* pageCount )
{
    // Each thread gets sum of all values of numSetBits for entire warp also do
    // a prefix sum for help indexing later on.
    unsigned int numSetBits = __popc( pageBits );
    unsigned int index      = numSetBits;

#if defined( __CUDACC__ )
#pragma unroll
#endif
    for( unsigned int i = 1; i < 32; i *= 2 )
    {
        numSetBits += __shfl_xor_sync( 0xFFFFFFFF, numSetBits, i );
        unsigned int n = __shfl_up_sync( 0xFFFFFFFF, index, i );

        if( laneId >= i )
            index += n;
    }
    index = __shfl_up_sync( 0xFFFFFFFF, index, 1 );

    // One thread from each warp reserves its global index and updates the count
    // for other warps.
    int startingIndex;
    if( laneId == 0 )
    {
        index = 0;
        if( numSetBits )
            startingIndex = atomicAdd( pageCount, numSetBits );
    }
    index += __shfl_sync( 0xFFFFFFFF, startingIndex, 0 );

    return index;
}

__device__ __forceinline__ void addPagesToList( unsigned int  startingIndex,
                                                unsigned int  pageBits,
                                                unsigned int  pageBitOffset,
                                                unsigned int  maxCount,
                                                unsigned int* outputArray )
{
    while( pageBits != 0 && ( startingIndex < maxCount ) )
    {
        // Find index of least significant bit and clear it
        unsigned int bitIndex = __ffs( pageBits ) - 1;
        pageBits ^= ( 1U << bitIndex );

        // Add the requested page to the queue
        outputArray[startingIndex++] = pageBitOffset + bitIndex;
    }
}

__global__ void devicePullRequests( unsigned int* usageBits,
                                    unsigned int* residenceBits,
                                    unsigned int  maxVaSizeInPages,
                                    unsigned int* devRequestedPages,
                                    unsigned int  numRequestedPages,
                                    unsigned int* numRequestedPagesReturned,
                                    unsigned int* devStalePages,
                                    unsigned int  numStalePages,
                                    unsigned int* numStalePagesReturned,
                                    unsigned int* devEvictablePages,
                                    unsigned int  numEvictablePages,
                                    unsigned int* numEvictablePagesReturned )
{
    unsigned int globalIndex   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int pageBitOffset = globalIndex * 32;

    const unsigned int laneId = globalIndex % 32;
    while( pageBitOffset < maxVaSizeInPages )
    {
        const unsigned int requestWord   = usageBits[globalIndex];
        const unsigned int residenceWord = residenceBits[globalIndex];

        // Gather the outstanding page requests.  A request is 'outstanding' if it
        // is requested but not resident; otherwise we don't need to return the request
        // to the host.
        const unsigned int outstandingRequests = ~residenceWord & requestWord;
        const unsigned int requestIndex = countSetBitsAndCalcIndex( laneId, outstandingRequests, numRequestedPagesReturned );
        addPagesToList( requestIndex, outstandingRequests, pageBitOffset, numRequestedPages, devRequestedPages );

        // Gather the stale pages, which are pages that are resident but not requested.
        const unsigned int stalePages = ~requestWord & residenceWord;
        const unsigned int staleIndex = countSetBitsAndCalcIndex( laneId, stalePages, numStalePagesReturned );
        addPagesToList( staleIndex, stalePages, pageBitOffset, numStalePages, devStalePages );

        globalIndex += gridDim.x * blockDim.x;
        pageBitOffset = globalIndex * 32;
    }

    // TODO: Gather the evictable pages? Or is that host-side work?

    // Clamp counts of returned pages, since they may have been over-incremented
    if( laneId == 0 )
    {
        atomicMin( numRequestedPagesReturned, numRequestedPages );
        atomicMin( numStalePagesReturned, numStalePages );
    }
}

__global__ void deviceFillPages( unsigned long long* pageTable, unsigned int* residenceBits, PageMapping* devFilledPages, int filledPageCount )
{
    int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    while( globalIndex < filledPageCount )
    {
        const PageMapping& devFilledPage = devFilledPages[globalIndex];
        pageTable[devFilledPage.id]      = devFilledPage.page;
        atomicSetBit( devFilledPage.id, residenceBits );
        globalIndex += gridDim.x * blockDim.x;
    }
}

__global__ void deviceInvalidatePages( unsigned int* residenceBits, unsigned int* devInvalidatedPages, int invalidatedPageCount )
{
    int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    while( globalIndex < invalidatedPageCount )
    {
        atomicUnsetBit( devInvalidatedPages[globalIndex], residenceBits );
        globalIndex += gridDim.x * blockDim.x;
    }
}
