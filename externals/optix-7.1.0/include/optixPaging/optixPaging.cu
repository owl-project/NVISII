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

#include <optixPaging/optixPagingImpl.cpp>

__host__ void optixPagingPullRequests( OptixPagingContext* context,
                                       unsigned int*       devRequestedPages,
                                       unsigned int        numRequestedPages,
                                       unsigned int*       devStalePages,
                                       unsigned int        numStalePages,
                                       unsigned int*       devEvictablePages,
                                       unsigned int        numEvictablePages,
                                       unsigned int*       devNumPagesReturned )
{
    OPTIX_PAGING_CHECK_CUDA_ERROR( cudaMemset( devRequestedPages, 0, numRequestedPages * sizeof( unsigned int ) ) );
    OPTIX_PAGING_CHECK_CUDA_ERROR( cudaMemset( devStalePages, 0, numStalePages * sizeof( unsigned int ) ) );
    OPTIX_PAGING_CHECK_CUDA_ERROR( cudaMemset( devEvictablePages, 0, numEvictablePages * sizeof( unsigned int ) ) );
    OPTIX_PAGING_CHECK_CUDA_ERROR( cudaMemset( devNumPagesReturned, 0, 3 * sizeof( unsigned int ) ) );

    int numPagesPerThread = context->maxVaSizeInPages / 65536;
    numPagesPerThread     = ( numPagesPerThread + 31 ) & 0xFFFFFFE0;  // Round to nearest multiple of 32
    if( numPagesPerThread < 32 )
        numPagesPerThread = 32;

    const int numThreadsPerBlock = 64;
    const int numPagesPerBlock   = numPagesPerThread * numThreadsPerBlock;
    const int numBlocks          = ( context->maxVaSizeInPages + ( numPagesPerBlock - 1 ) ) / numPagesPerBlock;

    devicePullRequests<<<numBlocks, numThreadsPerBlock>>>( context->usageBits, context->residenceBits, context->maxVaSizeInPages,
                                                           devRequestedPages, numRequestedPages, devNumPagesReturned,
                                                           devStalePages, numStalePages, devNumPagesReturned + 1,
                                                           devEvictablePages, numEvictablePages, devNumPagesReturned + 2 );
}

__host__ void optixPagingPushMappings( OptixPagingContext* context,
                                       PageMapping*        devFilledPages,
                                       int                 filledPageCount,
                                       unsigned int*       devInvalidatedPages,
                                       int                 invalidatedPageCount )
{
    // Zero out the reference bits
    unsigned int referenceBitsSizeInBytes = sizeof( unsigned int ) * static_cast<unsigned int>( context->residenceBits - context->usageBits );
    OPTIX_PAGING_CHECK_CUDA_ERROR( cudaMemset( context->usageBits, 0, referenceBitsSizeInBytes ) );

    const int numPagesPerThread = 2;
    const int numThreadsPerBlock = 128;
    const int numPagesPerBlock = numPagesPerThread * numThreadsPerBlock;
    if( filledPageCount != 0 )
    {
        const int numFilledPageBlocks = ( filledPageCount + numPagesPerBlock - 1 ) / numPagesPerBlock;
        deviceFillPages<<<numFilledPageBlocks, numThreadsPerBlock>>>( context->pageTable, context->residenceBits,
                                                                      devFilledPages, filledPageCount );
    }

    if( invalidatedPageCount != 0 )
    {
        const int numInvalidatedPageBlocks = ( invalidatedPageCount + numPagesPerBlock - 1 ) / numPagesPerBlock;
        deviceInvalidatePages<<<numInvalidatedPageBlocks, numThreadsPerBlock>>>( context->residenceBits, devInvalidatedPages,
                                                                                 invalidatedPageCount );
    }
}

__host__ void optixPagingCreate( OptixPagingOptions* options, OptixPagingContext** context )
{
    *context                       = new OptixPagingContext;
    ( *context )->maxVaSizeInPages = options->maxVaSizeInPages;
    ( *context )->usageBits        = nullptr;
    ( *context )->pageTable        = nullptr;
}

__host__ void optixPagingDestroy( OptixPagingContext* context )
{
    delete context;
}

__host__ void optixPagingCalculateSizes( unsigned int vaSizeInPages, OptixPagingSizes& sizes )
{
    //TODO: decide on limit for sizes, add asserts

    // Number of entries * 8 bytes per entry
    sizes.pageTableSizeInBytes = vaSizeInPages * sizeof( unsigned long long );

    // Calc reference bits size with 128 byte alignnment, residence bits are same size.
    // Usage bits is the concatenation of the two.
    unsigned int referenceBitsSizeInBytes = ( ( vaSizeInPages + 1023 ) & 0xFFFFFC00 ) / 8;
    unsigned int residenceBitsSizeInBytes = referenceBitsSizeInBytes;
    sizes.usageBitsSizeInBytes            = referenceBitsSizeInBytes + residenceBitsSizeInBytes;
}

__host__ void optixPagingSetup( OptixPagingContext* context, const OptixPagingSizes& sizes, int numWorkers )
{
    // TODO: decide on limit for numWorkers, add asserts

    // This doubles as a memset and a check to make sure they allocated the device pointers
    OPTIX_PAGING_CHECK_CUDA_ERROR( cudaMemset( context->pageTable, 0, sizes.pageTableSizeInBytes ) );
    OPTIX_PAGING_CHECK_CUDA_ERROR( cudaMemset( context->usageBits, 0, sizes.usageBitsSizeInBytes * numWorkers ) );

    // Set residence bits pointer in context (index half way into usage bits)
    context->residenceBits = context->usageBits + ( sizes.usageBitsSizeInBytes / sizeof(unsigned int) / 2 );
}
