#include <nvisii/nvisii.h>
#include <optix_stubs.h>

// In the future, this file can be used for stuff that uses the CUDA Thrust library

__global__
void _reproject(glm::vec4 *sampleBuffer, glm::vec4 *t0AlbedoBuffer, glm::vec4 *t1AlbedoBuffer, glm::vec4 *mvecBuffer, glm::vec4 *scratchBuffer, glm::vec4 *imageBuffer, bool copy, int width, int height)
{
    // Compute column and row indices.
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = r * width + c; // 1D flat index
    if (i >= (width * height)) return;

    if (copy == true) {
        scratchBuffer[i] = imageBuffer[i];
        return;
    }

    glm::vec2 mvec = -glm::vec2(mvecBuffer[i]) * glm::vec2(width, height);
    glm::vec2 p = glm::vec2(c, r);// / glm::vec2(width, height);

    // mvec = -mvec * glm::vec4(width, height, 0, 0);
    // mvec.x = 0.0f;

    float weight = .95;

    glm::ivec2 reproj = ivec2(p + mvec);
    if (any(greaterThan(reproj, ivec2(width - 1, height - 1)))) weight = 0.0f;
    if (any(lessThan(reproj, ivec2(0, 0)))) weight = 0.0f;
    if (mvecBuffer[i].w < 0) weight = 0.f;

    const int i_reproj = reproj.y * width + reproj.x; // 1D flat index 

    glm::vec4 oldCol = scratchBuffer[i_reproj];
    glm::vec4 curCol = sampleBuffer[i];

    glm::vec4 oldAlb = t0AlbedoBuffer[i_reproj];
    glm::vec4 curAlb = t1AlbedoBuffer[i];

    if (!glm::all(glm::equal(oldAlb, curAlb))) weight = 0.f;

    glm::vec4 newCol = glm::mix(curCol, oldCol, glm::vec4(weight));
    imageBuffer[i] = newCol;
}

void reproject(glm::vec4 *sampleBuffer, glm::vec4 *t0AlbedoBuffer, glm::vec4 *t1AlbedoBuffer, glm::vec4 *mvecBuffer, glm::vec4 *scratchBuffer, glm::vec4 *imageBuffer, int width, int height)
{
    // TEMPORARY, reproject
    dim3 blockSize(32,32);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3 (bx, by);
    _reproject<<<gridSize,blockSize>>>(sampleBuffer, t0AlbedoBuffer, t1AlbedoBuffer, mvecBuffer, scratchBuffer, imageBuffer, true, width, height);
    _reproject<<<gridSize,blockSize>>>(sampleBuffer, t0AlbedoBuffer, t1AlbedoBuffer, mvecBuffer, scratchBuffer, imageBuffer, false, width, height);
}

#include "work_distribution.h"

extern "C" __global__ void fillSamples(
        int   gpu_idx,
        int   num_gpus,
        int   width,
        int   height,
        int2* sample_indices )
{
    StaticWorkDistribution wd;
    wd.setRasterSize( width, height );
    wd.setNumGPUs( num_gpus );

    const int sample_idx = blockIdx.x;
    sample_indices[sample_idx] = wd.getSamplePixel( gpu_idx, sample_idx );
}


extern "C" __host__ void fillSamplesCUDA(
        int          num_samples,
        cudaStream_t stream,
        int          gpu_idx,
        int          num_gpus,
        int          width,
        int          height,
        int2*        sample_indices )
{
    fillSamples<<<num_samples, 1, 0, stream>>>(
        gpu_idx,
        num_gpus,
        width,
        height,
        sample_indices );
}
