#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SWD_HOSTDEVICE __host__ __device__
#    define SWD_INLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
#else
#    define SWD_HOSTDEVICE
#    define SWD_INLINE inline
#    define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#endif

#include <stdint.h>

#include <vector_types.h>

class StaticWorkDistribution
{
public:
    SWD_INLINE SWD_HOSTDEVICE void setRasterSize( int width, int height )
    {
        m_width = width;
        m_height = height;
    }


    SWD_INLINE SWD_HOSTDEVICE void setNumGPUs( int32_t num_gpus )
    {
        m_num_gpus = num_gpus;
    }


    SWD_INLINE SWD_HOSTDEVICE int32_t numSamples( )
    {
        const int tile_strip_width  = TILE_WIDTH*m_num_gpus;
        const int tile_strip_height = TILE_HEIGHT;
        const int num_tile_strip_cols = m_width /tile_strip_width  + ( m_width %tile_strip_width  == 0 ? 0 : 1 );
        const int num_tile_strip_rows = m_height/tile_strip_height + ( m_height%tile_strip_height == 0 ? 0 : 1 );
        return num_tile_strip_rows*num_tile_strip_cols*TILE_WIDTH*TILE_HEIGHT;
    }


    SWD_INLINE SWD_HOSTDEVICE int2 getSamplePixel( int32_t gpu_idx, int32_t sample_idx )
    {
        const int tile_strip_width  = TILE_WIDTH*m_num_gpus;
        const int tile_strip_height = TILE_HEIGHT;
        const int num_tile_strip_cols = m_width /tile_strip_width + ( m_width % tile_strip_width == 0 ? 0 : 1 );

        const int tile_strip_idx     = sample_idx / (TILE_WIDTH*TILE_HEIGHT );
        const int tile_strip_y       = tile_strip_idx / num_tile_strip_cols;
        const int tile_strip_x       = tile_strip_idx - tile_strip_y * num_tile_strip_cols;
        const int tile_strip_x_start = tile_strip_x * tile_strip_width;
        const int tile_strip_y_start = tile_strip_y * tile_strip_height;

        const int tile_pixel_idx     = sample_idx - ( tile_strip_idx * TILE_WIDTH*TILE_HEIGHT );
        const int tile_pixel_y       = tile_pixel_idx / TILE_WIDTH;
        const int tile_pixel_x       = tile_pixel_idx - tile_pixel_y * TILE_WIDTH;

        const int tile_offset_x      = ( gpu_idx + tile_strip_y % m_num_gpus ) % m_num_gpus * TILE_WIDTH;

        const int pixel_y = tile_strip_y_start + tile_pixel_y;
        const int pixel_x = tile_strip_x_start + tile_pixel_x + tile_offset_x ;
        return make_int2( pixel_x, pixel_y );
    }


private:
    int32_t m_num_gpus = 0;
    int32_t m_width    = 0;
    int32_t m_height   = 0;

    static const int32_t TILE_WIDTH  = 8;
    static const int32_t TILE_HEIGHT = 4;
};
