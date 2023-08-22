/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda.h>
#include "NvAnalysis.h"

#include <stdio.h>

__global__ void
decoupleLRKernel(int *pDevPtr, int pitch)
{
    int col = threadIdx.x;
    int row = blockIdx.x;

    __shared__ char line[IMG_W*2];

    // Boundary guard.
    if ((col > ((IMG_W*2 + 1) / 2)) || (row > (IMG_H -1))) {
        return;
    }

    char *pLineBegin = (char *)pDevPtr + row * pitch;

    // Use thread 0 of each block to load line into shared mem.
    if ((!col)) {
        // Copy line.
        memcpy(line, pLineBegin, IMG_W*2);
    }

    // Make sure all data are already read before writing.
    __syncthreads();


    // Write pixel to correct position.
    // Remember to process two pixels per thread!
    pLineBegin[col] = line[2*col];
    pLineBegin[(IMG_W*2) / 2 + col] = line[2*col+1];

    __syncthreads();

    return;
}

int
decoupleLR(CUdeviceptr pDevPtr, int pitch)
{
    // Process a whole image line per block, but threads per block has 1k limit.
    // So each thread processes two adjacent pixels.
    dim3 threadsPerBlock((IMG_W*2)/2);
    dim3 blocks(IMG_H);

    //printf("pitch=%d\n", pitch);
    decoupleLRKernel<<<blocks,threadsPerBlock>>>((int *)pDevPtr, pitch);

    return 0;
}


// Remap kernel modified from https://github.com/Wizapply/OvrvisionPro/blob/master/build/linux/CUDA/Remap.cu.
__global__ void remap_kernel(const uint8_t* src, uint8_t* dst, const float* mapx, const float* mapy, int img_pitch)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < IMG_W && y < IMG_H)
    {
        int step = y * IMG_W + x;
        float xcoo = mapx[step];
        float ycoo = mapy[step];
        int X = trunc(xcoo);
        int Y = trunc(ycoo);
        float xfrac = xcoo - X;
        float yfrac = ycoo - Y;
        if (0 <= X && X < IMG_W && 0 <= Y && Y < IMG_H)
        {
            int p00 = src[Y * img_pitch + X];
            int p10 = src[(Y + 1) * img_pitch + X];
            int p01 = src[Y * img_pitch + X + 1];
            int p11 = src[(Y + 1) * img_pitch + X + 1];

            // bilinear interpolation 
            float tmp = ((float)p00 * (1.f - xfrac) + (float)p01 * xfrac) * (1.f - yfrac) 
                        + ((float)p10 * (1.f - xfrac) + (float)p11 * xfrac) * yfrac;
            dst[y * img_pitch + x] = (int)tmp;
        }
    }
}

int remap(const uint8_t* src, uint8_t* dst, const float* mapx, const float* mapy, int img_pitch)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((IMG_W + 15) / (threadsPerBlock.x), (IMG_H + 15) / (threadsPerBlock.y));

    remap_kernel<<<blocks, threadsPerBlock>>>(src, dst, mapx, mapy, img_pitch);

    return 0;
}




__global__ void
addLabelsKernel(int *pDevPtr, int pitch)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y + BOX_H;
    int col = blockIdx.x * blockDim.x + threadIdx.x + BOX_W;
    char *pElement = (char *)pDevPtr + row * pitch + col;

    pElement[0] = 0;

    return;
}

int
addLabels(CUdeviceptr pDevPtr, int pitch)
{
    dim3 threadsPerBlock(BOX_W, BOX_H);
    dim3 blocks(1,1);

    addLabelsKernel<<<blocks,threadsPerBlock>>>((int *)pDevPtr, pitch);

    return 0;
}


__global__ void
convertIntToFloatKernelRGB(CUdeviceptr pDevPtr, int width, int height,
                void* cuda_buf, int pitch, void* offsets_gpu, void* scales_gpu)
{
    float *pdata = (float *)cuda_buf;
    char *psrcdata = (char *)pDevPtr;
    int *offsets = (int *)offsets_gpu;
    float *scales = (float *)scales_gpu;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        for (int k = 0; k < 3; k++)
        {
            pdata[width * height * k + row * width + col] =
                (float)(*(psrcdata + row * pitch + col * 4 + (3 - 1 - k)) - offsets[k]) * scales[k];
        }
    }
}

__global__ void
convertIntToFloatKernelBGR(CUdeviceptr pDevPtr, int width, int height,
                void* cuda_buf, int pitch, void* offsets_gpu, void* scales_gpu)
{
    float *pdata = (float *)cuda_buf;
    char *psrcdata = (char *)pDevPtr;
    int *offsets = (int *)offsets_gpu;
    float *scales = (float *)scales_gpu;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (col < width && row < height)
    {
        // For V4L2_PIX_FMT_ABGR32 --> BGRA-8-8-8-8
        for (int k = 0; k < 3; k++)
        {
            pdata[width * height * k + row * width + col] =
                (float)(*(psrcdata + row * pitch + col * 4 + k) - offsets[k]) * scales[k];
        }
    }
}

int convertIntToFloat(CUdeviceptr pDevPtr,
                      int width,
                      int height,
                      int pitch,
                      COLOR_FORMAT color_format,
                      void* offsets,
                      void* scales,
                      void* cuda_buf, void* pstream)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height +
          threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaStream_t stream;
    if (pstream!= NULL)
        stream = *(cudaStream_t*)pstream;
    else
        stream = 0;

    if (color_format == COLOR_FORMAT_RGB)
    {
        convertIntToFloatKernelRGB<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
                height, cuda_buf, pitch, offsets, scales);
    }
    else if (color_format == COLOR_FORMAT_BGR)
    {
        convertIntToFloatKernelBGR<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
                height, cuda_buf, pitch, offsets, scales);
    }

    return 0;
}
