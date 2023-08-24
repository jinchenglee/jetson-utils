/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "v4l2Camera.h"
#include "glDisplay.h"
#include "NvAnalysis.h"
#include "mapxpy.h"

// AprilTag related
#include "apriltag.h"
#include "tag16h5.h"

#include <stdio.h>
#include <signal.h>
#include <cassert>
#include <cuda.h>

bool signal_recieved = false;

void sig_handler(int signo)
{
    if( signo == SIGINT )
    {
        printf("received SIGINT\n");
        signal_recieved = true;
    }
}



int main( int argc, char** argv )
{
    printf("v4l2-console\n  args (%i):  ", argc);
    
    /*
     * verify parameters
     */
    for( int i=0; i < argc; i++ )
        printf("%i [%s]  ", i, argv[i]);
        
    printf("\n");
    
    if( argc < 2 )
    {
        printf("v4l2-console:  0 arguments were supplied.\n");
        printf("usage:  v4l2-console <filename>\n");
        printf("      ./v4l2-console /dev/video0\n");
        
        return 0;
    }
    
    const char* dev_path = argv[1];
    printf("v4l2-console:   attempting to initialize video device '%s'\n\n", dev_path);
    
    if( signal(SIGINT, sig_handler) == SIG_ERR )
        printf("\ncan't catch SIGINT\n");

    /*
     * create the camera device
     */
    v4l2Camera* camera = v4l2Camera::Create(dev_path);
    
    if( !camera )
    {
        printf("\nv4l2-console:  failed to initialize video device '%s'\n", dev_path);
        return 0;
    }
    
    printf("\nv4l2-console:  successfully initialized video device '%s'\n", dev_path);
    printf("    width:  %u\n", camera->GetWidth());
    printf("   height:  %u\n", camera->GetHeight());
    printf("    depth:  %u (bpp)\n", camera->GetPixelDepth());
    

    /*
     * create openGL window
     */
    glDisplay* display = glDisplay::Create("", camera->GetPitch(), camera->GetHeight(), 0.f, 0.f, 0.f, 0.f);
    
    if( !display )
    {
        printf("\nv4l2-display:  failed to create openGL display\n");
        return 0;
    }

    
    /*
     * start streaming
     */
    if( !camera->Open() )
    {
        printf("\nv4l2-console:  failed to open camera '%s' for streaming\n", dev_path);
        return 0;
    }
    
    printf("\nv4l2-console:  camera '%s' open for streaming\n", dev_path);

    assert(camera->GetPitch() == IMG_W*2);
    assert(camera->GetHeight() == IMG_H);
    int height = IMG_H;
    int width = IMG_W;
    size_t sizeOfImage = width * height;

    // malloc() apriltag image on Host
    image_u8_t* img_tag = image_u8_create(2*camera->GetWidth(), camera->GetHeight());

    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_family_t *tf = tag16h5_create();


    // Device memory for CUDA processing.
    uint8_t* img_dev = nullptr;
    float *mapxDevPtr, *mapyDevPtr;
    if( CUDA(cudaMalloc(&img_dev, 2*sizeOfImage * sizeof(uint8_t))) )
        printf("cudaMalloc img_dev failed!\n");
    if( CUDA(cudaMalloc(&mapxDevPtr, sizeOfImage * sizeof(float))) )
        printf("cudaMalloc mapxDevPtr failed!\n");
    if( CUDA(cudaMalloc(&mapyDevPtr, sizeOfImage * sizeof(float))) )
        printf("cudaMalloc mapyDevPtr failed!\n");

    // Copy mapx mapy to device mem.
    cudaMemcpy(mapxDevPtr, mapx, sizeOfImage * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mapyDevPtr, mapy, sizeOfImage * sizeof(float), cudaMemcpyHostToDevice);


    uint32_t img_cnt = 0;
    while( !signal_recieved )
    {
        uint8_t* img = (uint8_t*)camera->Capture(500);
        
        if( !img )
        {
            //printf("got NULL image from camera capture\n");
            continue;
        }
        else
        {
            //printf("recieved new video frame\n");

            if (img_cnt==0) {
                FILE *fout = fopen("frame.raw", "wb");
                fwrite(img, camera->GetPitch()*camera->GetHeight(), 1, fout);
                fclose(fout);
            }

            img_cnt++;

            cudaMemcpy(img_dev, img, 2*sizeOfImage*sizeof(uint8_t), cudaMemcpyHostToDevice);
            //printf("Copied img to img_dev.\n");

            // CUDA proc
            decoupleLR((CUdeviceptr) img_dev, width*2);
            cudaDeviceSynchronize();
            remap(img_dev, img_dev + width, mapxDevPtr, mapyDevPtr, width*2);
            cudaDeviceSynchronize();
            //printf("CUDA kernels done.\n");

            // Copy undistorted image to host.
            cudaMemcpy(img_tag->buf, img_dev, 2*sizeOfImage*sizeof(uint8_t), cudaMemcpyDeviceToHost);

            apriltag_detector_add_family(td, tf);
            zarray_t *detections = apriltag_detector_detect(td, img_tag);

            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t *det;
                zarray_get(detections, i, &det);
            
                // Do stuff with detections here.
                printf("detected id: %d\n", det->id);
            }

            // update display
            if( display != NULL )
            {
                //display->Render((uint8_t*)img, camera->GetWidth(), camera->GetHeight(), IMAGE_RGBA8);
                //display->RenderOnce((uint8_t*)img_dev, camera->GetPitch(), camera->GetHeight(), IMAGE_GRAY8, 0, 0);

                // Manually control the rendering.
	            display->BeginRender();

	            display->RenderImage((uint8_t*)img_dev, camera->GetPitch(), camera->GetHeight(), IMAGE_GRAY8, 0, 0);
	            display->RenderLine(10.f, 10.f, 50.f, 50.f, 0.2f, 0.3f, 0.5f);

	            display->EndRender();

                // update status bar
                char str[256];
                sprintf(str, "v4l2-console (%ux%u) | %.0f FPS", camera->GetWidth(), camera->GetHeight(), display->GetFPS());
                display->SetTitle(str); 

                // check if the user quit
                if( display->IsClosed() )
                    signal_recieved = true;
            }

        }
            
    }
    
    // Free cuda allocations.
    cudaFree(img_dev);
    cudaFree(mapxDevPtr);
    cudaFree(mapyDevPtr);

    // Cleanup.
    tag16h5_destroy(tf);
    apriltag_detector_destroy(td);

    image_u8_destroy(img_tag);
    
    /*
     * shutdown the camera device
     */
    if( display != NULL )
    {
        delete display;
        display = NULL;
    }

    printf("\nv4l2-console:  un-initializing video device '%s'\n", dev_path);
    if( camera != NULL )
    {
        delete camera;
        camera = NULL;
    }
    
    printf("v4l2-console:  video device '%s' has been un-initialized.\n", dev_path);
    printf("v4l2-console:  this concludes the test of video device '%s'\n", dev_path);
    return 0;
}
