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
#include "tag36h11.h"

#include <stdio.h>
#include <signal.h>
#include <cassert>
#include <cuda.h>
#include <algorithm>

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
        printf("usage:  v4l2-console <filename> [--tag16h5]\n");
        printf("      ./v4l2-console /dev/video0\n");
        printf("      ./v4l2-console /dev/video0 --tag16h5\n");
        
        return 0;
    }
    
    const char* dev_path = argv[1];
    bool use_tag16h5 = false;
    
    // Parse command line arguments
    for(int i = 2; i < argc; i++) {
        if(strcmp(argv[i], "--tag16h5") == 0) {
            use_tag16h5 = true;
        }
    }
    
    printf("v4l2-console:   attempting to initialize video device '%s'\n", dev_path);
    printf("Using tag family: %s\n", use_tag16h5 ? "tag16h5" : "tag36h11");
    
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
    
    // Initialize CUDA-OpenGL interoperability
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("Failed to initialize CUDA device: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    // Print CUDA device properties
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus == cudaSuccess) {
        printf("CUDA Device Properties:\n");
        printf("  Device Name: %s\n", deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Can Map Host Memory: %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Integrated GPU: %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Unified Addressing: %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
    }

    /*
     * create openGL window
     */
    videoOptions displayOpts;
    displayOpts.resource = "display://0";
    displayOpts.width = camera->GetPitch();
    displayOpts.height = camera->GetHeight();
    
    glDisplay* display = glDisplay::Create(displayOpts);
    
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
    //
    // Not sure why below line doesn't work. Seems the buffer never really got allocated.
    //image_u8_t* img_tag = image_u8_create(2*camera->GetWidth(), camera->GetHeight());
    //
    // Below line works.
	image_u8_t im = {
		.width = 2*camera->GetWidth(),
		.height = camera->GetHeight(),
		.stride = 2*camera->GetWidth(),
		.buf = new uint8_t[2*camera->GetWidth() * camera->GetHeight()]
	};
    image_u8_t* img_tag = &im;

    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_family_t *tf = nullptr;
    
    if(use_tag16h5) {
        tf = tag16h5_create();
        printf("Initialized tag16h5 detector\n");
    } else {
        tf = tag36h11_create();
        printf("Initialized tag36h11 detector\n");
    }
    
    apriltag_detector_add_family(td, tf);

    // Config tag detector.
	td->quad_decimate = 2.0;
	td->quad_sigma = 0.0;
	td->nthreads = 8;
	td->debug = false;  // Disable debug mode for better performance
	td->refine_edges = false;  // Disable edge refinement for better performance



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
        uint8_t* img = (uint8_t*)camera->Capture(50);
        
        if( !img )
        {
            printf("got NULL image from camera capture\n");
            continue;
        }
        else
        {
            //printf("recieved new video frame\n");

            cudaMemcpy(img_dev, img, 2*sizeOfImage*sizeof(uint8_t), cudaMemcpyHostToDevice);
            //printf("Copied img to img_dev.\n");

            // CUDA proc
            decoupleLR((CUdeviceptr) img_dev, width*2);
            cudaDeviceSynchronize();
            remap(img_dev, img_dev + width, mapxDevPtr, mapyDevPtr, width*2);
            cudaDeviceSynchronize();
            //printf("CUDA kernels done.\n");

            // Copy undistorted image to host.
            cudaError_t err = cudaMemcpy(img_tag->buf, img_dev, 2*sizeOfImage*sizeof(uint8_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("CUDA memcpy failed: %s\n", cudaGetErrorString(err));
                continue;
            }

            // Debug: Check input values
            //if (img_cnt == 30) {  
            //    uint8_t min_val = 255;
            //    uint8_t max_val = 0;
            //    for(int i = 0; i < img_tag->width * img_tag->height; i++) {
            //        min_val = std::min(min_val, img_tag->buf[i]);
            //        max_val = std::max(max_val, img_tag->buf[i]);
            //    }
            //    printf("Input image stats - Min: %d, Max: %d\n", min_val, max_val);
            //}

            zarray_t *detections = apriltag_detector_detect(td, img_tag);

            // Print detection statistics
            //if (img_cnt % 30 == 0) {  // Print every 30 frames
            //    printf("Detection statistics:\n");
            //    printf("  Number of edges detected: %d\n", td->nedges);
            //    printf("  Number of segments detected: %d\n", td->nsegments);
            //    printf("  Number of quads detected: %d\n", td->nquads);
            //    printf("  Number of tags detected: %d\n", zarray_size(detections));
            //    
            //    // Print timing information
            //    timeprofile_display(td->tp);
            //}

            //if (img_cnt==30) {
            //    FILE *fout = fopen("frame.raw", "wb");
            //    fwrite(img_tag->buf,camera->GetPitch()*camera->GetHeight(), 1, fout);
            //    fclose(fout);
            //}

            img_cnt++;

            // update display
            if( display != NULL )
            {
                display->BeginRender();
                
                // Render the image directly from apriltag buffer with explicit format.
                // glDisplay requires renderImage buffer at device, so cannot render img_tag->buf directly.
                display->RenderImage(img_dev, camera->GetPitch(), camera->GetHeight(), IMAGE_GRAY8, 0, 0, false);
                //printf("Update display.\n");

                for (int i = 0; i < zarray_size(detections); i++) {
                    apriltag_detection_t *det;
                    zarray_get(detections, i, &det);

                    //printf("det->decision_margin = %f.\n", det->decision_margin);
                
                    if (det->decision_margin > 100.f) {
                        printf("detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                            i, det->family->nbits, det->family->h, det->id, det->hamming, det->decision_margin);

                        display->RenderLine(det->p[1][0], det->p[1][1], det->p[0][0], det->p[0][1], 0.9f, 0.f, 0.f);
                        display->RenderLine(det->p[0][0], det->p[0][1], det->p[3][0], det->p[3][1], 0.f, 0.9f, 0.f);
                    }
                }


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
            
        // Sleep number seconds.
        //sleep(1);
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
