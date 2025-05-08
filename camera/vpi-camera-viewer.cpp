/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "gstCamera.h"
#include "glDisplay.h"
#include "glTexture.h"
#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"
#include "cudaOverlay.h"
#include "cudaResize.h"

//#include <vpi/OpenCVInterop.hpp>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/AprilTags.h>

#include <signal.h>
#include <stdio.h>
#include <string.h>

bool signal_recieved = false;

void sig_handler(int signo)
{
    if( signo == SIGINT )
    {
        printf("received SIGINT\n");
        signal_recieved = true;
    }
}

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            printf("%s: %s\n", vpiStatusGetName(status), buffer); \
            return 0;                                         \
        }                                                     \
    } while (0);

int main( int argc, char** argv )
{
    printf("vpi-camera-viewer\n  args (%i):  ", argc);
    for( int i=0; i < argc; i++ )
        printf("%i [%s]  ", i, argv[i]);
    printf("\n");
    
    if( argc < 2 )
    {
        printf("vpi-camera-viewer:  0 arguments were supplied.\n");
        printf("usage:  vpi-camera-viewer <filename> [--backend=cpu|pva]\n");
        printf("      ./vpi-camera-viewer /dev/video0\n");
        printf("      ./vpi-camera-viewer /dev/video0 --backend=pva\n");
        return 0;
    }

    // Parse command line arguments
    const char* dev_path = argv[1];
    VPIBackend backend = VPI_BACKEND_CPU;
    
    // Camera configuration
    videoOptions cameraOpts;
    cameraOpts.resource = "csi://0";  // Use CSI camera 0
    cameraOpts.width = 1280;
    cameraOpts.height = 720;
    cameraOpts.frameRate = 30;
    cameraOpts.flipMethod = videoOptions::FLIP_ROTATE_180;  // Adjust if needed
    
    for(int i = 2; i < argc; i++) {
        if(strncmp(argv[i], "--backend=", 10) == 0) {
            const char* backend_str = argv[i] + 10;
            if(strcmp(backend_str, "pva") == 0)
                backend = VPI_BACKEND_PVA;
            else if(strcmp(backend_str, "cpu") == 0)
                backend = VPI_BACKEND_CPU;
            else {
                printf("Invalid backend specified. Using CPU.\n");
            }
        }
    }
    
    printf("Using VPI backend: %s\n", backend == VPI_BACKEND_PVA ? "PVA" : "CPU");
    
    if( signal(SIGINT, sig_handler) == SIG_ERR )
        printf("\ncan't catch SIGINT\n");

    // Create camera with options
    gstCamera* camera = gstCamera::Create(cameraOpts);
    
    if( !camera )
    {
        printf("\nvpi-camera-viewer:  failed to initialize video device '%s'\n", dev_path);
        return 0;
    }
    
    printf("\nvpi-camera-viewer:  successfully initialized video device '%s'\n", dev_path);
    printf("    width:  %u\n", camera->GetWidth());
    printf("   height:  %u\n", camera->GetHeight());
    
    // Create display window
    glDisplay* display = glDisplay::Create();
    
    if( !display )
    {
        printf("\nvpi-camera-viewer:  failed to create openGL display\n");
        return 0;
    }

    // Create VPI resources
    VPIImage imgInput = NULL;
    VPIImage imgGrayscale = NULL;
    VPIArray detections = NULL;
    VPIArray poses = NULL;
    VPIStream stream = NULL;
    VPIPayload payload = NULL;

    // AprilTag parameters
    const int maxHamming = 1;
    const VPIAprilTagFamily family = VPI_APRILTAG_36H11;
    VPIAprilTagDecodeParams apritagDecodeParams = {NULL, 0, maxHamming, family};
    const int maxDetections = 64;

    // Create VPI stream
    CHECK_STATUS(vpiStreamCreate(0, &stream));

    // Create VPI images - create once and reuse
    CHECK_STATUS(vpiImageCreate(camera->GetWidth(), camera->GetHeight(), VPI_IMAGE_FORMAT_RGB8, 0, &imgInput));
    CHECK_STATUS(vpiImageCreate(camera->GetWidth(), camera->GetHeight(), VPI_IMAGE_FORMAT_U8, 0, &imgGrayscale));

    // Create detection and pose arrays
    CHECK_STATUS(vpiArrayCreate(maxDetections, VPI_ARRAY_TYPE_APRILTAG_DETECTION, VPI_BACKEND_CPU | VPI_BACKEND_PVA, &detections));
    CHECK_STATUS(vpiArrayCreate(maxDetections, VPI_ARRAY_TYPE_POSE, VPI_BACKEND_CPU | VPI_BACKEND_PVA, &poses));

    // Create AprilTag detector payload
    CHECK_STATUS(vpiCreateAprilTagDetector(backend, camera->GetWidth(), camera->GetHeight(), &apritagDecodeParams, &payload));

    // Camera intrinsics for pose estimation
    const VPICameraIntrinsic intrinsics = {
        {camera->GetWidth() / 3.5f, 0.0f, camera->GetWidth() / 2.f},
        {0.0f, camera->GetHeight() / 3.6f, camera->GetHeight() / 2.f}
    };
    const float tagSize = 0.2f;

    // Start streaming
    if( !camera->Open() )
    {
        printf("\nvpi-camera-viewer:  failed to open camera '%s' for streaming\n", dev_path);
        return 0;
    }
    
    printf("\nvpi-camera-viewer:  camera '%s' open for streaming\n", dev_path);

    // Main processing loop
    while( !signal_recieved )
    {
        // Capture image in RGBA format with increased timeout
        float* imgRGBA = NULL;
        if( !camera->CaptureRGBA(&imgRGBA, 1000) )  // Increased timeout to 1000ms
        {
            printf("failed to capture RGBA image\n");
            continue;
        }

        // Copy RGBA data to VPI image (converting from float to uint8)
        VPIImageData imgData;
        CHECK_STATUS(vpiImageLockData(imgInput, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));
        
        // Convert RGBA float to RGB uint8
        uint8_t* dst = (uint8_t*)imgData.buffer.pitch.planes[0].data;
        const float* src = imgRGBA;
        for(int i = 0; i < camera->GetWidth() * camera->GetHeight(); i++) {
            dst[i*3 + 0] = (uint8_t)(src[i*4 + 0] * 255.0f); // R
            dst[i*3 + 1] = (uint8_t)(src[i*4 + 1] * 255.0f); // G
            dst[i*3 + 2] = (uint8_t)(src[i*4 + 2] * 255.0f); // B
        }
        
        CHECK_STATUS(vpiImageUnlock(imgInput));

        // Convert to grayscale
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CPU, imgInput, imgGrayscale, NULL));

        // Detect AprilTags
        CHECK_STATUS(vpiSubmitAprilTagDetector(stream, backend, payload, maxDetections, imgGrayscale, detections));

        // Estimate poses
        CHECK_STATUS(vpiSubmitAprilTagPoseEstimation(stream, VPI_BACKEND_CPU, detections, intrinsics, tagSize, poses));

        // Wait for processing to complete
        CHECK_STATUS(vpiStreamSync(stream));

        // Get detection results
        VPIArrayData outDetectionsData;
        VPIArrayData outPosesData;
        CHECK_STATUS(vpiArrayLockData(detections, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outDetectionsData));
        CHECK_STATUS(vpiArrayLockData(poses, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outPosesData));

        VPIAprilTagDetection* outDetections = (VPIAprilTagDetection*)outDetectionsData.buffer.aos.data;
        VPIPose* outPoses = (VPIPose*)outPosesData.buffer.aos.data;
        int numDetections = *outDetectionsData.buffer.aos.sizePointer;

        // Display the image
        display->BeginRender();

        // Render the camera image
        display->RenderImage(imgRGBA, camera->GetWidth(), camera->GetHeight(), IMAGE_RGB8);

        // Draw detections
        for (int i = 0; i < numDetections; ++i) {
            const VPIAprilTagDetection& det = outDetections[i];
            
            // Draw tag corners
            for (int j = 0; j < 4; ++j) {
                int next = (j + 1) % 4;
                display->RenderLine(det.corners[j].x, det.corners[j].y,
                                  det.corners[next].x, det.corners[next].y,
                                  0.0f, 1.0f, 0.0f);
            }

            // Draw tag center as a small rectangle instead of circle
            display->RenderRect(det.center.x - 2, det.center.y - 2, 5, 5, 1.0f, 0.0f, 0.0f);

            // Draw tag ID using RenderLine to create text-like effect
            char str[32];
            sprintf(str, "ID: %d", det.id);
            // Draw a small line to indicate ID position
            display->RenderLine(det.center.x + 10, det.center.y, 
                              det.center.x + 30, det.center.y,
                              1.0f, 0.0f, 0.0f);
        }

        display->EndRender();

        // Update status bar
        char str[256];
        sprintf(str, "vpi-camera-viewer (%ux%u) | %.0f FPS", 
                camera->GetWidth(), camera->GetHeight(), display->GetFPS());
        display->SetTitle(str);

        // Check if the user quit
        if( display->IsClosed() )
            signal_recieved = true;

        // Unlock arrays
        CHECK_STATUS(vpiArrayUnlock(poses));
        CHECK_STATUS(vpiArrayUnlock(detections));
    }

    // Cleanup
    vpiImageDestroy(imgInput);
    vpiImageDestroy(imgGrayscale);
    vpiArrayDestroy(detections);
    vpiArrayDestroy(poses);
    vpiPayloadDestroy(payload);
    vpiStreamDestroy(stream);

    if( display != NULL )
    {
        delete display;
        display = NULL;
    }

    if( camera != NULL )
    {
        delete camera;
        camera = NULL;
    }

    printf("vpi-camera-viewer:  video device '%s' has been un-initialized.\n", dev_path);
    return 0;
} 