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

#include "videoSource.h"
#include "glDisplay.h"
#include "logging.h"
#include "commandLine.h"
#include "glUtility.h"
#include "cudaGrayscale.h"

#include <signal.h>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <math.h>

// VPI includes
//#include <vpi/OpenCVInterop.hpp>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/AprilTags.h>

bool signal_recieved = false;

void sig_handler(int signo)
{
    if( signo == SIGINT )
    {
        LogInfo("received SIGINT\n");
        signal_recieved = true;
    }
}

int usage()
{
    printf("usage: video-viewer-pva [--help] input_URI\n\n");
    printf("View a video or image stream with VPI AprilTag detection.\n");
    printf("See below for additional arguments that may not be shown above.\n\n");
    printf("positional arguments:\n");
    printf("    input_URI       resource URI of input stream  (see videoSource below)\n\n");

    printf("%s", videoSource::Usage());
    printf("%s", Log::Usage());

    return 0;
}

// Define CHECK_STATUS macro
#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

#if 0 // Old method to draw AprilTag detections directly on image.
// Function to draw AprilTag detections
void drawAprilTagDetections(glDisplay* display, VPIAprilTagDetection* detections, VPIPose* poses, int numDetections)
{
    for (int i = 0; i < numDetections; ++i) {
        const VPIAprilTagDetection& det = detections[i];
        const VPIPose& pose = poses[i];

        // Only draw high confidence detections
        if (det.decisionMargin > 30.0f) {
            // Draw lines connecting the four corners of the tag
            display->RenderLine(det.corners[0].x, det.corners[0].y, 
                              det.corners[1].x, det.corners[1].y, 
                              0.9f, 0.0f, 0.0f); // Red line from top-left to top-right
            
            display->RenderLine(det.corners[1].x, det.corners[1].y, 
                              det.corners[2].x, det.corners[2].y, 
                              0.0f, 0.9f, 0.0f); // Green line from top-right to bottom-right

            // Draw tag ID using lines
            char str[32];
            sprintf(str, "ID: %d", det.id);
            
            // Draw a small rectangle for the ID
            float x = det.center.x + 10;
            float y = det.center.y;
            float width = 40;
            float height = 20;
            display->RenderRect(x, y, width, height, 0.0f, 0.0f, 0.0f, 0.5f); // Semi-transparent black background
            display->RenderOutline(x, y, width, height, 0.0f, 1.0f, 0.0f); // Green outline
        }
    }
}
#endif

// Function to draw AprilTag detections
void drawAprilTagDetectionsNew(glDisplay* display, VPIAprilTagDetection* detections, VPIPose* poses, int numDetections, 
                          float fx, float fy, float cx, float cy)
{
    for (int i = 0; i < numDetections; ++i) {
        const VPIAprilTagDetection& det = detections[i];
        const VPIPose& pose = poses[i];

        // Only draw high confidence detections
        if (det.decisionMargin > 30.0f) {
            // Draw complete tag outline
            display->RenderLine(det.corners[0].x, det.corners[0].y, 
                              det.corners[1].x, det.corners[1].y, 
                              0.9f, 0.0f, 0.0f); // Red line from top-left to top-right
            
            display->RenderLine(det.corners[1].x, det.corners[1].y, 
                              det.corners[2].x, det.corners[2].y, 
                              0.0f, 0.9f, 0.0f); // Green line from top-right to bottom-right
            
            //display->RenderLine(det.corners[2].x, det.corners[2].y, 
            //                  det.corners[3].x, det.corners[3].y, 
            //                  0.0f, 0.0f, 0.9f); // Blue line from bottom-right to bottom-left
            
            //display->RenderLine(det.corners[3].x, det.corners[3].y, 
            //                  det.corners[0].x, det.corners[0].y, 
            //                  0.9f, 0.9f, 0.0f); // Yellow line from bottom-left to top-left

            //// Draw center cross using lines
            //float crossSize = 5.0f;
            //display->RenderLine(det.center.x - crossSize, det.center.y,
            //                  det.center.x + crossSize, det.center.y,
            //                  1.0f, 1.0f, 1.0f); // White horizontal line
            //display->RenderLine(det.center.x, det.center.y - crossSize,
            //                  det.center.x, det.center.y + crossSize,
            //                  1.0f, 1.0f, 1.0f); // White vertical line

            // Draw coordinate axes using the pose transformation
            const float axisLength = 0.1f; // Length of coordinate axes in meters (10cm)
            
            // Define 3D points for the coordinate axes (in tag's coordinate system)
            float3 origin = {0.0f, 0.0f, 0.0f};
            float3 xAxis = {axisLength, 0.0f, 0.0f};
            float3 yAxis = {0.0f, axisLength, 0.0f};
            float3 zAxis = {0.0f, 0.0f, -1.0f * axisLength};

            // Define cube corners (in tag's coordinate system)
            // Using the same scale as axisLength for consistency
            float3 cubeCorners[8] = {
                {-axisLength, -axisLength, 0.0f},           // 0: origin
                {axisLength, -axisLength, 0.0f},            // 1: x
                {-axisLength, axisLength, 0.0f},            // 2: y
                {axisLength, axisLength, 0.0f},             // 3: x+y
                {-axisLength, -axisLength, -2.f * axisLength},                               // 4: z
                {axisLength, -axisLength, -2.f * axisLength},     // 5: x+z
                {-axisLength, axisLength, -2.f * axisLength},     // 6: y+z
                {axisLength, axisLength, -2.f * axisLength}       // 7: x+y+z
            };

            // Project 3D points to 2D using the pose transformation and camera intrinsics
            float2 origin2D, xAxis2D, yAxis2D, zAxis2D;
            
            // Helper function to project a 3D point to 2D
            auto projectPoint = [&pose, fx, fy, cx, cy](const float3& point3D) -> float2 {
                // Transform point from tag's coordinate system to camera's coordinate system
                float x = point3D.x * pose.transform[0][0] + point3D.y * pose.transform[0][1] + point3D.z * pose.transform[0][2] + pose.transform[0][3];
                float y = point3D.x * pose.transform[1][0] + point3D.y * pose.transform[1][1] + point3D.z * pose.transform[1][2] + pose.transform[1][3];
                float z = point3D.x * pose.transform[2][0] + point3D.y * pose.transform[2][1] + point3D.z * pose.transform[2][2] + pose.transform[2][3];
                
                // Project to image plane using camera intrinsics
                float u = (x * fx / z) + cx;
                float v = (y * fy / z) + cy;
                
                return {u, v};
            };

            // Project all points
            origin2D = projectPoint(origin);
            xAxis2D = projectPoint(xAxis);
            yAxis2D = projectPoint(yAxis);
            zAxis2D = projectPoint(zAxis);

            // Project cube corners
            float2 cubeCorners2D[8];
            for(int i = 0; i < 8; i++) {
                cubeCorners2D[i] = projectPoint(cubeCorners[i]);
            }

            // Print transformed coordinates
            printf("\nTag %d Pose Debug:\n", det.id);
            printf("Transform Matrix (Tag to Camera):\n");
            printf("[%f %f %f %f]\n", pose.transform[0][0], pose.transform[0][1], pose.transform[0][2], pose.transform[0][3]);
            printf("[%f %f %f %f]\n", pose.transform[1][0], pose.transform[1][1], pose.transform[1][2], pose.transform[1][3]);
            printf("[%f %f %f %f]\n", pose.transform[2][0], pose.transform[2][1], pose.transform[2][2], pose.transform[2][3]);
            printf("\nProjected Points (in pixels):\n");
            printf("Origin: (%.2f, %.2f)\n", origin2D.x, origin2D.y);
            printf("X-axis: (%.2f, %.2f)\n", xAxis2D.x, xAxis2D.y);
            printf("Y-axis: (%.2f, %.2f)\n", yAxis2D.x, yAxis2D.y);
            printf("Z-axis: (%.2f, %.2f)\n", zAxis2D.x, zAxis2D.y);
            printf("Error: %f\n", pose.error);

            // Draw coordinate axes
            display->RenderLine(origin2D.x, origin2D.y, xAxis2D.x, xAxis2D.y, 1.0f, 0.0f, 0.0f); // X-axis in red
            display->RenderLine(origin2D.x, origin2D.y, yAxis2D.x, yAxis2D.y, 0.0f, 1.0f, 0.0f); // Y-axis in green
            display->RenderLine(origin2D.x, origin2D.y, zAxis2D.x, zAxis2D.y, 0.0f, 0.0f, 1.0f); // Z-axis in blue

            // Draw cube edges (12 lines)
            // Bottom face
            display->RenderLine(cubeCorners2D[0].x, cubeCorners2D[0].y, cubeCorners2D[1].x, cubeCorners2D[1].y, 1.0f, 1.0f, 1.0f); // 0-1
            display->RenderLine(cubeCorners2D[1].x, cubeCorners2D[1].y, cubeCorners2D[3].x, cubeCorners2D[3].y, 1.0f, 1.0f, 1.0f); // 1-3
            display->RenderLine(cubeCorners2D[3].x, cubeCorners2D[3].y, cubeCorners2D[2].x, cubeCorners2D[2].y, 1.0f, 1.0f, 1.0f); // 3-2
            display->RenderLine(cubeCorners2D[2].x, cubeCorners2D[2].y, cubeCorners2D[0].x, cubeCorners2D[0].y, 1.0f, 1.0f, 1.0f); // 2-0

            // Top face
            display->RenderLine(cubeCorners2D[4].x, cubeCorners2D[4].y, cubeCorners2D[5].x, cubeCorners2D[5].y, 1.0f, 1.0f, 1.0f); // 4-5
            display->RenderLine(cubeCorners2D[5].x, cubeCorners2D[5].y, cubeCorners2D[7].x, cubeCorners2D[7].y, 1.0f, 1.0f, 1.0f); // 5-7
            display->RenderLine(cubeCorners2D[7].x, cubeCorners2D[7].y, cubeCorners2D[6].x, cubeCorners2D[6].y, 1.0f, 1.0f, 1.0f); // 7-6
            display->RenderLine(cubeCorners2D[6].x, cubeCorners2D[6].y, cubeCorners2D[4].x, cubeCorners2D[4].y, 1.0f, 1.0f, 1.0f); // 6-4

            // Vertical edges
            display->RenderLine(cubeCorners2D[0].x, cubeCorners2D[0].y, cubeCorners2D[4].x, cubeCorners2D[4].y, 1.0f, 1.0f, 1.0f); // 0-4
            display->RenderLine(cubeCorners2D[1].x, cubeCorners2D[1].y, cubeCorners2D[5].x, cubeCorners2D[5].y, 1.0f, 1.0f, 1.0f); // 1-5
            display->RenderLine(cubeCorners2D[2].x, cubeCorners2D[2].y, cubeCorners2D[6].x, cubeCorners2D[6].y, 1.0f, 1.0f, 1.0f); // 2-6
            display->RenderLine(cubeCorners2D[3].x, cubeCorners2D[3].y, cubeCorners2D[7].x, cubeCorners2D[7].y, 1.0f, 1.0f, 1.0f); // 3-7
        }
    }
}

int main(int argc, char** argv)
{
    // Parse command line
    commandLine cmdLine(argc, argv);

    if (cmdLine.GetFlag("help"))
        return usage();

    // Attach signal handler
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        LogError("can't catch SIGINT\n");

	// Check if user specified input resolution
	if( !cmdLine.GetFlag("input-width") && !cmdLine.GetFlag("input-height") )
	{
		// Set default resolution if not specified
		// This is imx296 native resolution.
		cmdLine.AddArg("--input-width=1456");  // Set default width
		cmdLine.AddArg("--input-height=1088"); // Set default height
		LogInfo("Using default camera resolution: 1456x1088\n");
	}

    // Create input video stream
    videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

    if (!input) {
        LogError("video-viewer-pva: failed to create input stream\n");
        return 0;
    }

    // Dry-run the input stream to get the image size
    //uchar3* image = NULL;
    uint8_t* image = NULL;
    int status = 0;
    if (!input->Capture(&image, &status)) {
        LogError("video-viewer-pva: failed to capture input image\n");
        SAFE_DELETE(input);
        return 0;
    }

    // Create OpenGL window
    videoOptions displayOpts;
    displayOpts.resource = "display://0";
    displayOpts.width = input->GetWidth();
    displayOpts.height = input->GetHeight();
    
    glDisplay* display = glDisplay::Create(displayOpts);
    
    if (!display) {
        LogError("video-viewer-pva: failed to create openGL display\n");
        return 0;
    }

	// Allocate pitched memory for grayscale image
	uint8_t* image_dev = nullptr;
	size_t gray_pitch = 0;
	if( CUDA_FAILED(cudaMallocPitch(&image_dev, &gray_pitch, input->GetWidth() * sizeof(uint8_t), input->GetHeight())) )
	{
		LogError("failed to allocate pitched GPU memory for grayscale image\n");
		SAFE_DELETE(input);
		return 0;
	}

    // Initialize VPI resources
    VPIImage imgGrayscale = NULL;
    VPIArray detections = NULL;
    VPIArray poses = NULL;
    VPIStream stream = NULL;
    VPIPayload payload = NULL;

    try {
        // Create VPI stream
        CHECK_STATUS(vpiStreamCreate(0, &stream));

        // Create VPI images directly from RGB8 format
        CHECK_STATUS(vpiImageCreate(input->GetWidth(), input->GetHeight(), VPI_IMAGE_FORMAT_U8, 0, &imgGrayscale));

        // Create detection and pose arrays
        const int maxDetections = 64;
        CHECK_STATUS(vpiArrayCreate(maxDetections, VPI_ARRAY_TYPE_APRILTAG_DETECTION, VPI_BACKEND_CPU | VPI_BACKEND_PVA, &detections));
        CHECK_STATUS(vpiArrayCreate(maxDetections, VPI_ARRAY_TYPE_POSE, VPI_BACKEND_CPU | VPI_BACKEND_PVA, &poses));

        // CPU backend is actually faster than PVA backend.
        // On Orin NX 16GB. CPU backend takes ~6ms, PVA backend takes ~8.6ms.
        //auto backend = VPI_BACKEND_CPU;
        auto backend = VPI_BACKEND_PVA;
        auto strBackend = "cpu";
        if (backend == VPI_BACKEND_PVA) {
            strBackend = "pva";
        }
        LogInfo("Using backend: %s\n", strBackend);

        // Create AprilTag detector payload
        VPIAprilTagDecodeParams apritagDecodeParams = {NULL, 0, 1, VPI_APRILTAG_36H11};
        CHECK_STATUS(vpiCreateAprilTagDetector(backend, input->GetWidth(), input->GetHeight(), &apritagDecodeParams, &payload));

        // AprilTag pose estimation parameters
        const VPICameraIntrinsic intrinsics = {
            {input->GetWidth() / 3.5f, 0.0f, input->GetWidth() / 2.f},
            {0.0f, input->GetHeight() / 3.6f, input->GetHeight() / 2.f}
        };
        const float tagSize = 0.2f;

        // Main processing loop
        uint32_t numFrames = 0;
        uint32_t framesWithTags = 0;
        uint32_t totalTagsDetected = 0;

        while (!signal_recieved) {
            // Capture next frame
            if (!input->Capture(&image, &status)) {
                if (status == videoSource::TIMEOUT)
                    continue;
                break;
            }

            numFrames++;

            // Import grayscale data directly into VPI image
            VPIImageData imgData;
            CHECK_STATUS(vpiImageLockData(imgGrayscale, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData));

            // Map data from CUDA memory to VPI image
            uint8_t* vpiData = (uint8_t*)imgData.buffer.pitch.planes[0].data;
            size_t vpiPitch = imgData.buffer.pitch.planes[0].pitchBytes;

		    // Copy input image to GPU with pitch
            // TODO: FIXME: Use remap() kernel to replace the cuda memcpy.
		    if( CUDA_FAILED(cudaMemcpy2D(vpiData, gray_pitch, image, input->GetWidth() * sizeof(uint8_t), input->GetWidth() * sizeof(uint8_t), input->GetHeight(), cudaMemcpyDeviceToDevice)) )
		    {
		    	LogError("failed to copy grayscale image to GPU\n");
		    	break;
		    }
	
            CHECK_STATUS(vpiImageUnlock(imgGrayscale));

            // Detect AprilTags
            CHECK_STATUS(vpiSubmitAprilTagDetector(stream, backend, payload, maxDetections, imgGrayscale, detections));

            // Estimate poses // Only CPU backend is supported for pose estimation.
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

            if (display != NULL) {
                display->BeginRender();

                // Render grayscale image
                display->RenderImage(vpiData, vpiPitch, input->GetHeight(), IMAGE_GRAY8, 0.0f, 0.0f);

                // Draw detections
                if (numDetections > 0) {
                    framesWithTags++;
                    totalTagsDetected += numDetections;
                    drawAprilTagDetectionsNew(display, outDetections, outPoses, numDetections, input->GetWidth() / 3.5f, input->GetHeight() / 3.6f, input->GetWidth() / 2.0f, input->GetHeight() / 2.0f);
                }

                display->EndRender();

                // Update status bar
                char str[256];
                sprintf(str, "Video Viewer PVA (%ux%u) | %.1f FPS | Tags: %d | Total Tags: %d", 
                    input->GetWidth(), input->GetHeight(), display->GetFPS(), 
                    numDetections, totalTagsDetected);
                display->SetTitle(str);

                if (display->IsClosed())
                    break;
            }

            // Unlock arrays
            CHECK_STATUS(vpiArrayUnlock(poses));
            CHECK_STATUS(vpiArrayUnlock(detections));
        }

        // Clean up VPI resources
        vpiImageDestroy(imgGrayscale);
        vpiArrayDestroy(detections);
        vpiArrayDestroy(poses);
        vpiPayloadDestroy(payload);
        vpiStreamDestroy(stream);

        // Print final statistics
        printf("\nFinal detection statistics:\n");
        printf("  Total frames processed: %u\n", numFrames);
        printf("  Frames with tags detected: %u (%.1f%%)\n", 
            framesWithTags, (float)framesWithTags/numFrames*100.0f);
        printf("  Total tags detected: %u\n", totalTagsDetected);

    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    SAFE_DELETE(input);
    SAFE_DELETE(display);

    printf("video-viewer-pva: shutdown complete\n");
    return 0;
} 
