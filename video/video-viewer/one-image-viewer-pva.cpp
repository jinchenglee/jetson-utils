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
#include "videoOutput.h"
#include "logging.h"
#include "commandLine.h"
#include "cudaGrayscale.h"

#include <signal.h>
#include <cuda.h>
#include <iostream>
#include <sstream>

// AprilTag includes
extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
#include "tag16h5.h"
#include "apriltag_pose.h"
}

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
    printf("usage: one-image-viewer-pva [--help] input_image [output_image]\n\n");
    printf("Process a single image with VPI AprilTag detection.\n");
    printf("See below for additional arguments that may not be shown above.\n\n");
    printf("positional arguments:\n");
    printf("    input_image     path to input image file (jpg, png, etc.)\n");
    printf("    output_image    path to save the processed image (optional)\n\n");

    printf("%s", videoSource::Usage());
    printf("%s", videoOutput::Usage());
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

// Function to draw AprilTag detections
void drawAprilTagDetections(uchar3* image, int width, int height, VPIAprilTagDetection* detections, VPIPose* poses, int numDetections)
{
    for (int i = 0; i < numDetections; ++i) {
        const VPIAprilTagDetection& det = detections[i];
        const VPIPose& pose = poses[i];

        // Only draw high confidence detections
        if (det.decisionMargin > 30.0f) {
            // Draw lines connecting the four corners of the tag
            for (int j = 0; j < 4; j++) {
                int k = (j + 1) % 4;
                int x1 = static_cast<int>(det.corners[j].x);
                int y1 = static_cast<int>(det.corners[j].y);
                int x2 = static_cast<int>(det.corners[k].x);
                int y2 = static_cast<int>(det.corners[k].y);

                // Draw line using Bresenham's line algorithm
                int dx = abs(x2 - x1);
                int dy = abs(y2 - y1);
                int sx = (x1 < x2) ? 1 : -1;
                int sy = (y1 < y2) ? 1 : -1;
                int err = dx - dy;

                while (true) {
                    // Draw point in red with thickness
                    for (int t = -1; t <= 1; t++) {
                        for (int s = -1; s <= 1; s++) {
                            int px = x1 + t;
                            int py = y1 + s;
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                image[py * width + px].x = 255;  // Red
                                image[py * width + px].y = 0;    // Green
                                image[py * width + px].z = 0;    // Blue
                            }
                        }
                    }

                    if (x1 == x2 && y1 == y2) break;

                    int e2 = 2 * err;
                    if (e2 > -dy) {
                        err -= dy;
                        x1 += sx;
                    }
                    if (e2 < dx) {
                        err += dx;
                        y1 += sy;
                    }
                }
            }
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

    // Create input image source
    videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

    if (!input) {
        LogError("one-image-viewer-pva: failed to create input image source\n");
        return 0;
    }

    // Create output image writer
    videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));

    if (!output) {
        LogError("one-image-viewer-pva: failed to create output image writer\n");
        return 0;
    }

    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        LogError("Failed to initialize CUDA device: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    // Print CUDA device properties
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus == cudaSuccess) {
        LogInfo("CUDA Device Properties:\n");
        LogInfo("  Device Name: %s\n", deviceProp.name);
        LogInfo("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    }

    // Capture input image
    uchar3* image = NULL;
    int status = 0;

    if (!input->Capture(&image, &status)) {
        LogError("one-image-viewer-pva: failed to capture input image\n");
        SAFE_DELETE(input);
        SAFE_DELETE(output);
        return 0;
    }

    // Initialize VPI resources
    VPIImage imgInput = NULL;
    VPIImage imgGrayscale = NULL;
    VPIArray detections = NULL;
    VPIArray poses = NULL;
    VPIStream stream = NULL;
    VPIPayload payload = NULL;

    try {
        // Create VPI stream
        CHECK_STATUS(vpiStreamCreate(0, &stream));

        // Create VPI images
        CHECK_STATUS(vpiImageCreate(input->GetWidth(), input->GetHeight(), VPI_IMAGE_FORMAT_RGB8, 0, &imgInput));
        CHECK_STATUS(vpiImageCreate(input->GetWidth(), input->GetHeight(), VPI_IMAGE_FORMAT_U8, 0, &imgGrayscale));

        // Create detection and pose arrays
        const int maxDetections = 64;
        CHECK_STATUS(vpiArrayCreate(maxDetections, VPI_ARRAY_TYPE_APRILTAG_DETECTION, VPI_BACKEND_CPU | VPI_BACKEND_PVA, &detections));
        CHECK_STATUS(vpiArrayCreate(maxDetections, VPI_ARRAY_TYPE_POSE, VPI_BACKEND_CPU | VPI_BACKEND_PVA, &poses));

        auto backend = VPI_BACKEND_CPU;
        //auto backend = VPI_BACKEND_PVA;
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

        // Initialize CUDA memory for grayscale conversion
        uint8_t* img_gray8_dev = nullptr;
        size_t pitch = 0;
        if (CUDA_FAILED(cudaMallocPitch(&img_gray8_dev, &pitch, input->GetWidth(), input->GetHeight()))) {
            LogError("Failed to allocate CUDA memory for grayscale image\n");
            return 0;
        }

	// Create AprilTag image structure
	image_u8_t im = {
		.width = static_cast<int32_t>(input->GetWidth()),
		.height = static_cast<int32_t>(input->GetHeight()),
		.stride = static_cast<int32_t>(input->GetWidth()),
		.buf = new uint8_t[input->GetWidth() * input->GetHeight()]
	};

	/*
	 * Process single image
	 */

	// Convert RGB to grayscale
	for(int y = 0; y < im.height; y++) {
		for(int x = 0; x < im.width; x++) {
			uchar3 pixel = image[y * im.width + x];
			im.buf[y * im.stride + x] = (uint8_t)((pixel.x * 0.299 + pixel.y * 0.587 + pixel.z * 0.114));
		}
	}

	// Copy grayscale image to device with pitch
	cudaMemcpy2D(img_gray8_dev, pitch, im.buf, im.stride, im.width, im.height, cudaMemcpyHostToDevice);


#if 0

        // Save grayscale image for debugging
        videoOptions grayOpts;
        grayOpts.resource = "file://grayscale.jpg";
        grayOpts.width = input->GetWidth();
        grayOpts.height = input->GetHeight();
        
        videoOutput* grayOutput = videoOutput::Create(grayOpts);
        if (grayOutput != NULL) {

			// Convert the grayscale remapped image back to RGB for saving
			image_u8_t debugImage = {
				.width = static_cast<int32_t>(input->GetWidth()),
				.height = static_cast<int32_t>(input->GetHeight()),
				.stride = static_cast<int32_t>(input->GetWidth()),
				.buf = new uint8_t[input->GetWidth() * input->GetHeight()]
			};
            memset(debugImage.buf, 0, input->GetWidth() * input->GetHeight() * sizeof(uint8_t));

            if (CUDA_FAILED(cudaMemcpy2D(
                debugImage.buf,                    // destination
                input->GetWidth(),                  // destination pitch
                img_gray8_dev,             // source
                pitch,                     // source pitch
                input->GetWidth(),         // width in bytes
                input->GetHeight(),        // height
                cudaMemcpyDeviceToHost))) {
                LogError("Failed to copy grayscale data to VPI image\n");
                return 0;
            }

			// Convert the grayscale remapped image back to RGB for saving
			uchar3* remapImage = new uchar3[input->GetWidth() * input->GetHeight()];
			for(int y = 0; y < input->GetHeight(); y++) {
				for(int x = 0; x < input->GetWidth(); x++) {
					uint8_t gray = debugImage.buf[y * debugImage.stride + x];
					remapImage[y * input->GetWidth() + x].x = gray;  // R
					remapImage[y * input->GetWidth() + x].y = gray;  // G
					remapImage[y * input->GetWidth() + x].z = gray;  // B
				}
			}

            // Save the grayscale image
            grayOutput->Render(remapImage, input->GetWidth(), input->GetHeight());
            LogInfo("Saved grayscale image for debugging\n");
            
            // Clean up
            SAFE_DELETE(grayOutput);
        }
#endif


        // Import grayscale data directly into VPI image
        VPIImageData imgData;
        CHECK_STATUS(vpiImageLockData(imgGrayscale, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData));
        
        // Map data from CUDA memory to VPI image
        uint8_t* vpiData = (uint8_t*)imgData.buffer.pitch.planes[0].data;
        size_t vpiPitch = imgData.buffer.pitch.planes[0].pitchBytes;
        
        // Debug: Print pitch information
        LogInfo("Copy debug info:\n");
        LogInfo("  Source pitch: %zu bytes\n", pitch);
        LogInfo("  VPI pitch: %zu bytes\n", vpiPitch);
        LogInfo("  Image width: %d pixels (%d bytes)\n", input->GetWidth(), input->GetWidth() * sizeof(uint8_t));
        
        // Copy the entire image at once using cudaMemcpy2D
        if (CUDA_FAILED(cudaMemcpy2D(
            vpiData,                    // destination
            vpiPitch,                  // destination pitch
            img_gray8_dev,             // source
            pitch,                     // source pitch
            input->GetWidth(),         // width in bytes
            input->GetHeight(),        // height
            cudaMemcpyDeviceToDevice))) {
            LogError("Failed to copy grayscale data to VPI image\n");
            return 0;
        }
        CHECK_STATUS(vpiImageUnlock(imgGrayscale));

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

        // Print detection statistics
        LogInfo("Detection statistics:\n");
        LogInfo("  Number of tags detected: %d\n", numDetections);

        // Save the processed image if output is specified
        if (output != NULL) {
            // Create a new output for the processed image
            videoOptions processedOpts;
            processedOpts.resource = "file://processed.jpg";
            processedOpts.width = input->GetWidth();
            processedOpts.height = input->GetHeight();
            
            videoOutput* processedOutput = videoOutput::Create(processedOpts);
            if (processedOutput != NULL) {
                // Create a copy of the input image for drawing
                uchar3* outputImage = new uchar3[input->GetWidth() * input->GetHeight()];
                if (CUDA_FAILED(cudaMemcpy(outputImage, image, input->GetWidth() * input->GetHeight() * sizeof(uchar3), cudaMemcpyDeviceToHost))) {
                    LogError("Failed to copy image from GPU to CPU memory\n");
                    delete[] outputImage;
                    SAFE_DELETE(processedOutput);
                    return 0;
                }

                // Draw detections on the image
                if (numDetections > 0) {
                    drawAprilTagDetections(outputImage, input->GetWidth(), input->GetHeight(), outDetections, outPoses, numDetections);
                }

                // Save the image with detections
                processedOutput->Render(outputImage, input->GetWidth(), input->GetHeight());
                LogInfo("Saved processed image with detections\n");

                // Clean up
                delete[] outputImage;
                SAFE_DELETE(processedOutput);
            }
        }

        // Unlock arrays
        CHECK_STATUS(vpiArrayUnlock(poses));
        CHECK_STATUS(vpiArrayUnlock(detections));

        // Clean up CUDA resources
        cudaFree(img_gray8_dev);

        // Clean up VPI resources
        vpiImageDestroy(imgInput);
        vpiImageDestroy(imgGrayscale);
        vpiArrayDestroy(detections);
        vpiArrayDestroy(poses);
        vpiPayloadDestroy(payload);
        vpiStreamDestroy(stream);

    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    SAFE_DELETE(input);
    SAFE_DELETE(output);

    printf("one-image-viewer-pva: shutdown complete\n");
    return 0;
} 
