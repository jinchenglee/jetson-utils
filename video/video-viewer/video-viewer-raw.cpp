/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "glUtility.h"  // For GL drawing functions
#include "mapxpy.h"     // Add include for mapxpy
#include "NvAnalysis.h" // Add include for remap function
#include "cudaGrayscale.h" // Add include for cudaRGB8ToGray8

#include <signal.h>
#include <cuda.h>

// AprilTag includes
extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
#include "tag16h5.h"
#include "apriltag_pose.h"
#include "apriltag_math.h"  // For pose estimation
}

// Global variables for AprilTag
apriltag_detector_t* tag_detector = NULL;
apriltag_family_t* tag_family = NULL;

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
	printf("usage: video-viewer [--help] [--tag-family=36h11|16h5] input_URI\n\n");
	printf("View a video or image stream with AprilTag detection.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n\n");
	printf("optional arguments:\n");
	printf("    --tag-family    AprilTag family to detect (36h11 or 16h5, default: 36h11)\n\n");

	printf("%s", videoSource::Usage());
	printf("%s", Log::Usage());

	return 0;
}

// Function to initialize AprilTag detector with specified family
bool init_apriltag_detector(const char* family_name)
{
	// Clean up existing detector if any
	if(tag_detector != NULL) {
		apriltag_detector_destroy(tag_detector);
		tag_detector = NULL;
	}
	if(tag_family != NULL) {
		if(strcmp(family_name, "36h11") == 0)
			tag36h11_destroy(tag_family);
		else if(strcmp(family_name, "16h5") == 0)
			tag16h5_destroy(tag_family);
		tag_family = NULL;
	}

	// Create new detector
	tag_detector = apriltag_detector_create();
	if(!tag_detector) {
		LogError("Failed to create AprilTag detector\n");
		return false;
	}

	// Create and add tag family
	if(strcmp(family_name, "36h11") == 0) {
		tag_family = tag36h11_create();
	} else if(strcmp(family_name, "16h5") == 0) {
		tag_family = tag16h5_create();
	} else {
		LogError("Unsupported tag family: %s\n", family_name);
		apriltag_detector_destroy(tag_detector);
		tag_detector = NULL;
		return false;
	}

	apriltag_detector_add_family(tag_detector, tag_family);
	
	// Configure detector
	tag_detector->quad_decimate = 2.0;
	tag_detector->quad_sigma = 0.0;
	tag_detector->nthreads = 8;
	tag_detector->debug = false;
	tag_detector->refine_edges = true;

	return true;
}

int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();

	/*
	 * attach signal handler
	 */	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
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

	/*
	 * create input video stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("video-viewer:  failed to create input stream\n");
		return 0;
	}

	// Dry-run the input stream to get the image size
	uint8_t* image = NULL;
	int status = 0;
	if( !input->Capture(&image, &status) )
	{
		LogError("video-viewer:  failed to capture input image\n");
		SAFE_DELETE(input);
		return 0;
	}

	/*
	 * create openGL window
	 */
	videoOptions displayOpts;
	displayOpts.resource = "display://0";
	displayOpts.width = input->GetWidth();
	displayOpts.height = input->GetHeight();
	
	glDisplay* display = glDisplay::Create(displayOpts);
	
	if( !display )
	{
		LogError("video-viewer:  failed to create openGL display\n");
		return 0;
	}

	/*
	 * Initialize AprilTag detector
	 */
	const char* tag_family_name = cmdLine.GetString("tag-family", "36h11");
	if(!init_apriltag_detector(tag_family_name)) {
		LogError("Failed to initialize AprilTag detector\n");
		SAFE_DELETE(input);
		return 0;
	}

	LogInfo("Using AprilTag family: %s\n", tag_family_name);

	// Initialize CUDA memory for remapping
	uint8_t* img_dst_gray8_dev = nullptr;
	float *mapxDevPtr, *mapyDevPtr;
	size_t sizeOfImage = input->GetWidth() * input->GetHeight();
	size_t pitch = 0;

	// Allocate pitched memory for image processing
	if( CUDA(cudaMallocPitch(&img_dst_gray8_dev, &pitch, input->GetWidth() * sizeof(uint8_t), input->GetHeight() * sizeof(uint8_t))) )
		LogError("cudaMalloc img_dst_gray8_dev failed!\n");
	
	// Allocate memory for mapx and mapy
	if( CUDA(cudaMalloc(&mapxDevPtr, sizeOfImage * sizeof(float))) )
		LogError("cudaMalloc mapxDevPtr failed!\n");
	if( CUDA(cudaMalloc(&mapyDevPtr, sizeOfImage * sizeof(float))) )
		LogError("cudaMalloc mapyDevPtr failed!\n");

	// Copy mapx mapy to device memory
	cudaMemcpy(mapxDevPtr, mapx, sizeOfImage * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(mapyDevPtr, mapy, sizeOfImage * sizeof(float), cudaMemcpyHostToDevice);

	// Create reusable grayscale image buffer
	image_u8_t im = {
		.width = static_cast<int32_t>(input->GetWidth()),
		.height = static_cast<int32_t>(input->GetHeight()),
		.stride = static_cast<int32_t>(input->GetWidth()),
		.buf = new uint8_t[input->GetWidth() * input->GetHeight()]
	};

	/*
	 * capture/display loop
	 */
	uint32_t numFrames = 0;
	uint32_t framesWithTags = 0;
	uint32_t framesWithHighConfidenceTags = 0;
	uint32_t totalTagsDetected = 0;
	uint32_t highConfidenceTagsDetected = 0;

	while( !signal_recieved )
	{
		if( !input->Capture(&image, &status) )
		{
			if( status == videoSource::TIMEOUT )
				continue;
			
			break; // EOS
		}

		// Apply remap for undistortion on NV12 input Y/Luma component directly
		remap((uint8_t*)image, img_dst_gray8_dev, mapxDevPtr, mapyDevPtr, im.width, im.width, im.height);
		cudaDeviceSynchronize();

		// copy over to cpu side
		cudaError_t err = cudaMemcpy2D(im.buf, 
			im.width,           // destination pitch
			img_dst_gray8_dev,
			im.width,           // source pitch (notice here we use im.width not pitch) 
			im.width,           // width to copy (use full pitch)
			im.height,          // height to copy
			cudaMemcpyDeviceToHost);
		//printf("CUDA memcpy error: %s\n", cudaGetErrorString(err));

		// Detect AprilTags
		zarray_t* detections = apriltag_detector_detect(tag_detector, &im);

		// Print detection statistics every 90 frames
		if( numFrames % 90 == 0 )
		{
			printf("\nDetection statistics for frame %u:\n", numFrames);
			printf("  Number of edges detected: %d\n", tag_detector->nedges);
			printf("  Number of segments detected: %d\n", tag_detector->nsegments);
			printf("  Number of quads detected: %d\n", tag_detector->nquads);
			printf("  Number of tags detected: %d\n", zarray_size(detections));
			printf("  Frames with tags detected: %u/%u (%.1f%%)\n", 
				framesWithTags, numFrames, (float)framesWithTags/numFrames*100.0f);
			printf("  Frames with high confidence tags: %u/%u (%.1f%%)\n",
				framesWithHighConfidenceTags, numFrames, (float)framesWithHighConfidenceTags/numFrames*100.0f);
			printf("  Total tags detected: %u\n", totalTagsDetected);
			printf("  High confidence tags: %u (%.1f%%)\n",
				highConfidenceTagsDetected, totalTagsDetected > 0 ? (float)highConfidenceTagsDetected/totalTagsDetected*100.0f : 0.0f);
			
			// Print timing information
			timeprofile_display(tag_detector->tp);
		}

		if( display != NULL )
		{
			// Begin OpenGL rendering
			display->BeginRender();

			// Below render code requries an image on the device.
			// Passing img on host caused panic error. 
			// Also, please notice the pitch is used to pass in the image.
			// 
			// The RenderImage() function has a default y offset of 30.0f. <= hard to debug if not noticed. Explicitly set to 0.0f offsets.
			display->RenderImage(img_dst_gray8_dev, im.width, im.height, IMAGE_GRAY8, 0.0f, 0.0f);

			// Draw detections on the remapped image
			int numTags = zarray_size(detections);
			if( numTags > 0 )
			{
				framesWithTags++;
				bool hasHighConfidenceTag = false;
				
				for(int i = 0; i < numTags; i++) {
					apriltag_detection_t* det;
					zarray_get(detections, i, &det);
					
					totalTagsDetected++;

					if( det->decision_margin > 30.0f )
					{
						highConfidenceTagsDetected++;
						hasHighConfidenceTag = true;

						// Draw lines connecting the four corners of the tag
						display->RenderLine(det->p[0][0], det->p[0][1], det->p[1][0], det->p[1][1], 0.9f, 0.0f, 0.0f); // Red line from top-left to top-right
						display->RenderLine(det->p[3][0], det->p[3][1], det->p[0][0], det->p[0][1], 0.0f, 0.9f, 0.0f); // Green line from top-right to bottom-right
					}
					
					// Print tag info for each detection
					if( numFrames % 90 == 0 )
					{
						printf("  Tag %d: id=%d, hamming=%d, margin=%.3f\n", 
							i, det->id, det->hamming, det->decision_margin);
					}
				}
				
				if( hasHighConfidenceTag )
					framesWithHighConfidenceTags++;
			}

			// End OpenGL rendering
			display->EndRender();

			// update status bar
			char str[256];
			sprintf(str, "Video Viewer (%ux%u) | %.1f FPS | Tags: %d/%d | Tag Family: %s", 
				input->GetWidth(), input->GetHeight(), display->GetFPS(), 
				highConfidenceTagsDetected, totalTagsDetected, tag_family_name);
			display->SetTitle(str);    

			// check if the user quit
			if( display->IsClosed() )
				break;
		}

		// Clean up detections
		apriltag_detections_destroy(detections);

		numFrames++;
	}

	/*
	 * destroy resources
	 */
	printf("video-viewer:  shutting down...\n");
	printf("Final detection statistics:\n");
	printf("  Total frames processed: %u\n", numFrames);
	printf("  Frames with tags detected: %u (%.1f%%)\n", 
		framesWithTags, (float)framesWithTags/numFrames*100.0f);
	printf("  Frames with high confidence tags: %u (%.1f%%)\n",
		framesWithHighConfidenceTags, (float)framesWithHighConfidenceTags/numFrames*100.0f);
	printf("  Total tags detected: %u\n", totalTagsDetected);
	printf("  High confidence tags: %u (%.1f%%)\n",
		highConfidenceTagsDetected, totalTagsDetected > 0 ? (float)highConfidenceTagsDetected/totalTagsDetected*100.0f : 0.0f);
	
	// Clean up CUDA resources
	cudaFree(img_dst_gray8_dev);
	cudaFree(mapxDevPtr);
	cudaFree(mapyDevPtr);

	// Clean up AprilTag resources
	apriltag_detector_destroy(tag_detector);
	if(strcmp(tag_family_name, "36h11") == 0)
		tag36h11_destroy(tag_family);
	else if(strcmp(tag_family_name, "16h5") == 0)
		tag16h5_destroy(tag_family);
	
	// Clean up image buffers
	delete[] im.buf;
	
	SAFE_DELETE(input);
	SAFE_DELETE(display);

	printf("video-viewer:  shutdown complete\n");
}

