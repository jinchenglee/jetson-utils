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
#include "videoOutput.h"

#include "logging.h"
#include "commandLine.h"
#include "mapxpy.h"
#include "NvAnalysis.h"

#include <signal.h>
#include <cuda.h>

// AprilTag includes
extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
#include "tag16h5.h"
#include "apriltag_pose.h"
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
	printf("usage: one-image-viewer [--help] [--tag-family=36h11|16h5] input_image [output_image]\n\n");
	printf("Process a single image with AprilTag detection.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("positional arguments:\n");
	printf("    input_image     path to input image file (jpg, png, etc.)\n");
	printf("    output_image    path to save the processed image (optional)\n\n");
	printf("optional arguments:\n");
	printf("    --tag-family    AprilTag family to detect (36h11 or 16h5, default: 36h11)\n\n");

	printf("%s", videoSource::Usage());
	printf("%s", videoOutput::Usage());
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
	tag_detector->nthreads = 4;
	tag_detector->debug = true;  // Enable debug mode
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

	/*
	 * create input image source
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("one-image-viewer:  failed to create input image source\n");
		return 0;
	}

	/*
	 * create output image writer
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
	{
		LogError("one-image-viewer:  failed to create output image writer\n");
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

	/*
	 * Initialize AprilTag detector
	 */
	const char* tag_family_name = cmdLine.GetString("tag-family", "36h11");
	if(!init_apriltag_detector(tag_family_name)) {
		LogError("Failed to initialize AprilTag detector\n");
		SAFE_DELETE(input);
		SAFE_DELETE(output);
		return 0;
	}

	LogInfo("Using AprilTag family: %s\n", tag_family_name);

	uchar3* image = NULL;
	int status = 0;

	if( !input->Capture(&image, &status) )
	{
		LogError("one-image-viewer:  failed to capture input image\n");
		SAFE_DELETE(input);
		SAFE_DELETE(output);
		return 0;
	}

	// Initialize CUDA memory for remapping
	uint8_t* img_src_dev = nullptr;
	uint8_t* img_dst_dev = nullptr;
	float *mapxDevPtr, *mapyDevPtr;
	size_t sizeOfImage = input->GetWidth() * input->GetHeight();
	size_t pitch = 0;

	// Allocate pitched memory for image processing
	if( CUDA(cudaMallocPitch(&img_src_dev, &pitch, input->GetWidth() * sizeof(uint8_t), input->GetHeight() * sizeof(uint8_t))) )
		LogError("cudaMalloc img_src_dev failed!\n");
	if( CUDA(cudaMallocPitch(&img_dst_dev, &pitch, input->GetWidth() * sizeof(uint8_t), input->GetHeight() * sizeof(uint8_t))) )
		LogError("cudaMalloc img_dst_dev failed!\n");
	
	// Allocate memory for mapx and mapy
	if( CUDA(cudaMalloc(&mapxDevPtr, sizeOfImage * sizeof(float))) )
		LogError("cudaMalloc mapxDevPtr failed!\n");
	if( CUDA(cudaMalloc(&mapyDevPtr, sizeOfImage * sizeof(float))) )
		LogError("cudaMalloc mapyDevPtr failed!\n");

	LogInfo("Processing image: %ux%u, pitch: %zu, elements of image: %zu\n", input->GetWidth(), input->GetHeight(), pitch, sizeOfImage);

	// Allocate CUDA memory for image processing
	// Copy mapx mapy to device memory
	cudaMemcpy(mapxDevPtr, mapx, sizeOfImage * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(mapyDevPtr, mapy, sizeOfImage * sizeof(float), cudaMemcpyHostToDevice);

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
	cudaMemcpy2D(img_src_dev, pitch, im.buf, im.stride, im.width, im.height, cudaMemcpyHostToDevice);

	//// Apply remap for undistortion
	remap(img_src_dev, img_dst_dev, mapxDevPtr, mapyDevPtr, pitch, im.width, im.height);
	cudaDeviceSynchronize();

	//// Clear the image buffer
	//for(int y = 0; y < im.height; y++) {
	//	for(int x = 0; x < im.width; x++) {
	//		im.buf[y * im.stride + x] = (uint8_t)0;
	//	}
	//}

	// Copy processed image back to host with pitch
	cudaMemcpy2D(im.buf, im.stride, img_dst_dev, pitch, im.width, im.height, cudaMemcpyDeviceToHost);

	// Detect AprilTags
	zarray_t* detections = apriltag_detector_detect(tag_detector, &im);

	// Print detection statistics
	LogInfo("Detection statistics:\n");
	LogInfo("  Number of edges detected: %d\n", tag_detector->nedges);
	LogInfo("  Number of segments detected: %d\n", tag_detector->nsegments);
	LogInfo("  Number of quads detected: %d\n", tag_detector->nquads);
	LogInfo("  Number of tags detected: %d\n", zarray_size(detections));

	// Print timing information
	timeprofile_display(tag_detector->tp);

	// Save the processed image
	if( output != NULL )
	{
		// Create a new output for the remapped image
		videoOptions remapOpts;
		remapOpts.resource = "file://remapped.jpg";
		remapOpts.width = input->GetWidth();
		remapOpts.height = input->GetHeight();
		
		videoOutput* remapOutput = videoOutput::Create(remapOpts);
		if( remapOutput != NULL )
		{
			// Convert the grayscale remapped image back to RGB for saving
			uchar3* remapImage = new uchar3[input->GetWidth() * input->GetHeight()];
			for(int y = 0; y < input->GetHeight(); y++) {
				for(int x = 0; x < input->GetWidth(); x++) {
					uint8_t gray = im.buf[y * im.stride + x];
					remapImage[y * input->GetWidth() + x].x = gray;  // R
					remapImage[y * input->GetWidth() + x].y = gray;  // G
					remapImage[y * input->GetWidth() + x].z = gray;  // B
				}
			}

			// Draw detections on the remapped image
			for(int i = 0; i < zarray_size(detections); i++) {
				apriltag_detection_t* det;
				zarray_get(detections, i, &det);

				// Print detailed information about each detection
				LogInfo("Tag %d:\n", i);
				LogInfo("  ID: %d\n", det->id);
				LogInfo("  Hamming distance: %d\n", det->hamming);
				LogInfo("  Decision margin: %.3f\n", det->decision_margin);
				LogInfo("  Center: (%.1f, %.1f)\n", det->c[0], det->c[1]);
				LogInfo("  Corners: (%.1f, %.1f), (%.1f, %.1f), (%.1f, %.1f), (%.1f, %.1f)\n",
					det->p[0][0], det->p[0][1],
					det->p[1][0], det->p[1][1],
					det->p[2][0], det->p[2][1],
					det->p[3][0], det->p[3][1]);

				// Draw tag outline on the remapped image
				for(int j = 0; j < 4; j++) {
					int k = (j + 1) % 4;
					// Draw line between corners
					int x1 = static_cast<int>(det->p[j][0]);
					int y1 = static_cast<int>(det->p[j][1]);
					int x2 = static_cast<int>(det->p[k][0]);
					int y2 = static_cast<int>(det->p[k][1]);
					
					// Draw line using Bresenham's line algorithm
					int dx = abs(x2 - x1);
					int dy = abs(y2 - y1);
					int sx = (x1 < x2) ? 1 : -1;
					int sy = (y1 < y2) ? 1 : -1;
					int err = dx - dy;
					
					while(true) {
						// Draw point in red with thickness
						for(int t = -1; t <= 1; t++) {
							for(int s = -1; s <= 1; s++) {
								int px = x1 + t;
								int py = y1 + s;
								if(px >= 0 && px < input->GetWidth() && py >= 0 && py < input->GetHeight()) {
									remapImage[py * input->GetWidth() + px].x = 255;  // Red
									remapImage[py * input->GetWidth() + px].y = 0;    // Green
									remapImage[py * input->GetWidth() + px].z = 0;    // Blue
								}
							}
						}
						
						if(x1 == x2 && y1 == y2) break;
						
						int e2 = 2 * err;
						if(e2 > -dy) {
							err -= dy;
							x1 += sx;
						}
						if(e2 < dx) {
							err += dx;
							y1 += sy;
						}
					}
				}
			}

			// Save the remapped image with detections
			remapOutput->Render(remapImage, input->GetWidth(), input->GetHeight());
			LogInfo("Saved remapped image with detections\n");

			// Clean up
			delete[] remapImage;
			SAFE_DELETE(remapOutput);
		}
	}

	/*
	 * destroy resources
	 */
	printf("one-image-viewer:  shutting down...\n");
	
	// Clean up CUDA resources
	cudaFree(img_src_dev);
	cudaFree(img_dst_dev);
	cudaFree(mapxDevPtr);
	cudaFree(mapyDevPtr);

	// Clean up AprilTag resources
	apriltag_detector_destroy(tag_detector);
	if(strcmp(tag_family_name, "36h11") == 0)
		tag36h11_destroy(tag_family);
	else if(strcmp(tag_family_name, "16h5") == 0)
		tag16h5_destroy(tag_family);
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);

	printf("one-image-viewer:  shutdown complete\n");
} 
