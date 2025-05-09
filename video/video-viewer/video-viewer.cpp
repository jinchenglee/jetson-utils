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
#include "glDisplay.h"  // Add include for glDisplay

#include "logging.h"
#include "commandLine.h"
#include "glUtility.h"  // For GL drawing functions

#include <signal.h>

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
	printf("usage: video-viewer [--help] [--tag-family=36h11|16h5] input_URI [output_URI]\n\n");
	printf("View/output a video or image stream with AprilTag detection.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");
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

	/*
	 * create input video stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("video-viewer:  failed to create input stream\n");
		return 0;
	}

	/*
	 * create output video stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
	{
		LogError("video-viewer:  failed to create output stream\n");
		return 0;
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

	// Create reusable grayscale image buffer
	image_u8_t im = {
		.width = input->GetWidth(),
		.height = input->GetHeight(),
		.stride = input->GetWidth(),
		.buf = new uint8_t[input->GetWidth() * input->GetHeight()]
	};

	// Initialize GL display and begin rendering context
	glDisplay* display = NULL;
	if( output->IsType<glDisplay>() )
	{
		display = (glDisplay*)output;
	}

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
		uchar3* image = NULL;
		int status = 0;
		
		if( !input->Capture(&image, &status) )
		{
			if( status == videoSource::TIMEOUT )
				continue;
			
			break; // EOS
		}

		if( numFrames % 25 == 0 || numFrames < 15 )
			LogVerbose("video-viewer:  captured %u frames (%ux%u)\n", numFrames, input->GetWidth(), input->GetHeight());
		
		numFrames++;

		// Convert RGB to grayscale
		for(int y = 0; y < im.height; y++) {
			for(int x = 0; x < im.width; x++) {
				uchar3 pixel = image[y * im.width + x];
				im.buf[y * im.stride + x] = (uint8_t)((pixel.x * 0.299 + pixel.y * 0.587 + pixel.z * 0.114));
			}
		}

		// Detect AprilTags
		zarray_t* detections = apriltag_detector_detect(tag_detector, &im);

		// Print detection statistics every 30 frames
		if( numFrames % 30 == 0 )
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

		if( output != NULL )
		{
			// Begin OpenGL rendering
			display->BeginRender();

			// First render the image
			output->Render(image, input->GetWidth(), input->GetHeight());

			// Then draw the detections
			int numTags = zarray_size(detections);
			if( numTags > 0 )
			{
				framesWithTags++;
				bool hasHighConfidenceTag = false;
				
				for(int i = 0; i < numTags; i++) {
					apriltag_detection_t* det;
					zarray_get(detections, i, &det);
					
					totalTagsDetected++;
					if( det->decision_margin > 100.0f )
					{
						highConfidenceTagsDetected++;
						hasHighConfidenceTag = true;
					}
					
					// Print tag info for each detection
					if( numFrames % 30 == 0 )
					{
						printf("  Tag %d: id=%d, hamming=%d, margin=%.3f %s\n", 
							i, det->id, det->hamming, det->decision_margin,
							det->decision_margin > 100.0f ? "(high confidence)" : "(low confidence)");
					}
					
					// Only draw if detection confidence is high enough
					if( det->decision_margin > 100.0f )
					{
						display->RenderLine(det->p[0][0], det->p[0][1], det->p[1][0], det->p[1][1], 0.9f, 0.0f, 0.0f);
						display->RenderLine(det->p[1][0], det->p[1][1], det->p[2][0], det->p[2][1], 0.0f, 0.9f, 0.0f);
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
				input->GetWidth(), input->GetHeight(), output->GetFrameRate(), 
				highConfidenceTagsDetected, totalTagsDetected, tag_family_name);
			output->SetStatus(str);	

			// check if the user quit
			if( !output->IsStreaming() )
				break;
		}
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
	
	// Clean up AprilTag resources
	apriltag_detector_destroy(tag_detector);
	if(strcmp(tag_family_name, "36h11") == 0)
		tag36h11_destroy(tag_family);
	else if(strcmp(tag_family_name, "16h5") == 0)
		tag16h5_destroy(tag_family);
	
	// Clean up grayscale image buffer
	delete[] im.buf;
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);

	printf("video-viewer:  shutdown complete\n");
}

