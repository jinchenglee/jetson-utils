#ifndef APRILTAG_UTILS_H_
#define APRILTAG_UTILS_H_

#include "opencv2/opencv.hpp"

extern "C" {
#include "apriltag.h"
#include "common/getopt.h"
#include "common/matd.h"
#include "tag16h5.h"
#include "tag25h9.h"
#include "tag36h11.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
}

using namespace cv;

bool setup_tag_family(apriltag_family_t **tf, const char *famname);
void teardown_tag_family(apriltag_family_t **tf, const char *famname);
void draw_detection_outlines(Mat &im, zarray_t *detections);
void print_detections(zarray_t *detections);

#endif