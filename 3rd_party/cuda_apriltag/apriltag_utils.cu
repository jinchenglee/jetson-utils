#include <iomanip>
#include <iostream>

#include "apriltag_utils.h"
#include "glog/logging.h"

using namespace std;
using namespace cv;

bool setup_tag_family(apriltag_family_t **tf, const char *famname) {
  if (!strcmp(famname, "tag36h11")) {
    *tf = tag36h11_create();
  } else if (!strcmp(famname, "tag25h9")) {
    *tf = tag25h9_create();
  } else if (!strcmp(famname, "tag16h5")) {
    *tf = tag16h5_create();
  } else if (!strcmp(famname, "tagCircle21h7")) {
    *tf = tagCircle21h7_create();
  } else if (!strcmp(famname, "tagCircle49h12")) {
    *tf = tagCircle49h12_create();
  } else if (!strcmp(famname, "tagStandard41h12")) {
    *tf = tagStandard41h12_create();
  } else if (!strcmp(famname, "tagStandard52h13")) {
    *tf = tagStandard52h13_create();
  } else if (!strcmp(famname, "tagCustom48h12")) {
    *tf = tagCustom48h12_create();
  } else {
    LOG(ERROR) << "Unknown tag family: " << famname;
    return (false);
  }
  return (true);
}

void teardown_tag_family(apriltag_family_t **tf, const char *famname) {
  if (!strcmp(famname, "tag36h11")) {
    tag36h11_destroy(*tf);
  } else if (!strcmp(famname, "tag25h9")) {
    tag25h9_destroy(*tf);
  } else if (!strcmp(famname, "tag16h5")) {
    tag16h5_destroy(*tf);
  } else if (!strcmp(famname, "tagCircle21h7")) {
    tagCircle21h7_destroy(*tf);
  } else if (!strcmp(famname, "tagCircle49h12")) {
    tagCircle49h12_destroy(*tf);
  } else if (!strcmp(famname, "tagStandard41h12")) {
    tagStandard41h12_destroy(*tf);
  } else if (!strcmp(famname, "tagStandard52h13")) {
    tagStandard52h13_destroy(*tf);
  } else if (!strcmp(famname, "tagCustom48h12")) {
    tagCustom48h12_destroy(*tf);
  }
}

void draw_detection_outlines(Mat &im, zarray_t *detections) {
  for (int i = 0; i < zarray_size(detections); i++) {
    apriltag_detection_t *det;
    zarray_get(detections, i, &det);
    line(im, Point(det->p[0][0], det->p[0][1]),
         Point(det->p[1][0], det->p[1][1]), Scalar(0, 0xff, 0), 2);
    line(im, Point(det->p[0][0], det->p[0][1]),
         Point(det->p[3][0], det->p[3][1]), Scalar(0, 0, 0xff), 2);
    line(im, Point(det->p[1][0], det->p[1][1]),
         Point(det->p[2][0], det->p[2][1]), Scalar(0xff, 0, 0), 2);
    line(im, Point(det->p[2][0], det->p[2][1]),
         Point(det->p[3][0], det->p[3][1]), Scalar(0xff, 0, 0), 2);

    stringstream ss;
    ss << det->id;
    String text = ss.str();
    int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontscale = 1.0;
    int baseline;
    Size textsize = getTextSize(text, fontface, fontscale, 2, &baseline);
    putText(
        im, text,
        Point(det->c[0] - textsize.width / 2, det->c[1] + textsize.height / 2),
        fontface, fontscale, Scalar(0xff, 0x99, 0), 2);
  }
}

void print_detections(zarray_t *detections) {
  for (int i = 0; i < zarray_size(detections); i++) {
    apriltag_detection_t *det;
    zarray_get(detections, i, &det);
    std::cout << "tag #: " << det->id << std::endl;
    std::cout << "hamming: " << det->hamming << std::endl;
    std::cout << "margin: " << det->decision_margin << std::endl;
    std::cout << "center: " << det->c[0] << "," << det->c[1] << std::endl;
    for (size_t j = 0; j < det->H->ncols; ++j) {
      std::cout << std::endl;
      for (size_t k = 0; k < det->H->nrows; ++k) {
        std::cout << matd_get(det->H, j, k) << " ";
      }
    }
    std::cout << std::endl;
  }
}
