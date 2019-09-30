
#include <iostream>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace std;
using namespace cv;
using namespace dlib;

#include "include/splines.h"
#include "include/expression.h"

expression::expression() {}

expression::expression(cv::Mat &image, full_object_detection faceLandmarks_) {
	cv_image<bgr_pixel> dlib_image(image);
	setLandmarks(faceLandmarks_);
	setSplines();
}

void expression::setLandmarks(full_object_detection faceLandmarks_) {
	faceLandmarks = faceLandmarks_;
}