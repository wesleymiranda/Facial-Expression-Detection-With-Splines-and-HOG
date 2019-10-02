
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

void expression::setPoints(int init, int size, spline &s, int other = 0) {
	Eigen::VectorXd x;
	Eigen::VectorXd y;
	x = Eigen::VectorXd::Zero(size);
	y = Eigen::VectorXd::Zero(size);

	for (int i = 0; i < size; i++) {
		x(i) =  (double)faceLandmarks.part(i + init).x();
		y(i) =  (double)faceLandmarks.part(i + init).y();
	}

	if (other != 0) {
		x.conservativeResize(x.size() + 1);
		y.conservativeResize(y.size() + 1);

		x(x.size() - 1) =  (double)faceLandmarks.part(other).x();
		y(y.size() - 1) =  (double)faceLandmarks.part(other).y();
	}

	s = spline(x, y);
}
