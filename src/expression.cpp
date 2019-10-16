
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

void expression::setSplines() {

	setPoints(0, 17, jaw);
	setPoints(17, 5, leftEyebrown);
	setPoints(22, 5, rightEyebrown);
	setPoints(27, 4, nasalBridge);
	setPoints(31, 5, nose);
	setPoints(36, 4, leftUpperEye);
	setPoints(39, 3, leftLowerEye, 36);
	setPoints(42, 4, rightUpperEye);
	setPoints(45, 3, rightLowerEye, 42);
	setPoints(48, 7, outerUpperLip);
	setPoints(54, 6, outerLowerLip, 48);
	setPoints(60, 5, innerUpperLip);
	setPoints(64, 4, innerLowerLip, 60);

}

void expression::drawFace(Mat &img) {
	jaw.drawSplines(img);
	leftEyebrown.drawSplines(img);
	rightEyebrown.drawSplines(img);
	nasalBridge.drawSplines(img);
	nose.drawSplines(img);
	leftUpperEye.drawSplines(img);
	leftLowerEye.drawSplines(img);
	rightUpperEye.drawSplines(img);
	rightLowerEye.drawSplines(img);
	outerUpperLip.drawSplines(img);
	outerLowerLip.drawSplines(img);
	innerUpperLip.drawSplines(img);
	innerLowerLip.drawSplines(img);
}

Mat expression::getFeatures() {

	features = jaw.getCoefficients();

	
	hconcat(features, leftEyebrown.getCoefficients(), features);
	hconcat(features, rightEyebrown.getCoefficients(), features);
	hconcat(features, nasalBridge.getCoefficients(), features);
	hconcat(features, nose.getCoefficients(), features);
	hconcat(features, leftUpperEye.getCoefficients(), features);
	hconcat(features, leftLowerEye.getCoefficients(), features);
	hconcat(features, rightUpperEye.getCoefficients(), features);
	hconcat(features, rightLowerEye.getCoefficients(), features);
	hconcat(features, outerUpperLip.getCoefficients(), features);
	hconcat(features, outerLowerLip.getCoefficients(), features);
	hconcat(features, innerUpperLip.getCoefficients(), features);
	hconcat(features, innerLowerLip.getCoefficients(), features);

	return features;
}