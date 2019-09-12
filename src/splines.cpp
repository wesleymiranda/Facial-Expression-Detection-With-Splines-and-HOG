#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

#include "include/splines.h"

spline::spline() {

}

spline::spline(Eigen::VectorXd x_, Eigen::VectorXd y_) {
	setPointCoord(x_, y_);
	setDifferenceH();
	setMatrixA();
	setVectorB();
	solveEquations();
	setCoefficients();

}

void spline::setPointCoord(Eigen::VectorXd x_, Eigen::VectorXd y_) {
	N = x_.size() - 1;
	x = x_;
	y = y_;
}

