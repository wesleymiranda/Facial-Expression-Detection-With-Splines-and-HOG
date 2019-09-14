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

void spline::setDifferenceH() {
	double interval =(double) 1 / N;
	h = Eigen::VectorXd::Constant(N, 1, interval);
}

void spline::setMatrixA() {
	A = Eigen::MatrixXd::Zero(N - 1, N - 1);

	for (uint64_t i = 0; i < N - 1; i++) {
		if (i > 0) A(i, i - 1) = h(i);
		A(i, i) = 2 * (h(i + 1) + h(i));
		if (i < N - 2) A(i, i + 1) = h(i + 1);
	}
}

void spline::setVectorB() {
	B_x = Eigen::VectorXd::Zero(N - 1);
	B_y = Eigen::VectorXd::Zero(N - 1);

	for (uint64_t i = 1; i < N; i++) {
		B_x(i - 1) = 6 * ((x(i + 1) - x(i)) / h(i) - (x(i) - x(i - 1)) / h(i - 1));
		B_y(i - 1) = 6 * ((y(i + 1) - y(i)) / h(i) - (y(i) - y(i - 1)) / h(i - 1));
	}
}
