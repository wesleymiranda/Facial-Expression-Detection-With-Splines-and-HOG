#include <iostream>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace cv;
using namespace dlib;

#include "include/splines.h"
#include "include/expression.h"
#include "include/detector.h"

int main() {

	detector expressionDetector;
	
	expressionDetector.train();
	expressionDetector.test();
	std::cin.get();
	/*
	while (1) {
		cam >> frame;
		if (frame.empty()) break;
			

			cv_image<bgr_pixel> dlib_frame(frame);
			faceRects = detector(dlib_frame);
			
			for (int i = 0; i < faceRects.size(); i++) {
				faceLandmarks = landmarkDetector(dlib_frame, faceRects[i]);
				expression Expression(frame, faceLandmarks);
			}

			cv::imshow("VIDEO", frame);
			waitKey(1);
	}
	*/
	return 0;
}