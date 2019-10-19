#include <string>
#include <iostream>
#include <filesystem>

#include <Eigen/Dense>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;
using namespace dlib;
namespace fs = std::filesystem;

#include "include/splines.h"
#include "include/expression.h"
#include "include/detector.h"

detector::detector() {
	deserialize("../datas/face_predictor/predictor_face_landmarks.dat") >> landmarkDetector;
	faceDetector = get_frontal_face_detector();
	//svm = ml::SVM::load("ml/svm.yml");
	//ann = ml::ANN_MLP::load("ml/ann.yml");
}

