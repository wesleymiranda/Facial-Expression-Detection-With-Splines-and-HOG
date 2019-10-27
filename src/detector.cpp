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

void detector::init() {
	//Inicialização do descritor hog

	nclasses = 4; //Número de saídas
	Size win = Size(128, 128);
	Size block = Size(16, 16);
	Size stride = Size(8, 8);
	Size cell = Size(8, 8);
	int bins = 7;

	hog = new HOGDescriptor(win, block, stride, cell, bins);

	//Inicialização do SVM
	svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setC(10);
	svm->setGamma(0.001);
	svm->setKernel(ml::SVM::INTER);
	svm->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	svmGeometric = ml::SVM::create();
	svmGeometric->setType(ml::SVM::C_SVC);
	svmGeometric->setC(10);
	svmGeometric->setGamma(0.001);
	svmGeometric->setKernel(ml::SVM::INTER);
	svmGeometric->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));


	svmApparence = ml::SVM::create();
	svmApparence->setType(ml::SVM::C_SVC);
	svmApparence->setC(10);
	svmApparence->setGamma(0.001);
	svmApparence->setKernel(ml::SVM::INTER);
	svmApparence->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	//Inicialização da rede neural mlp
	int geometricFeatures = 63 * 4 * 2;
	int apparenceFeatures = ((win.width / cell.width) - 1) * ((win.width / cell.width) - 1) * 4 * bins;

	ann = ml::ANN_MLP::create();
	Mat_<int> layers(4, 1);
	layers(0) = geometricFeatures + apparenceFeatures;
	layers(1) = nclasses * 32;
	layers(2) = nclasses * 16;
	layers(3) = nclasses;
	ann->setLayerSizes(layers);
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
	ann->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 1e-3));
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);

	
	annGeometric = ml::ANN_MLP::create();
	Mat_<int> layersGeo(4, 1);
	layersGeo(4, 1);
	layersGeo(0) = geometricFeatures;
	layersGeo(1) = nclasses * 32;
	layersGeo(2) = nclasses * 16;
	layersGeo(3) = nclasses;
	annGeometric->setLayerSizes(layersGeo);
	annGeometric->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
	annGeometric->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 1e-3));
	annGeometric->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
	
	
	
	annApparence = ml::ANN_MLP::create();
	Mat_<int> layersApp(4, 1);
	layersApp(4, 1);
	layersApp(0) = apparenceFeatures;
	layersApp(1) = nclasses * 32;
	layersApp(2) = nclasses * 16;
	layersApp(3) = nclasses;
	annApparence->setLayerSizes(layersApp);
	annApparence->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
	annApparence->setTermCriteria(cv::TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 1e-3));
	annApparence->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
	
}

