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

void detector::train() {

	init();

	std::string path = "../datas/images/DBs/mydb/train";

	Mat trainClassesMLP, trainClassesSVM; // saída esperada
	Mat geometricFeatures, apparenceFeatures;

	int cont = 0;
	for (const auto& entry : fs::directory_iterator(path)) {
		std::string file = entry.path().string();
		
		Mat img = imread(file);
		DLIBImage dlibImg(img);
		
		if (img.empty()) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}

		//----------------------//
		//PARA O CK DATABASE
		//DLIBRects dlibRects;
		//dlibRects.push_back(dlib::rectangle(0, 0, img.cols, img.rows));
		//PARA OUTROS DATABASES
		DLIBRects dlibRects = faceDetector(dlibImg);
		//---------------------//


		if (dlibRects.empty()) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}

		for (int i = 0; i < dlibRects.size(); i++){
			
			std::vector<float> descriptor;
			Rect r = dlibRectToOpenCV(dlibRects[i]);
			if (r.x + r.width > img.cols || r.y + r.height > img.rows) {
				cout << "PROBLEM--" << file << endl;
				break;
			}
			Mat resizedImg = img(r);
			cv::resize(resizedImg, resizedImg, Size(128, 128), 0, 0, INTER_CUBIC);
			hog->compute(resizedImg, descriptor);
			Mat aux(1, descriptor.size(), cv::DataType<float>::type, descriptor.data());
			apparenceFeatures.push_back(aux);

			full_object_detection landmarks;
			getLandmarks(landmarks, dlibImg, dlibRects[i]);
			expression facialExpression = expression(img, landmarks);
			geometricFeatures.push_back(facialExpression.getFeatures());

			Mat m(200, 200, CV_8UC3, Scalar(0));
			facialExpression.drawFace(m);

			trainClassesMLP.push_back(Mat::zeros(1, nclasses, CV_32F));
			trainClassesSVM.push_back(Mat::zeros(1, 1, CV_32S));

			if (file.find("happy") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 0) = 1;
				trainClassesSVM.at<int>(cont, 0) = 0;
			}
			else if (file.find("neutral") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 1) = 1;
				trainClassesSVM.at<int>(cont, 0) = 1;
			}
			else if (file.find("sad") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 2) = 1;
				trainClassesSVM.at<int>(cont, 0) = 2;
			}
			else if (file.find("surprised") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 3) = 1;
				trainClassesSVM.at<int>(cont, 0) = 3;
			}/*
			else if (file.find("fear") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 4) = 1;
				trainClassesSVM.at<int>(cont, 0) = 4;
			}else if (file.find("joy") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 5) = 1;
				trainClassesSVM.at<int>(cont, 0) = 5;
			}
			else if (file.find("neutral") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 6) = 1;
				trainClassesSVM.at<int>(cont, 0) = 6;
			}
			else if (file.find("pride") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 7) = 1;
				trainClassesSVM.at<int>(cont, 0) = 7;
			}
			else if (file.find("sad") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 8) = 1;
				trainClassesSVM.at<int>(cont, 0) = 8;
			}
			else if (file.find("surprised") != std::string::npos) {
				trainClassesMLP.at<float>(cont, 9) = 1;
				trainClassesSVM.at<int>(cont, 0) = 9;
			}*/
			
		}
		cont++;
	}

	annGeometric->train(geometricFeatures, ml::ROW_SAMPLE, trainClassesMLP);
	annApparence->train(apparenceFeatures, ml::ROW_SAMPLE, trainClassesMLP);
	svmGeometric->train(geometricFeatures, ml::ROW_SAMPLE, trainClassesSVM);
	svmApparence->train(apparenceFeatures, ml::ROW_SAMPLE, trainClassesSVM);

	Mat features;
	hconcat(apparenceFeatures, geometricFeatures, features);
	
	svm->train(features, ml::ROW_SAMPLE, trainClassesSVM);
	ann->train(features, ml::ROW_SAMPLE, trainClassesMLP);
	
	//saveTrain();
}

void detector::saveTrain() {
	svm->save("ml/svm.yml");
	ann->save("ml/ann.yml");
}

void detector::test() {

	std::string path = "../datas/images/DBs/mydb/test";

	Mat geometricFeatures, apparenceFeatures;
	Mat testLabels;
	
	for (const auto& entry : fs::directory_iterator(path)) {
		std::string file = entry.path().string();

		Mat img = imread(file);
		DLIBImage dlibImg(img);
		
		if (img.empty()) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}

		DLIBRects dlibRects = faceDetector(dlibImg);
		if (dlibRects.empty()) {
			std::cout << "---PROBLEM" << file << std::endl;
			continue;
		}

		for (int i = 0; i < dlibRects.size(); i++) {

			std::vector<float> descriptor;
			Rect r = dlibRectToOpenCV(dlibRects[i]);
			if (r.x + r.width > img.cols || r.y + r.height > img.rows) {
				cout << "PROBLEM--" << file << endl;
				break;
			}
			Mat resizedImg = img(r);
			cv::resize(resizedImg, resizedImg, Size(128, 128), 0, 0, INTER_CUBIC);
			hog->compute(resizedImg, descriptor);
			Mat aux(1, descriptor.size(), cv::DataType<float>::type, descriptor.data());
			apparenceFeatures.push_back(aux);
			
			full_object_detection landmarks;
			getLandmarks(landmarks, dlibImg, dlibRects[i]);
			expression facialExpression = expression(img, landmarks);
			geometricFeatures.push_back(facialExpression.getFeatures());


			if (file.find("happy") != std::string::npos) {
				testLabels.push_back(0);
			}
			else if (file.find("neutral") != std::string::npos) {
				testLabels.push_back(1);
			}
			else if (file.find("sad") != std::string::npos) {
				testLabels.push_back(2);
			}
			else if (file.find("surprised") != std::string::npos) {
				testLabels.push_back(3);
			}/*
			else if (file.find("fear") != std::string::npos) {
				testLabels.push_back(4);
			}
			else if (file.find("joy") != std::string::npos) {
				testLabels.push_back(5);
			}
			else if (file.find("neutral") != std::string::npos) {
				testLabels.push_back(6);
			}
			else if (file.find("pride") != std::string::npos) {
				testLabels.push_back(7);
			}
			else if (file.find("sad") != std::string::npos) {
				testLabels.push_back(8);
			}
			else if (file.find("surprised") != std::string::npos) {
				testLabels.push_back(9);
			}*/
		}
	}

	Mat features;
	hconcat(apparenceFeatures, geometricFeatures, features);

	int predMLP, predSVM;
	int truthMLP, truthSVM;
	Mat resultMLP, resultSVM;
	Mat confusionMLP(nclasses, nclasses, CV_32S, Scalar(0));
	Mat confusionSVM(nclasses, nclasses, CV_32S, Scalar(0));

	for (int i = 0; i < geometricFeatures.rows; i++) {
		predMLP = annGeometric->predict(geometricFeatures.row(i), resultMLP);
		predSVM = svmGeometric->predict(geometricFeatures.row(i), resultSVM);
		predSVM = (int) resultSVM.at<float>(0,0);
		truthMLP = testLabels.at<int>(i);
		truthSVM = testLabels.at<int>(i);
		confusionMLP.at<int>(predMLP, truthMLP)++;
		confusionSVM.at<int>(predSVM, truthSVM)++;

	}

	Mat correct = confusionMLP.diag();
	float accuracyMLP = sum(correct)[0] / sum(confusionMLP)[0];
	std::cerr << "GEOMETRIC: " << std::endl;
	std::cerr << "accuracyMLP: " << accuracyMLP << std::endl;
	std::cerr << "confusion:\n " << confusionMLP << std::endl;

	Mat correctSVM = confusionSVM.diag();
	float accuracySVM = sum(correctSVM)[0] / sum(confusionSVM)[0];
	std::cerr << "accuracySVM: " << accuracySVM << std::endl;
	std::cerr << "confusionSVM:\n " << confusionSVM << std::endl;


	confusionMLP =  Mat::zeros(nclasses, nclasses, CV_32S);
	confusionSVM =  Mat::zeros(nclasses, nclasses, CV_32S);


	for (int i = 0; i < apparenceFeatures.rows; i++) {
		predMLP = annApparence->predict(apparenceFeatures.row(i), resultMLP);
		predSVM = svmApparence->predict(apparenceFeatures.row(i), resultSVM);
		predSVM = (int)resultSVM.at<float>(0, 0);
		truthMLP = testLabels.at<int>(i);
		truthSVM = testLabels.at<int>(i);
		confusionMLP.at<int>(predMLP, truthMLP)++;
		confusionSVM.at<int>(predSVM, truthSVM)++;

	}

	correct = confusionMLP.diag();
	accuracyMLP = sum(correct)[0] / sum(confusionMLP)[0];
	std::cerr << "APPARENCE: " << std::endl;
	std::cerr << "accuracyMLP: " << accuracyMLP << std::endl;
	std::cerr << "confusion:\n " << confusionMLP << std::endl;

	correctSVM = confusionSVM.diag();
	accuracySVM = sum(correctSVM)[0] / sum(confusionSVM)[0];
	std::cerr << "accuracySVM: " << accuracySVM << std::endl;
	std::cerr << "confusionSVM:\n " << confusionSVM << std::endl;


	confusionMLP =  Mat::zeros(nclasses, nclasses, CV_32S);
	confusionSVM =  Mat::zeros(nclasses, nclasses, CV_32S);

	for (int i = 0; i < features.rows; i++) {
		predMLP = ann->predict(features.row(i), resultMLP);
		predSVM = svm->predict(features.row(i), resultSVM);
		predSVM = (int)resultSVM.at<float>(0, 0);
		truthMLP = testLabels.at<int>(i);
		truthSVM = testLabels.at<int>(i);
		confusionMLP.at<int>(predMLP, truthMLP)++;
		confusionSVM.at<int>(predSVM, truthSVM)++;

	}


	correct = confusionMLP.diag();
	accuracyMLP = sum(correct)[0] / sum(confusionMLP)[0];
	std::cerr << "UNION: " << std::endl;
	std::cerr << "accuracyMLP: " << accuracyMLP << std::endl;
	std::cerr << "confusion:\n " << confusionMLP << std::endl;

	correctSVM = confusionSVM.diag();
	accuracySVM = sum(correctSVM)[0] / sum(confusionSVM)[0];
	std::cerr << "accuracySVM: " << accuracySVM << std::endl;
	std::cerr << "confusionSVM:\n " << confusionSVM << std::endl;
}

cv::Rect detector::dlibRectToOpenCV(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right(), r.bottom()));
}

void detector::getLandmarks(full_object_detection &flNormalized, DLIBImage dlibImg,dlib::rectangle r) 
{
	full_object_detection faceLandmarks = landmarkDetector(dlibImg, r);
	chip_details chip = get_face_chip_details(faceLandmarks, 200);
	flNormalized = map_det_to_chip(faceLandmarks, chip);

}