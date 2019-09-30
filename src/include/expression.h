#pragma once

class expression {

private:
	
	full_object_detection faceLandmarks;
	Mat features;

	spline jaw;				//jaw: maxilar
	spline leftEyebrown;	//left eyebrown: sobrancelha esquerda
	spline rightEyebrown;	//right eyebrown: sobrancelha direita
	spline nasalBridge;		//nasal bridge: ponte nasal
	spline nose;			//nose: nariz inferior
	spline leftUpperEye;	//left upper eye: olho esquerdo parte superior
	spline leftLowerEye;	//left lower eye: olho esquerdo parte inferior
	spline rightUpperEye;	//right upper eye: olho direito parte superior
	spline rightLowerEye;	//right lower eye: olho direito parte inferior
	spline outerUpperLip;	//outer upper lip: lábio superior parte externa
	spline outerLowerLip;	//outer lower lip: lábio inferior parte externa
	spline innerUpperLip;	//inner upper lip: lábio superior parte interna
	spline innerLowerLip;	//inner lower lip: lábio inferior parte interna
	
	void setLandmarks(full_object_detection faceLandmarks);
	void setPoints(int init, int size, spline& s, int other);
	void setSplines();
	

public:
	expression();
	expression(cv::Mat& image, full_object_detection faceLandmarks_);
	Mat getFeatures();
	void drawFace(Mat& img);
};


