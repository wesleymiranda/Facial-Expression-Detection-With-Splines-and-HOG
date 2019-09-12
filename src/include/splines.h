#pragma once

/// <summary>
/// Spline cúbica paramétrica natural, onde S''(t=0) = 0; S''(t=1) = 0
/// onde t(i) = i/N, portanto 0 < t < N
/// </summary>
class spline {
private:

	uint64_t N;								//Número de equações
	Eigen::VectorXd x, y;					// Coordenadas x e y dos pontos de entrada
	Eigen::MatrixXd A;						// AX=B
	Eigen::VectorXd X_x, X_y, B_x, B_y;		// AX=B, temos de fazer os cálculos para x e para y
	Eigen::VectorXd h;						// h é o vetor que contém "t(i) - t(i-1)"
	Eigen::MatrixXd xCoeff, yCoeff;			// Coeficientes dos polinômios
	Mat cvCoeff;
public:
	spline();
	spline(Eigen::VectorXd x, Eigen::VectorXd y);
	/// <summary>
	/// Armazena as coordenadas dos pontos
	/// </summary>
	/// <param name="x">coordenada x</param>
	/// <param name="y">coordenada y</param>
	void setPointCoord(Eigen::VectorXd x_, Eigen::VectorXd y_);
	/// <summary>
	/// h é o vetor que contém "t(i) - t(i-1)"
	/// considerando t(i) = i/N, portanto 0 < t < N
	/// o intervalo entre t(i) e t(i-i) é igualmente espaçado
	/// e podemos dizer que o espaçamento é de t=1/N
	/// </summary>
	void setDifferenceH();
	/// <summary>
	/// Prepara a matriz A, que será utilizada na resolução do sistema: AX = B.
	/// A matriz A só depende de t, por isso é a mesma matriz para as splines em x e em y.
	/// por isso criei apenas uma matriz A.
	/// </summary>
	void setMatrixA();
	/// <summary>
	/// Prepara o vetor B, que será utilizado na resolução do sistema AX = B.
	/// O vetor B depende de x e y, por isso não é o mesmo para as splines em x e y.
	/// por isso criei B_x e B_y.
	/// </summary>
	void setVectorB();
	/// <summary>
	/// Calcula o sistema linear AX = B
	/// </summary>
	void solveEquations();
	/// <summary>
	/// Armazena os valores dos coeficientes a, b, c, d tanto para as splines em x, quanto para as splines em y
	/// </summary>
	void setCoefficients();
	/// <summary>
	/// Calcula o resultado de uma função paramétrica cúbica.
	/// Onde a função varia em t e tem retornos em x e em y.
	/// </summary>
	/// <param name="t">vatiável de entrada </param>
	/// <param name="tk">constante</param>
	/// <param name="line">linha onde se encontram os coeficientes: a, b, c, d</param>
	/// <param name="x">resultado do polinômio em x</param>
	/// <param name="y">resultado do polinômio em y</param>
	void cubicFunction(double t, double tk, int line, double& x, double& y);
	/// <summary>
	/// Desenha as splines em uma imagem do OpenCV
	/// </summary>
	/// <param name="img">imagem onde serão desenhadas as splines </param>
	void drawSplines(Mat& img);

	Mat getCoefficients();
};

