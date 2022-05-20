#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <time.h>
#include <vector>
#include <iostream> // library that contain basic input/output functions
#include <fstream> 
#include <iomanip>
#include <string.h>


#include "VMF_Filter_CPU.h"	
#include "DDF_Filter_CPU.h"
#include "BVDF_Filter_CPU.h"
#include "FastDDF_Filter_CPU.h"
#include "FastBVDF_Filter_CPU.h"
#include "VMF_Filter_CPU.h"
#include "GVDF_Filter_CPU.h"
#include "FuzzyPeerGroup.h"
#include "FTSCF_Filter_CPU.h"


#define	M			512		// horizontal, x   //estan bien estos columnnas
#define N			512  // verticual, y   // filas
#define nChanels	3


using namespace cv;
using namespace std;


double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);        // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}

double getMAE(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|

	Scalar s = sum(s1);        // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double mae = sse / (double)(I1.channels() * I1.total());

		return mae;
	}
}



float getMCRE(const Mat& I1, const Mat& I2)
{
	long int JJ;
	double XYZ[9], x, y, z, x1, y1, z1, X, Y, Z, dist = 0;
	int k;
	XYZ[0] = 0.489989; XYZ[1] = 0.310008; XYZ[2] = 0.2; XYZ[3] = 0.176962;
	XYZ[4] = 0.81240; XYZ[5] = 0.01; XYZ[6] = 0.0; XYZ[7] = 0.01; XYZ[8] = 0.99;

	unsigned char *datosI1, *datosI2;
	datosI1 = (unsigned char*)(I1.data); datosI2 = (unsigned char*)(I2.data);

	for (int Row = 1; Row < I1.rows - 1; Row++) {
		for (int Col = 1; Col < I1.cols - 1; Col++) {
			if (datosI1[(Row * I1.rows + Col) * 3 + 0] == 0 & datosI1[(Row * I1.rows + Col) * 3 + 1] == 0 & datosI1[(Row * I1.rows + Col) * 3 + 2] == 0)
			{
				x = 0, y = 0, z = 0;
			}
			else {
				X = XYZ[0] * datosI1[(Row * I1.rows + Col) * 3 + 2] + XYZ[1] * datosI1[(Row * I1.rows + Col) * 3 + 1] + XYZ[2] * datosI1[(Row * I1.rows + Col) * 3 + 0];
				Y = XYZ[3] * datosI1[(Row * I1.rows + Col) * 3 + 2] + XYZ[4] * datosI1[(Row * I1.rows + Col) * 3 + 1] + XYZ[5] * datosI1[(Row * I1.rows + Col) * 3 + 0];
				Z = XYZ[6] * datosI1[(Row * I1.rows + Col) * 3 + 2] + XYZ[7] * datosI1[(Row * I1.rows + Col) * 3 + 1] + XYZ[8] * datosI1[(Row * I1.rows + Col) * 3 + 0];
				x = X / (X + Y + Z);
				y = Y / (X + Y + Z);
				z = Z / (X + Y + Z);
			}
			//apartir de aqui son R1,B1 y G1
			if (datosI2[(Row * I1.rows + Col) * 3 + 0] + datosI2[(Row * I1.rows + Col) * 3 + 1] + datosI2[(Row * I1.rows + Col) * 3 + 2] == 0)
			{
				x1 = 0, y1 = 0, z1 = 0;
			}
			else
			{
				X = XYZ[0] * datosI2[(Row * I1.rows + Col) * 3 + 2] + XYZ[1] * datosI2[(Row * I1.rows + Col) * 3 + 1] + XYZ[2] * datosI2[(Row * I1.rows + Col) * 3 + 0];
				Y = XYZ[3] * datosI2[(Row * I1.rows + Col) * 3 + 2] + XYZ[4] * datosI2[(Row * I1.rows + Col) * 3 + 1] + XYZ[5] * datosI2[(Row * I1.rows + Col) * 3 + 0];
				Z = XYZ[6] * datosI2[(Row * I1.rows + Col) * 3 + 2] + XYZ[7] * datosI2[(Row * I1.rows + Col) * 3 + 1] + XYZ[8] * datosI2[(Row * I1.rows + Col) * 3 + 0];
				x1 = X / (X + Y + Z);
				y1 = Y / (X + Y + Z);
				z1 = Z / (X + Y + Z);
			}
			dist = sqrt(pow(x - x1, 2) + pow(y - y1, 2) + pow(z - z1, 2)) + dist;
		}
	}
	//printf("%f\n", dist / (I1.rows*I1.cols));
	return dist / (I1.rows*I1.cols);
}

float getNCD(const Mat& I1, const Mat& I2)
{
	long int JJ;
	int k;
	float xyz[20], XYZ[20], WPQ[20];
	float un, vn, L, L1, u, v, ul, vl, v_1, u_1, L_1, u_2, v_2;
	float suma = 0, sumar = 0, NCD = 0;

	xyz[0] = 0.412453;
	xyz[1] = 0.357580;
	xyz[2] = 0.180423;
	xyz[3] = 0.212671;
	xyz[4] = 0.715160;
	xyz[5] = 0.072169;
	xyz[6] = 0.019334;
	xyz[7] = 0.119193;
	xyz[8] = 0.950227;

	xyz[9] = xyz[0] + xyz[1] + xyz[2];
	xyz[10] = xyz[3] + xyz[4] + xyz[5];
	xyz[11] = xyz[6] + xyz[7] + xyz[8];

	un = (4 * xyz[9]) / (xyz[9] + 15 * xyz[10] + 3 * xyz[11]);
	vn = (9 * xyz[10]) / (xyz[9] + 15 * xyz[10] + 3 * xyz[11]);

	XYZ[0] = 0.412453;
	XYZ[1] = 0.357580;
	XYZ[2] = 0.180423;
	XYZ[3] = 0.212671;
	XYZ[4] = 0.715160;
	XYZ[5] = 0.072169;
	XYZ[6] = 0.019334;
	XYZ[7] = 0.119193;
	XYZ[8] = 0.950227;

	WPQ[0] = 0.412453;
	WPQ[1] = 0.357580;
	WPQ[2] = 0.180423;
	WPQ[3] = 0.212671;
	WPQ[4] = 0.715160;
	WPQ[5] = 0.072169;
	WPQ[6] = 0.019334;
	WPQ[7] = 0.119193;
	WPQ[8] = 0.950227;

	unsigned char *datosI1, *datosI2;
	datosI1 = (unsigned char*)(I1.data); datosI2 = (unsigned char*)(I2.data);

	for (int Row = 1; Row < I1.rows - 1; Row++) {
		for (int Col = 1; Col < I1.cols - 1; Col++) {
			XYZ[9] = XYZ[0] * datosI1[(Row * I1.rows + Col) * 3 + 2] + XYZ[1] * datosI1[(Row * I1.rows + Col) * 3 + 1] + XYZ[2] * datosI1[(Row * I1.rows + Col) * 3 + 0];
			XYZ[10] = XYZ[3] * datosI1[(Row * I1.rows + Col) * 3 + 2] + XYZ[4] * datosI1[(Row * I1.rows + Col) * 3 + 1] + XYZ[5] * datosI1[(Row * I1.rows + Col) * 3 + 0];
			XYZ[11] = XYZ[6] * datosI1[(Row * I1.rows + Col) * 3 + 2] + XYZ[7] * datosI1[(Row * I1.rows + Col) * 3 + 1] + XYZ[8] * datosI1[(Row * I1.rows + Col) * 3 + 0];
			L = XYZ[10] / xyz[10];

			if (L >= 0.008856) {
				L1 = 116 * (pow(L, 0.333)) - 16;
			}
			else {
				L1 = 903.3 * L;
			}
			if ((XYZ[9] + 15 * XYZ[10] + 3 * XYZ[11]) == 0) {
				u = 0;
				v = 0;
			}
			else {
				u = 4 * XYZ[9] / (XYZ[9] + 15 * XYZ[10] + 3 * XYZ[11]);
				v = 9 * XYZ[10] / (XYZ[9] + 15 * XYZ[10] + 3 * XYZ[11]);
			}
			ul = 13 * L1*(u - un);
			vl = 13 * L1*(v - vn);

			WPQ[9] = WPQ[0] * datosI2[(Row * I1.rows + Col) * 3 + 2] + WPQ[1] * datosI2[(Row * I1.rows + Col) * 3 + 1] + WPQ[2] * datosI2[(Row * I1.rows + Col) * 3 + 0];
			WPQ[10] = WPQ[3] * datosI2[(Row * I1.rows + Col) * 3 + 2] + WPQ[4] * datosI2[(Row * I1.rows + Col) * 3 + 1] + WPQ[5] * datosI2[(Row * I1.rows + Col) * 3 + 0];
			WPQ[11] = WPQ[6] * datosI2[(Row * I1.rows + Col) * 3 + 2] + WPQ[7] * datosI2[(Row * I1.rows + Col) * 3 + 1] + WPQ[8] * datosI2[(Row * I1.rows + Col) * 3 + 0];
			L = WPQ[10] / xyz[10];
			if (L > 0.008856) {
				L_1 = 116 * (pow(L, 0.333)) - 16;
			}
			else {
				L_1 = 903.3 * L;
			}

			if ((WPQ[9] + 15 * WPQ[10] + 3 * WPQ[11]) == 0) {
				u_1 = 0;
				v_1 = 0;
			}
			else {
				if (XYZ[9] == 0 && XYZ[10] == 0 && XYZ[11] == 0) {
					u_1 = 0;
					v_1 = 0;
				}
				else {
					u_1 = 4 * XYZ[9] / (XYZ[9] + 15 * XYZ[10] + 3 * XYZ[11]);
					v_1 = 9 * XYZ[10] / (XYZ[9] + 15 * XYZ[10] + 3 * XYZ[11]);
				}
			}
			u_2 = 13 * L_1*(u_1 - un);
			v_2 = 13 * L_1*(v_1 - vn);

			suma = sqrt(pow(L_1, 2) + pow(u_2, 2) + pow(v_2, 2)) + suma;
			sumar = sqrt(pow(L1 - L_1, 2) + pow(ul - u_2, 2) + pow(vl - v_2, 2)) + sumar;
		}
	}
	NCD = sumar / suma;
	return NCD;
}




double getMCRE_Mio(const Mat& I1, const Mat& I2)
{
	float pixelR1, pixelG1, pixelB1, pixelR2, pixelG2, pixelB2;
	float valMag1, valMag2;
	double distancia = 0.0, aux = 0.0;
	float pixelUnit1[3], pixelUnit2[3];
	unsigned char *datosI1, *datosI2;
	datosI1 = (unsigned char*)(I1.data);
	datosI2 = (unsigned char*)(I2.data);

	for (int Col = 2; Col <= I1.rows - 2; Col++) {
		for (int Row = 2; Row <= I1.rows - 2; Row++) {
			pixelR1 = datosI1[((Row)* I1.rows + (Col)) * 3 + 0];
			pixelG1 = datosI1[((Row)* I1.rows + (Col)) * 3 + 1];
			pixelB1 = datosI1[((Row)* I1.rows + (Col)) * 3 + 2];

			valMag1 = sqrt((pixelR1*pixelR1) + (pixelG1*pixelG1) + (pixelB1*pixelB1));



			pixelR2 = datosI2[((Row)* I1.rows + (Col)) * 3 + 0];
			pixelG2 = datosI2[((Row)* I1.rows + (Col)) * 3 + 1];
			pixelB2 = datosI2[((Row)* I1.rows + (Col)) * 3 + 2];

			valMag2 = sqrt((pixelR2*pixelR2) + (pixelG2*pixelG2) + (pixelB2*pixelB2));


			if (valMag1 == 0 || valMag2 == 0) {
				distancia += 0;
				//printf("divicion por cero\n");
			}
			else {
				pixelUnit1[0] = (pixelR1 / valMag1);
				pixelUnit1[1] = (pixelG1 / valMag1);
				pixelUnit1[2] = (pixelB1 / valMag1);

				pixelUnit2[0] = (pixelR2 / valMag2);
				pixelUnit2[1] = (pixelG2 / valMag2);
				pixelUnit2[2] = (pixelB2 / valMag2);

				distancia += sqrt(pow((pixelUnit1[0] * 255) - (pixelUnit2[0] * 255), 2)
					+ pow((pixelUnit1[1] * 255) - (pixelUnit2[1] * 255), 2)
					+ pow((pixelUnit1[2] * 255) - (pixelUnit2[2] * 255), 2));
			}

		}
	}
	aux = distancia / (I1.rows*I1.cols);
	return aux;
}

void EscribirCriterios(double *valPSNR, double *valMCRE, double *valNCD, double *valMAE, int nExperimentos)
{
	// Escribir PSNR en texto
	ofstream fout("PSNR.txt"); //opening an output stream for file test.txt
	if (fout.is_open())
	{
		//file opened successfully so we are here
		cout << "File Opened successfully!!!. Writing data from array to file" << endl;

		for (int g = 0; g <= nExperimentos; g++)
		{
			fout << fixed << setprecision(2) << valPSNR[g] << endl;
			//writing ith character of array in the file
		}
		cout << "Array data successfully saved into the file PSNR.txt" << endl;
		fout.close();
	}
	else //file could not be opened
	{
		cout << "File could not be opened." << endl;
	}

	ofstream foutMCRE("MCRE.txt"); //opening an output stream for file test.txt
	if (foutMCRE.is_open())
	{
		//file opened successfully so we are here
		cout << "File Opened successfully!!!. Writing data from array to file" << endl;

		for (int g = 0; g <= nExperimentos; g++)
		{
			foutMCRE << fixed << setprecision(8) << valMCRE[g] << endl;
			//writing ith character of array in the file
		}
		cout << "Array data successfully saved into the file MCRE.txt" << endl;
		foutMCRE.close();
	}
	else //file could not be opened
	{
		cout << "File could not be opened." << endl;
	}
	// Escribir PSNR en texto
	ofstream foutNCD("NCD.txt"); //opening an output stream for file test.txt
	if (foutNCD.is_open())
	{
		//file opened successfully so we are here
		cout << "File Opened successfully!!!. Writing data from array to file" << endl;

		for (int g = 0; g <= nExperimentos; g++)
		{
			foutNCD << fixed << setprecision(8) << valNCD[g] << endl;
			//writing ith character of array in the file
		}
		cout << "Array data successfully saved into the file NCD.txt" << endl;
		foutNCD.close();
	}
	else //file could not be opened
	{
		cout << "File could not be opened." << endl;
	}

	ofstream foutMAE("MAE.txt"); //opening an output stream for file test.txt
	if (foutMAE.is_open())
	{
		//file opened successfully so we are here
		cout << "File Opened successfully!!!. Writing data from array to file" << endl;

		for (int g = 0; g <= nExperimentos; g++)
		{
			foutMAE << fixed << setprecision(8) << valMAE[g] << endl;
			//writing ith character of array in the file
		}
		cout << "Array data successfully saved into the file MAE.txt" << endl;
		foutMAE.close();
	}
	else //file could not be opened
	{
		cout << "File could not be opened." << endl;
	}

	printf("escrito");
}

void ObtenerPath(char* Dir, int i) {

	char numeroDeImagen[3];
	_itoa_s(i, numeroDeImagen, 10);


	strcat_s(Dir, 200, numeroDeImagen);

	strcat_s(Dir, 200, ".bmp");

}


int main()
{
	clock_t start, diff;
	int msecMenor = 10000;
	int msec = 0, sumaT = 0;
	int nExperimentos = 10, nThreads = 0;
	int contador = 60;
	double valPSNR[100], valMCRE[100], valMCREMio[100], valNCD[100], valMAE[100];
	//Mat imageOriginal = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512/0.bmp", IMREAD_UNCHANGED);
	//Mat imageOriginal = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512_Aleatorio/0.bmp", IMREAD_UNCHANGED);

	Mat imageOriginal = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512/0.bmp", IMREAD_UNCHANGED);
	//Mat imageOriginal = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena320x320_Aleatorio/0.bmp", IMREAD_UNCHANGED);
	if (!imageOriginal.data)          // Check for invalid input
	{
		cout << "No esta la imagen1" << std::endl;
		_getch();
		return -1;
	}

	int size = (N)*(M) * sizeof(unsigned char)* nChanels;

	//Se usa malloc para poder procesar imagenes grandes
	unsigned char *h_in;
	h_in = (unsigned char *)malloc(size);
	h_in = (unsigned char*)(imageOriginal.data);					// puntero a los datos de la imagenIn
	Mat imagenNoise(N, M, CV_8UC1, Scalar(255));
	unsigned char *h_Noise;
	h_Noise = (unsigned char *)malloc((N)*(M)* sizeof(unsigned char));
	h_Noise = (unsigned char*)(imagenNoise.data);


	imshow("imagen de Original", imageOriginal);


	//printf_s("%d\n", omp_get_max_threads());


	//obtencion de imagen de prueba e imagen de salida y puntero
	Mat imagenOut(N, M, CV_8UC3, Scalar(255));
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512_Aleatorio/";
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512_Aleatorio/";
	char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512/";
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512/";
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena320x320_Aleatorio/";
	ObtenerPath(Dir, 5);


	Mat image = imread(Dir, IMREAD_UNCHANGED); if (!image.data) { cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/lena1024x1024_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/lena2048x2048_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena4096x4096_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena8192x8192_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data) { cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena320x320_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data) { cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	
	imshow("Imagen Ruidosa", image);

	h_in = (unsigned char *)malloc(size);
	h_in = (unsigned char*)(image.data);
	unsigned char *h_out;	h_out = (unsigned char *)malloc(size);
	imagenOut.data = h_out;

	
	//VMF_CPU_Multi_L1(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//FastBVDF_CPU_Multi(h_out, h_in, N, M, nChanels, 6);
	BVDF_CPU_Multi(h_out, h_in, N, M, nChanels, 6);
	//Detection_FuzzyMetric(h_Noise, h_in, N, M, nChanels, 6);
	//AMF_Filtering(h_out, h_in, h_Noise, N, M, nChanels, 6);
	//VMF_Filtering_Single(h_out, h_in, h_Noise, N, M, nChanels);
	//AMF_Filtering_ExpWindow(h_out, h_in, h_Noise, N, M, nChanels,8);
	//FiltradoPropuesta(h_out, h_in, h_Noise, N, M, nChanels, 8);
	//GVDF_Filter_CPU_Multi(h_out, h_in, N, M, nChanels);
	//VMF_CPU_Multi_Reuso_Sort(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//FTSCF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//MPGFMF(h_out, h_in, N, M, nChanels, 6);// Esta es la propuesta PeerGroup
	
	//FTSCF_CPU_Multi2(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//FTSCF_CPU_Multi_Original(h_out, h_in, N, M, nChanels, 6);


	printf("h_out[1500] = %d", h_out[1500]);

	imagenOut.data = h_out;
	imshow("imagenOut 5 de ruido", imagenOut);
	//imshow("Imagen Noise", imagenNoise);
	///spliting for displaying
	Mat bgr[3];   //destination array
	split(imagenOut, bgr);//split source  
	///

	imshow("bgr[0]", bgr[0]);
	imshow("bgr[1]", bgr[1]);
	imshow("bgr[2]", bgr[2]);

	//imshow("Imagen Noise", imagenNoise);	//waitKey(0);
	//
	
	/*
	/////////tiempo de ejecucion BVDF
	//nThreads =6;
	printf("BVDF\n");
	for (nThreads = 1; nThreads <= 6; nThreads++){
	printf("nThreads = %d\n", nThreads);

	for (int contador = 0; contador <= nExperimentos; contador++){

	start = clock();

	BVDF_CPU_Multi(h_out, h_in, N, M, nChanels, nThreads);
	//VMF_CPU_Multi(h_out, h_in,imageOriginal.rows, imageOriginal.cols,nChanels, nThreads);
	//DDF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, nThreads);
	//VectorUnit_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, nThreads);
	//FastDDF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, nThreads);
	//VMF_CPU_Multi_Optimizado(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 6);

	//VMF_CPU_Multi_Reuso(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VMF_CPU_Multi_Sort(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VMF_CPU_Multi_Reuso_Sort(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VMF_CPU_Multi_L1(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 6);

	//FastBVDF_CPU_Multi(h_out, h_in, N, M, nChanels, nThreads);

	//Detection_FuzzyMetric(h_Noise, h_in, N, M, nChanels, nThreads);
	//AMF_Filtering(h_out, h_in, h_Noise, N, M, nChanels, nThreads);
	//VMF_Filtering_Single(h_out, h_in, h_Noise, N, M, nChanels);
	
	//MPGFMF(h_out, h_in, N, M, nChanels, nThreads);// Esta es la propuesta PeerGroup

	diff = clock() - start;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	if (msec<msecMenor) msecMenor = msec;
	
	//printf("     contador = %d\n", contador);

	}
	printf("         nThreads = %d \n", nThreads);
	printf("         Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
	}
	*/
	/////////tiempo de ejecucion
	/*
	for (nThreads = 1; nThreads <= 6; nThreads++) {
		printf("nThreads = %d\n", nThreads);

		for (int contador = 0; contador <= nExperimentos; contador++) {
			printf("Contador = %d\n", contador);
			start = clock();

			BVDF_CPU_Multi(h_out, h_in, N, M, nChanels, nThreads);
			//VMF_CPU_Multi(h_out, h_in,imageOriginal.rows, imageOriginal.cols,nChanels, nThreads);
			//DDF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, nThreads);
			//VectorUnit_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, nThreads);
			//FastDDF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, nThreads);
			//VMF_CPU_Multi_Optimizado(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 6);

			//FastBVDF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, nThreads);

			//VMF_CPU_Multi_Reuso(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
			//VMF_CPU_Multi_Sort(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
			//VMF_CPU_Multi_Reuso_Sort(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
			//VMF_CPU_Multi_L1(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 6);

			//FastBVDF_CPU_Multi(h_out, h_in, N, M, nChanels, nThreads);

			//Detection_FuzzyMetric(h_Noise, h_in, N, M, nChanels, nThreads);
			//AMF_Filtering(h_out, h_in, h_Noise, N, M, nChanels, nThreads);
			//VMF_Filtering_Single(h_out, h_in, h_Noise, N, M, nChanels);
			//FTSCF_CPU_Multi_Original(h_out, h_in, N, M, nChanels, nThreads);

			//MPGFMF(h_out, h_in, N, M, nChanels, nThreads);// Esta es la propuesta PeerGroup

			diff = clock() - start;
			msec = diff * 1000 / CLOCKS_PER_SEC;
			if (msec<msecMenor) msecMenor = msec;

			//printf("     contador = %d\n", contador);

		}
		printf("         nThreads = %d \n", nThreads);
		printf("         Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
	}

	
	imagenOut.data = h_out;
	*/

	
	
	
	// Obtencion de PSNR
	
	
	for (contador = 0; contador <= 60; contador++){

	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512/";
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512_Aleatorio/";
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena320x320_Aleatorio/";
	char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512/";

	ObtenerPath(Dir, contador);
	Mat image = imread(Dir, IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen2" << std::endl; _getch(); return -1; }
	h_in = (unsigned char *)malloc(size);
	h_in = (unsigned char*)(image.data);
	//////////////////////////////// las lineas de arriva son para el PSNR

	BVDF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 6);
	//VMF_CPU_Multi(h_out, h_in,imageOriginal.rows, imageOriginal.cols,nChanels, 8);
	//DDF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VectorUnit_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//FastDDF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VMF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VMF_CPU_Single(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels);
	//VMF_CPU_Single_Optimizado(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels);
	//VMF_CPU_Multi_Optimizado(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels,8);
	//VMF_CPU_Multi_L1(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VMF_CPU_Multi_L2(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//FastBVDF_CPU_Multi(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 6);
	//VMF_CPU_Multi_Reuso(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VMF_CPU_Multi_Sort(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VMF_CPU_Multi_Reuso_Sort(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);
	//VMF_CPU_Multi_L1(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 8);

	//Detection_FuzzyMetric(h_Noise, h_in, N, M, nChanels, 6);
	//AMF_Filtering(h_out, h_in, h_Noise, N, M, nChanels, 6);
	//FiltradoPropuesta(h_out, h_in, h_Noise, N, M, nChanels, 6);
	//VMF_Filtering(h_out, h_in, h_Noise, N, M, nChanels, 6);
	//GVDF_Filter_CPU_Multi(h_out, h_in, N, M, nChanels);

	//FTSCF_CPU_Multi_Original(h_out, h_in, imageOriginal.rows, imageOriginal.cols, nChanels, 6);

	//MPGFMF(h_out, h_in, N, M, nChanels, 6);// Esta es la propuesta PeerGroup

	//FastBVDF_CPU_Multi(h_out, h_in, N, M, nChanels, nThreads);


	imagenOut.data = h_out;
	valPSNR[contador] = getPSNR(imageOriginal, imagenOut);
	valMCRE[contador] = getMCRE(imageOriginal, imagenOut);
	valNCD [contador] = getNCD(imageOriginal, imagenOut);
	valMAE [contador] = getMAE(imageOriginal, imagenOut);
	//valMCREMio[contador] = getMCRE_Mio(imageOriginal, imagenOut);
	printf("%d\n", contador);

	}
	
	
	
	// Pruebas FTSCF
	


	imshow("Imagen Filtrada Final", imagenOut);	waitKey(0); 


	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	imwrite("D:/Google Drive/Trabajo Doctorado/Resultados/Imagenes Filtradas/Mandril_FTSCF_5.bmp", imagenOut, compression_params);

	///////////




	EscribirCriterios(valPSNR, valMCRE, valNCD, valMAE, contador);

	free(h_in);
	free(h_out);

	return 0;
}

