#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include <time.h>
#include <iostream> // library that contain basic input/output functions
#include <fstream> 
#include <iomanip>
#include <string.h>
#include <algorithm>    // std::sort, max min


#include "FuncionesAux.h"
#include "FiltrosColor.cuh"


#define	M	512// horizontal, x   //estan bien estos columnnas
#define N	512// verticual, y   // filas

/*
#define	M	768// horizontal, x   //estan bien estos columnnas
#define N	512// verticual, y   // filas
*/

#define nChannels 3

#define Mask_width  3
#define Mask_radius Mask_width / 2
#define TILE_WIDTH  16
#define SIZE        (TILE_WIDTH + Mask_width - 1)





cudaError_t addWithCuda(unsigned char *dev_out, unsigned char *dev_in, int size); //salida,entrada y tamaño

using namespace cv;
using namespace std;


struct pixel
{
	float alpha;
	unsigned char R;
	unsigned char G;
	unsigned char B;
	int index;

};

__device__ int compareVMF(const struct pixel *a, const struct pixel *b)
{
	if (a->alpha < b->alpha) return -1;
	if (a->alpha == b->alpha) return 0;
	if (a->alpha > b->alpha) return 1;
}

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


__global__ void PeerGroup(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m) {
	//Calculate the row # of the d_Pin and d_Pout element to process 
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	// Calculate the column # of the d_Pin and d_Pout element to process 
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	// each thread computes one element of d_Pout if in range 
	// Se debe de checar si los pixeles esta dentro del intervalo de 8 bits
	int x = 0, posicion[9], hold2 = 0, F = 0;
	int K = 1024;
	float vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	float disteucl, disteucl1[9], hold;
	float  valAngulo = 0.0, r = 0.0;
	float arriva = 0.0, abajo = 0.0, val1, val2, val3, dist_M = 0;


	if ((Row < m - 1) && (Col < n - 1)) {


		//hacer el arreglo
		F = 0;

		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				vectR[F] = d_Pin[((Row + i) * n + (Col + j)) * 3 + 0];
				vectG[F] = d_Pin[((Row + i) * n + (Col + j)) * 3 + 1];
				vectB[F] = d_Pin[((Row + i) * n + (Col + j)) * 3 + 2];

				posicion[F] = F;
				F++;
			}
		}


		disteucl = 0;
		for (F = 0; F <= 8; F++) {
			arriva = min(vectR[F], vectR[4]) + K;
			abajo = max(vectR[F], vectR[4]) + K;
			val1 = arriva / abajo;

			arriva = min(vectG[F], vectG[4]) + K;
			abajo = max(vectG[F], vectG[4]) + K;
			val2 = arriva / abajo;

			arriva = min(vectB[F], vectB[4]) + K;
			abajo = max(vectB[F], vectB[4]) + K;
			val3 = arriva / abajo;
			dist_M = min(min(val1, val2), val3);

		}

		for (F = 0; F <= 8; F++) {
			for (x = 0; x <= 7; x++) {
				if (disteucl1[x] > disteucl1[x + 1]) {
					hold = disteucl1[x];
					hold2 = posicion[x];
					disteucl1[x] = disteucl1[x + 1];
					posicion[x] = posicion[x + 1];
					disteucl1[x + 1] = hold;
					posicion[x + 1] = hold2;
				}
			}
		}



		d_Pout[(Row * n + Col) * 3 + 0] = vectR[posicion[0]];
		d_Pout[(Row * n + Col) * 3 + 1] = vectG[posicion[0]];
		d_Pout[(Row * n + Col) * 3 + 2] = vectB[posicion[0]];

	}



}


int main()
{


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int msecMenor = 10000;
	float msec = 0, sumaT = 0;
	int nExperimentos = 60;
	double valPSNR[100], valMCRE[100], valMCREMio[100], valNCD[100], valMAE[100];



	Mat imageOriginal = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512/0.bmp", IMREAD_UNCHANGED);
	//Mat imageOriginal = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512/0.bmp", IMREAD_UNCHANGED);
	//Mat imageOriginal = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512_Aleatorio/0.bmp", IMREAD_UNCHANGED);

	//Mat imageOriginal = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/caps_768x512.png", IMREAD_UNCHANGED);
	//Mat imageOriginal = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena320x320_Aleatorio/0.bmp", IMREAD_UNCHANGED);
	if (!imageOriginal.data)          // Check for invalid input
	{
		cout << "No esta la imagen1" << std::endl;
		_getch();
		return -1;
	}

	int size = (N)*(M)* sizeof(unsigned char)* nChannels;

	//Se usa malloc para poder procesar imagenes grandes
	unsigned char *h_in;
	//h_in = (unsigned char *)malloc(size);
	//h_in = (unsigned char*)(imageOriginal.data);					// puntero a los datos de la imagenIn

	Mat imagenOut(N, M, CV_8UC3, Scalar(255));
	unsigned char *h_out;	h_out = (unsigned char *)malloc(size);
	imagenOut.data = h_out;

	//Noise
	Mat imagenOut_Noise(N, M, CV_8UC1, Scalar(255));
	unsigned char *h_out_Noise;	h_out_Noise = (unsigned char *)malloc(size);
	imagenOut_Noise.data = h_out_Noise;

	imshow("imagen de Original", imageOriginal);

	//obtencion de imagen de prueba e imagen de salida y puntero
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512_Aleatorio/0.bmp", CV_LOAD_IMAGE_UNCHANGED);

	char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512/";
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512_Aleatorio/";
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena320x320_Aleatorio/";
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512/";
	
	//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512_Aleatorio/";
	ObtenerPath(Dir, 10);

	
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/caps_10SyP.png", CV_LOAD_IMAGE_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	Mat image = imread(Dir, IMREAD_UNCHANGED); if (!image.data) { cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }

	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/lena1024x1024_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }

	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/lena2048x2048.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena2048x2048_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena4096x4096_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena8192x8192_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data) { cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena320x320_Ruido10.bmp", IMREAD_UNCHANGED); if (!image.data) { cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }


	h_in = (unsigned char*)(image.data);
	//
	unsigned char *dev_in, *dev_out, *Noise;

	if (cudaSuccess != cudaMalloc((void **)&dev_in, size))
	{
		printf("Error en cudaMalloc!\n");		_getch();
	}

	if (cudaSuccess != cudaMalloc((void **)&dev_out, size))
	{
		printf("Error en cudaMalloc!\n");		_getch();
	}
	if (cudaSuccess != cudaMalloc((void **)&Noise, size))
	{
		printf("Error en cudaMalloc!\n");		_getch();
	}
	

	//int nHilosporBloque = 4;
	//int nHilosporBloque = 8;//Tenia este
	int nHilosporBloque = 16;//con este funciona
	//int nHilosporBloque = 32;
	//int nHilosporBloque = 64;
	
	dim3 nThreads(nHilosporBloque, nHilosporBloque, 1);		// numeros de Hilos por bloque  (se selecciono asi aqui, tiene que ser un multiplo de 32)
	dim3 nBloques((M / nHilosporBloque) + 1, (N / nHilosporBloque) + 1, 1);

	//Copiar datos de Host a Device
	if (cudaSuccess != cudaMemcpy(dev_in, h_in, size, cudaMemcpyHostToDevice))
	{
		printf("Error!\n");		_getch();
	}

	//image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512/10.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen2" << std::endl; _getch(); return -1; }
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/lena2048x2048.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/lena1024x1024.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
	//image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/lena2048x2048.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }	
	//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/caps_20SyP.png", IMREAD_UNCHANGED);

	//h_in = (unsigned char *)malloc(size);
	//h_in = (unsigned char*)(image.data);

	/*
	if (cudaSuccess != cudaMemcpy(dev_in, h_in, size, cudaMemcpyHostToDevice))
	{
	printf("Error!\n");		_getch();
	}
	*/

	//MarginalMedianFilter_Global_Forgetfull << <nBloques, nThreads >> >(dev_out, dev_in, M, N);
	//VMF_Global_Forgetfull_Reuse << <nBloques, nThreads >> >(dev_out, dev_in, N,M );

	//Detection_FuzzyMetric << <nBloques, nThreads >> >(Noise, dev_in, N, M);
	//Detection_Euclidean << <nBloques, nThreads >> >(Noise, dev_in, N, M);
	//AMF_Filtering <<<nBloques, nThreads >>>(dev_out, dev_in, Noise, N, M);
	//VMF_Filtering << <nBloques, nThreads >> >(dev_out, dev_in, Noise, N, M);
	
	//FiltradoPropuesta << <nBloques, nThreads >> >(dev_out, dev_in, Noise, N, M);
	//VMF_GPU_GLOBAL << <nBloques, nThreads >> >(dev_out, dev_in, M, N);
	//FTSCF_GPU << <nBloques, nThreads >> >(dev_out, dev_in,10,60,.5, M, N);

	//FiltradoPropuesta_MMF << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	//FiltradoPropuesta_VMF   <<<nBloques, nThreads >> >(dev_out, dev_in, N, M);
	//FiltradoPropuesta_AMF << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	VectorUnit_GPU_Global << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	

	//FTSCF_GPU_Original << <nBloques, nThreads >> >(dev_out, dev_in, N, M);

	BVDF_GPU_Global << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	
	
	if (cudaSuccess != cudaMemcpy(h_out, dev_out, size, cudaMemcpyDeviceToHost))
	{
	printf("Error en copiar de Device a host!\n");	_getch();
	}

	imagenOut.data = h_out;
	imshow("Imagen Filtrada", imagenOut); waitKey();

	//Noise
	if (cudaSuccess != cudaMemcpy(h_out_Noise, Noise, size, cudaMemcpyDeviceToHost))
	{
	printf("Error en copiar de Device a host!\n");	_getch();
	}

	imagenOut_Noise.data = h_out_Noise;
	
	/*
	/////////tiempo de ejecucion
	for (int contador = 0; contador <= nExperimentos; contador++){

	cudaEventRecord(start);

	//Detection_FuzzyMetric << <nBloques, nThreads >> >(Noise, dev_in, N, M);
	//VMF_Filtering << <nBloques, nThreads >> >(dev_out, dev_in, Noise, N, M);
	//AMF_Filtering << <nBloques, nThreads >> >(dev_out, dev_in, Noise, N, M);
	//FiltradoPropuesta << <nBloques, nThreads >> >(dev_out, dev_in, Noise, N, M);
	//FiltradoPropuesta2 << <nBloques, nThreads >> >(dev_out, dev_in, M, N);
	//MarginalMedianFilter_Global_Forgetfull << <nBloques, nThreads >> >(dev_out, dev_in, M, N);
	//VMF_Global_Forgetfull_Reuse << <nBloques, nThreads >> >(dev_out, dev_in, M, N);
	//FiltradoPropuesta_MMF << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	//VMF_GPU_GLOBAL << <nBloques, nThreads >> >(dev_out, dev_in, M, N);
	//FTSCF_GPU << <nBloques, nThreads >> >(dev_out, dev_in, 0, 120, .5, M, N);

	VectorUnit_GPU_Global << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	//BVDF_GPU_Global << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	
	//FTSCF_GPU_Original << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	//FiltradoPropuesta_MMF << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	//FiltradoPropuesta_VMF << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
	//FiltradoPropuesta_AMF << <nBloques, nThreads >> >(dev_out, dev_in, N, M);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//if (milliseconds<msecMenor) msecMenor = milliseconds;
	printf("\n%f",milliseconds);

	}
	//printf("         Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
	//printf("\n%f",msecMenor);

	if ( cudaSuccess != cudaMemcpy(h_out, dev_out, size, cudaMemcpyDeviceToHost) )
	{printf( "Error en copiar de Device a host!\n" );	_getch();	}
	
	
	//Esto es para escribir imagenes en disco
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	
	
	*/
	
	// Obtencion de PSNR
	for (int contador = 0; contador <= nExperimentos; contador++) {

		char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512/";
		//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena512x512_Aleatorio/";
		//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512/";
		//char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Mandrill512x512_Aleatorio/";
		ObtenerPath(Dir, contador);
		Mat image = imread(Dir, IMREAD_UNCHANGED); if (!image.data) { cout << "No esta la imagen2" << std::endl; _getch(); return -1; }
		//Mat image = imread("C:/Users/AgustinQuadro4000/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/lena2048x2048.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }
		h_in = (unsigned char *)malloc(size);
		h_in = (unsigned char*)(image.data);
		if (cudaSuccess != cudaMemcpy(dev_in, h_in, size, cudaMemcpyHostToDevice))
		{
			printf("Error!\n");		_getch();
		}

		//Detection_FuzzyMetric << <nBloques, nThreads >> >(Noise, dev_in, N, M);
		//Detection_Euclidean << <nBloques, nThreads >> >(Noise, dev_in, N, M);
		//AMF_Filtering <<<nBloques, nThreads >>>(dev_out, dev_in, Noise, N, M);
		//VMF_Filtering << <nBloques, nThreads >> >(dev_out, dev_in, Noise, N, M);
		//FiltradoPropuesta << <nBloques, nThreads >> >(dev_out, dev_in, Noise, N, M);
		//FiltradoPropuesta2 << <nBloques, nThreads >> >(dev_out, dev_in, M, N);
		
		//MarginalMedianFilter_Global_Forgetfull << <nBloques, nThreads >> >(dev_out, dev_in, M, N);
		//VMF_Global_Forgetfull_Reuse << <nBloques, nThreads >> >(dev_out, dev_in, M, N);

		//FiltradoPropuesta_MMF << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
		//FTSCF_GPU_Original << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
		//FiltradoPropuesta_VMF   <<<nBloques, nThreads >> >(dev_out, dev_in, N, M);
		//FiltradoPropuesta_AMF << <nBloques, nThreads >> >(dev_out, dev_in, N, M);

		//FTSCF_GPU << <nBloques, nThreads >> >(dev_out, dev_in, 0, 120, .5, M, N);
		
		//FTSCF_GPU_Original << <nBloques, nThreads >> >(dev_out, dev_in, M, N);
		//FTSCF_GPU_Original_Params << <nBloques, nThreads >> >(dev_out, dev_in, M, N, 1, .8, .1, 60, 10, 1000);

		VectorUnit_GPU_Global << <nBloques, nThreads >> >(dev_out, dev_in, N, M);
		//BVDF_GPU_Global << <nBloques, nThreads >> >(dev_out, dev_in, N, M);

		if (cudaSuccess != cudaMemcpy(h_out, dev_out, size, cudaMemcpyDeviceToHost))
		{
			printf("Error en copiar de Device a host!\n");	_getch();
		}

		imagenOut.data = h_out;


		//Escribir una a una las imagenes resultantes

		//char Dir2[] = "D:/Google Drive/Trabajo Doctorado/Resultados/Imagenes Filtradas/SSIM_VSNR/PropuestaPeerGroup_Mandril512x512_RA/";
		//ObtenerPathWrite(Dir2, contador);
		//imwrite(Dir2, imagenOut, compression_params);


		valPSNR[contador] = getPSNR(imageOriginal, imagenOut);
		valMCRE[contador] = getMCRE(imageOriginal, imagenOut);
		//valMCRE[contador] = getMCRE_Mio(imageOriginal, imagenOut);
		valNCD[contador] = getNCD(imageOriginal, imagenOut);
		valMAE[contador] = getMAE(imageOriginal, imagenOut);
		//valMCREMio[contador] = getMCRE_Mio(imageOriginal, imagenOut);
		printf("%d\n", contador);

	}
	

	

	/*
	//valores optimos

	float med_1, var_1, med_2, med1, med2, var1, THS=0;
	float med_1_MAX, var_1_MAX, med_2_MAX, med1_MAX, med2_MAX, var1_MAX, THS_MAX;

	int THS_contador = 0, int contador =0;

	float valorPSNR[60];
	float MAX_PSNR = 0, Sum_PSNR = 0, Mejor_Sum_PSNR =0;
	int Iteracion = 0;
	
	unsigned char *h_inOpt; 
	unsigned char *h_outOpt;

	for (med_1 = 0.2; med_1 <= 1; med_1= med_1 + .05) {
		for (var_1 = 0.2; var_1 <= .8; var_1= var_1+ .05) {
			for (med_2 = 0.2; med_2 <= .8; med_2=med_2 + .05) {
				for (med1 = 10; med1 <= 150; med1++) {
					for (med2 = 10; med2 <= 150; med2++){
						for (var1 = 500; var1 <= 1000; var1 = var1 + 10){
							for (THS = 0; THS <= .5; THS=THS+0.1){
								printf("Iteracion=%d\n", Iteracion++);
								for (contador = 0; contador <= 60; contador++){
									Mat imagenOutOpt(N, M, CV_8UC3, Scalar(255));

									//unsigned char *h_inOpt = new(nothrow) unsigned char[ N * M * nChannels ];
									//unsigned char *h_outOpt = new(nothrow) unsigned char[N * M * nChannels];

									char Dir[] = "D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/Lena320x320_Aleatorio/";
									ObtenerPath(Dir, contador);
									Mat imageOpt = imread(Dir, IMREAD_UNCHANGED); if (!imageOpt.data){ cout << "No esta la imagen2" << std::endl; _getch(); return -1; }
									//Mat image = imread("D:/Google Drive/Trabajo Doctorado/VisualStudio y MAtlab/Matlab/lena2048x2048.bmp", IMREAD_UNCHANGED); if (!image.data){ cout << "No esta la imagen Dir11" << std::endl; _getch(); return -1; }

									h_inOpt = (unsigned char*)(imageOpt.data);
									h_outOpt = (unsigned char *)imagenOutOpt.data;
									
									if (cudaSuccess != cudaMemcpy(dev_in, h_inOpt, size, cudaMemcpyHostToDevice)){	printf("Error!\n");		_getch();}
									imageOpt.release();

									FTSCF_GPU_Original_Params << <nBloques, nThreads >> >(dev_out, dev_in, M, N, med_1, var_1, med_2, med1, med2, var1, THS);
		
									if (cudaSuccess != cudaMemcpy(h_outOpt, dev_out, size, cudaMemcpyDeviceToHost)){	printf("Error en copiar de Device a host!\n");	_getch();}
		
									valorPSNR[contador] = getPSNR(imageOriginal, imagenOutOpt);

									imageOpt.release();
									imagenOutOpt.release();

									//delete[] h_inOpt;
									//delete[] h_outOpt;

									Sum_PSNR = valorPSNR[contador] + Sum_PSNR;
									//printf("valorPSNR[%d] = %f\n", contador, valorPSNR[contador]);
																	
								}//Contador
								if (Sum_PSNR > Mejor_Sum_PSNR) {
									med_1_MAX = med_1;
									var_1_MAX = var_1;
									med_2_MAX = med_2;
									med1_MAX = med1;
									med2_MAX = med2;
									var1_MAX = var1;
									THS_MAX = THS;
						
									Mejor_Sum_PSNR = Sum_PSNR;

									printf("med_1=%f\n", med_1);
									printf("var_1=%f\n", var_1);
									printf("med_2=%f\n", med_2);
									printf("med1=%f\n", med1);
									printf("med2=%f\n", med2);
									printf("var1=%f\n", var1);

									printf("THS=%f  ", THS);

									printf("Sum_PSNR=%f\n", Sum_PSNR);
								}//if impresion
								Sum_PSNR = 0;
							}//THS
						}//var1
					}//med2
				}//med1
			}//med_2
		}//var_1
	}//med_1

	*/
	imagenOut.data = h_out;
	imshow("Imagen Ruido", image);
	imshow("Imagen Filtrada", imagenOut);

	//Noise
	imagenOut_Noise.data = h_out_Noise;

	imshow("Imagen Noise", imagenOut_Noise);	
	waitKey(0);

	//vector<int> compression_params;
	//compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	//compression_params.push_back(9);
	//	imwrite( "C:/Users/AgustinTortolero/Google Drive/Trabajo Doctorado/Resultados/Imagenes Filtradas/LenaPropuestaPeerMMF_SyP_05.bmp",imagenOut, compression_params );
	//imwrite("D:/Google Drive/Trabajo Doctorado/Resultados/Imagenes Filtradas/Noise_Lena05.bmp", imagenOut_Noise, compression_params);
	///////////




	EscribirCriterios(valPSNR, valMCRE, valNCD, valMAE, nExperimentos);
	cudaFree(dev_in);		cudaFree(dev_out);
	//cudaDeviceReset();

	free(h_in);				free(h_out);



	return 0;
}

