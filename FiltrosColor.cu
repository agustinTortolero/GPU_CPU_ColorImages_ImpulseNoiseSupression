include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"


__global__ void VMF_GPU_GLOBAL(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	float vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	float disteucl, disteucl1[9], hold;
	float  valAngulo = 0.0, r = 0.0;

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
			for (x = 0; x <= 8; x++) {
				disteucl += abs(vectB[F] - vectB[x]) + abs(vectG[F] - vectG[x]) + abs(vectR[F] - vectR[x]);
				//disteucl += sqrt(pow(vectB[F] - vectB[x], 2) + pow(vectG[F] - vectG[x], 2) + pow(vectR[F] - vectR[x], 2));
				//disteucl +=  (vectB[F]-vectB[x]) * (vectB[F]-vectB[x]) + (vectG[F]-vectG[x]) * (vectG[F]-vectG[x])+(vectR[F]-vectR[x]) * (vectR[F]-vectR[x]);
				//disteucl +=  (vectB[F]-vectB[x]);
				//disteucl += (vectB[F] - vectB[x])*(vectB[F] - vectB[x]) + (vectG[F] - vectG[x])*(vectG[F] - vectG[x]) + (vectR[F] - vectR[x])*(vectR[F] - vectR[x]);

			}
			disteucl1[F] = disteucl;
			disteucl = 0;
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


#define maxCUDA( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define minCUDA( a, b ) ( ((a) < (b)) ? (a) : (b) )

__global__ void Detection_FuzzyMetric(unsigned char* Noise, const unsigned char* d_Pin, int n, int m)
{
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	float vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	float disteucl, disteucl1[9], hold;
	float  valAngulo = 0.0, r = 0.0;
	float arriva = 0.0, abajo = 0.0, val1, val2, val3, dist_M = 0;
	unsigned int P = 0;
	const unsigned int K = 1024, q = 1;
	const float d = .95;

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
		for (F = 0; F <= 8; F++) {
			arriva = minCUDA(vectR[F], vectR[4]) + K;
			abajo = maxCUDA(vectR[F], vectR[4]) + K;
			val1 = arriva / abajo;

			arriva = minCUDA(vectG[F], vectG[4]) + K;
			abajo = maxCUDA(vectG[F], vectG[4]) + K;
			val2 = arriva / abajo;

			arriva = minCUDA(vectB[F], vectB[4]) + K;
			abajo = maxCUDA(vectB[F], vectB[4]) + K;
			val3 = arriva / abajo;

			dist_M = minCUDA(minCUDA(val1, val2), val3);
			if (dist_M>d)	P++;
		}

		if (P <= (q + 1)) {
			Noise[(Row * m + Col)] = 255;
		}
		else {
			Noise[(Row * m + Col)] = 0;
		}

	}
}

__device__ float Magnitud(unsigned char* VectR, unsigned char* VectG, unsigned char* VectB, unsigned int i, unsigned int j) {

	float distR = abs(VectR[i] - VectR[j]);
	float distG = abs(VectG[i] - VectG[j]);
	float distB = abs(VectB[i] - VectB[j]);

	return distR + distB + distG;

}
__global__ void Detection_Euclidean(unsigned char* Noise, const unsigned char* d_Pin, int n, int m)
{
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0;
	unsigned int F = 0;
	unsigned char vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	float disteucl, disteucl1[9], hold;
	float  valAngulo = 0.0, r = 0.0;
	float arriva = 0.0, abajo = 0.0, val1, val2, val3, dist_M = 0;
	unsigned int P = 0;
	const unsigned int K = 1024, q = 1, dEuclidiana = 45;
	const float d = .95;

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
		for (F = 0; F <= 8; F++) {

			dist_M = Magnitud(vectR, vectG, vectB, F, 4);
			if (dist_M>45)	P++;
		}

		if (P <= (q + 1)) {
			Noise[(Row * m + Col)] = 255;
		}
		else {
			Noise[(Row * m + Col)] = 0;
		}

	}
}



__global__ void AMF_Filtering(unsigned char* d_Pout, const unsigned char* d_Pin, unsigned char* Noise, int n, int m)
{
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	float vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	float disteucl, disteucl1[9], hold;
	float sumR = 0.0, sumG = 0.0, sumB = 0.0;
	unsigned int Div = 0;


	if ( (Row>1) && (Col>1) && ( Row < m - 1) && (Col < n - 1) ) {
		sumR = 0.0, sumG = 0.0, sumB = 0.0;
		
		if (Noise[(Row * m + Col)] == 255) {
			Div = 0;
			
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					
					if (Noise[((Row + i) * m + (Col + j))] == 0) {//solo los que no son Noise
						
						Div++;
						sumR += d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
						sumG += d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
						sumB += d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];
						
					}
					
					
					
				}
			}


			d_Pout[((Row*m) + Col) * 3 + 0] = sumR / Div;
			d_Pout[((Row*m) + Col) * 3 + 1] = sumG / Div;
			d_Pout[((Row*m) + Col) * 3 + 2] = sumB / Div;

		}//fin de if
		else {
			d_Pout[((Row*m) + Col) * 3 + 0] = d_Pin[((Row*m) + Col) * 3 + 0];
			d_Pout[((Row*m) + Col) * 3 + 1] = d_Pin[((Row*m) + Col) * 3 + 1];
			d_Pout[((Row*m) + Col) * 3 + 2] = d_Pin[((Row*m) + Col) * 3 + 2];

		}
		
	}
}

__global__ void VMF_Filtering(unsigned char* d_Pout, const unsigned char* d_Pin, unsigned char* Noise, int n, int m)
{
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	float vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	float disteucl, disteucl1[9], hold;
	unsigned char arrayFiltradoR[9], arrayFiltradoG[9], arrayFiltradoB[9];
	float mn, mx;
	int posMin = 0;

	int c = 0, i = 0, j = 0;
	unsigned char aux = 100;

	if ((Row>1) && (Col>1) && (Row < m - 1) && (Col < n - 1)) {
		if (Noise[(Row * m + Col)] == 255) {
			c = 0;
			F = 0;
			for (i = -1; i <= 1; i++) {
				for (j = -1; j <= 1; j++) {
					posicion[F] = 0;

					if (Noise[((Row + i) * m + (Col + j))] == 0) {//solo los que no son Noise

						arrayFiltradoR[c] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
						arrayFiltradoG[c] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
						arrayFiltradoB[c] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];
						aux = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];
						posicion[c] = c;
						c++;
					}
					F++;
				}
			}
			disteucl = 0;
			for (i = 0; i <= c - 1; i++) {
				disteucl = 0;
				for (j = 0; j <= c - 1; j++) {
					float distR = abs(arrayFiltradoR[i] - arrayFiltradoR[j]);
					float distG = abs(arrayFiltradoG[i] - arrayFiltradoG[j]);
					float distB = abs(arrayFiltradoB[i] - arrayFiltradoB[j]);
					disteucl += distR + distB + distG;

				}
				disteucl1[i] = disteucl;
			}
			mn = disteucl1[0];
			mx = disteucl1[0];
			posMin = 0;

			for (i = 0; i <= c - 1; i++)
			{
				if (mn>disteucl1[i])
				{
					mn = disteucl1[i];
					posMin = posicion[i];
				}
				else if (mx<disteucl1[i])
				{

				}
			}


			d_Pout[(Row * m + Col) * 3 + 0] = arrayFiltradoR[posMin];
			d_Pout[(Row * m + Col) * 3 + 1] = arrayFiltradoG[posMin];
			d_Pout[(Row * m + Col) * 3 + 2] = arrayFiltradoB[posMin];
		}//fin de if
		else {
			d_Pout[((Row*m) + Col) * 3 + 0] = d_Pin[((Row*m) + Col) * 3 + 0];
			d_Pout[((Row*m) + Col) * 3 + 1] = d_Pin[((Row*m) + Col) * 3 + 1];
			d_Pout[((Row*m) + Col) * 3 + 2] = d_Pin[((Row*m) + Col) * 3 + 2];

		}
	}
}




__global__ void FiltradoPropuesta(unsigned char* d_Pout, const unsigned char* d_Pin, unsigned char* Noise, int n, int m)
{
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	unsigned char vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	float disteucl, disteucl1[9], hold;
	unsigned char arrayFiltradoR[9], arrayFiltradoG[9], arrayFiltradoB[9];
	float mn, mx;
	int posMin = 0;
	int c = 0, i = 0, j = 0;
	unsigned char aux = 100;
	float D[40];
	if ((Row < m - 1) && (Col < n - 1)) {
		if (Noise[(Row * m + Col)] == 255) {
			c = 0;
			for (i = -1; i <= 1; i++) {
				for (j = -1; j <= 1; j++) {
					vectR[c] = d_Pin[((Row + i) * n + (Col + j)) * 3 + 0];
					vectG[c] = d_Pin[((Row + i) * n + (Col + j)) * 3 + 1];
					vectB[c] = d_Pin[((Row + i) * n + (Col + j)) * 3 + 2];

					posicion[c] = c;
					c++;
				}
			}
			//D[0]=Magnitud(vectR, vectG, vectB, i, j//i==0 y j==0 no se hace
			D[0] = (Magnitud(vectR, vectG, vectB, 0, 1));
			D[1] = (Magnitud(vectR, vectG, vectB, 0, 2));
			D[2] = (Magnitud(vectR, vectG, vectB, 0, 3));
			D[3] = (Magnitud(vectR, vectG, vectB, 0, 4));
			D[4] = (Magnitud(vectR, vectG, vectB, 0, 5));
			D[5] = (Magnitud(vectR, vectG, vectB, 0, 6));
			D[6] = (Magnitud(vectR, vectG, vectB, 0, 7));
			D[7] = (Magnitud(vectR, vectG, vectB, 0, 8));
			disteucl1[0] = D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7];

			//i=1,j=0 ya esta es D[0]
			//i=1,j=1 No se hace
			D[8] = (Magnitud(vectR, vectG, vectB, 1, 2));
			D[9] = (Magnitud(vectR, vectG, vectB, 1, 3));
			D[10] = (Magnitud(vectR, vectG, vectB, 1, 4));
			D[11] = (Magnitud(vectR, vectG, vectB, 1, 5));
			D[12] = (Magnitud(vectR, vectG, vectB, 1, 6));
			D[13] = (Magnitud(vectR, vectG, vectB, 1, 7));
			D[14] = (Magnitud(vectR, vectG, vectB, 1, 8));
			disteucl1[1] = D[0] + D[8] + D[9] + D[10] + D[11] + D[12] + D[13] + D[14];

			//i=2,j=0 ya esta es D[1]
			//i=2,j=1 ya esta es D[8]
			//i=2,j=2 No se hace
			D[15] = (Magnitud(vectR, vectG, vectB, 2, 3));
			D[16] = (Magnitud(vectR, vectG, vectB, 2, 4));
			D[17] = (Magnitud(vectR, vectG, vectB, 2, 5));
			D[18] = (Magnitud(vectR, vectG, vectB, 2, 6));
			D[19] = (Magnitud(vectR, vectG, vectB, 2, 7));
			D[20] = (Magnitud(vectR, vectG, vectB, 2, 8));
			disteucl1[2] = D[1] + D[8] + D[15] + D[16] + D[17] + D[18] + D[19] + D[20];

			//i=3,j=0 ya esta es D[2]
			//i=3,j=1 ya esta es D[9]
			//i=3,j=2 ya esta es D[15]
			//i=3,j=3 No se hace
			D[21] = (Magnitud(vectR, vectG, vectB, 3, 4));
			D[22] = (Magnitud(vectR, vectG, vectB, 3, 5));
			D[23] = (Magnitud(vectR, vectG, vectB, 3, 6));
			D[24] = (Magnitud(vectR, vectG, vectB, 3, 7));
			D[25] = (Magnitud(vectR, vectG, vectB, 3, 8));
			disteucl1[3] = D[2] + D[9] + D[15] + D[21] + D[22] + D[23] + D[24] + D[25];

			//i=4,j=0 ya esta es D[3]
			//i=4,j=1 ya esta es D[10]
			//i=4,j=2 ya esta es D[16]
			//i=4,j=3 ya esta es D[21]
			//i=4,j=4 No se hace
			D[26] = (Magnitud(vectR, vectG, vectB, 4, 5));
			D[27] = (Magnitud(vectR, vectG, vectB, 4, 6));
			D[28] = (Magnitud(vectR, vectG, vectB, 4, 7));
			D[29] = (Magnitud(vectR, vectG, vectB, 4, 8));
			disteucl1[4] = D[3] + D[10] + D[16] + D[21] + D[26] + D[27] + D[28] + D[29];

			//i=5,j=0 ya esta es D[4]
			//i=5,j=1 ya esta es D[11]
			//i=5,j=2 ya esta es D[17]
			//i=5,j=3 ya esta es D[22]
			//i=5,j=4 ya esta es D[26]
			//i=5,j=5 No se hace
			D[30] = (Magnitud(vectR, vectG, vectB, 5, 6));
			D[31] = (Magnitud(vectR, vectG, vectB, 5, 7));
			D[32] = (Magnitud(vectR, vectG, vectB, 5, 8));
			disteucl1[5] = D[4] + D[11] + D[17] + D[22] + D[26] + D[30] + D[31] + D[32];

			//i=6,j=0 ya esta es D[5]
			//i=6,j=1 ya esta es D[12]
			//i=6,j=2 ya esta es D[18]
			//i=6,j=3 ya esta es D[23]
			//i=6,j=4 ya esta es D[27]
			//i=6,j=5 ya esta es D[30]
			//i=6,j=6 No se hace
			D[33] = (Magnitud(vectR, vectG, vectB, 6, 7));
			D[34] = (Magnitud(vectR, vectG, vectB, 6, 8));
			disteucl1[6] = D[5] + D[12] + D[18] + D[23] + D[27] + D[30] + D[33] + D[34];

			//i=7,j=0 ya esta es D[6]
			//i=7,j=1 ya esta es D[13]
			//i=7,j=2 ya esta es D[19]
			//i=7,j=3 ya esta es D[24]
			//i=7,j=4 ya esta es D[28]
			//i=7,j=5 ya esta es D[31]
			//i=7,j=6 ya esta es D[33]
			//i=7,j=7 No se hace
			D[35] = (Magnitud(vectR, vectG, vectB, 7, 8));
			disteucl1[7] = D[6] + D[13] + D[19] + D[24] + D[28] + D[31] + D[33] + D[35];

			//i=8,j=0 ya esta es D[7]
			//i=8,j=1 ya esta es D[14]
			//i=8,j=2 ya esta es D[20]
			//i=8,j=3 ya esta es D[25]
			//i=8,j=4 ya esta es D[29]
			//i=8,j=5 ya esta es D[32]
			//i=8,j=6 ya esta es D[34]
			//i=8,j=7 ya esta es D[35]
			//i=8,j=8 No se hace
			disteucl1[8] = D[7] + D[14] + D[20] + D[25] + D[29] + D[32] + D[34] + D[35];

			mn = disteucl1[0];
			mx = disteucl1[0];

			posMin = 0;

			for (int i = 0; i<8; i++)
			{
				if (mn>disteucl1[i])
				{
					mn = disteucl1[i];
					posMin = posicion[i];
				}
				else if (mx<disteucl1[i])
				{

				}
			}

			d_Pout[(Row * m + Col) * 3 + 0] = vectR[posMin];
			d_Pout[(Row * m + Col) * 3 + 1] = vectG[posMin];
			d_Pout[(Row * m + Col) * 3 + 2] = vectB[posMin];
		}//fin de if (Noise[(Row * m + Col)] == 255)
		else {
			d_Pout[((Row*m) + Col) * 3 + 0] = d_Pin[((Row*m) + Col) * 3 + 0];
			d_Pout[((Row*m) + Col) * 3 + 1] = d_Pin[((Row*m) + Col) * 3 + 1];
			d_Pout[((Row*m) + Col) * 3 + 2] = d_Pin[((Row*m) + Col) * 3 + 2];

		}
	}
}

__global__ void FiltradoPropuesta2(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	unsigned char vectR[9], vectG[9], vectB[9];
	float disteucl, disteucl1[9], hold, D[40];
	float  valAngulo = 0.0, r = 0.0;
	float arriva = 0.0, abajo = 0.0, val1, val2, val3, dist_M = 0, Noise = 0.0;
	unsigned int P = 0;
	const unsigned int K = 1024, q = 1;
	const float d = .95;

	float mn, mx;
	int posMin = 0;
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
		for (F = 0; F <= 8; F++) {
			arriva = minCUDA(vectR[F], vectR[4]) + K;
			abajo = maxCUDA(vectR[F], vectR[4]) + K;
			val1 = arriva / abajo;

			arriva = minCUDA(vectG[F], vectG[4]) + K;
			abajo = maxCUDA(vectG[F], vectG[4]) + K;
			val2 = arriva / abajo;

			arriva = minCUDA(vectB[F], vectB[4]) + K;
			abajo = maxCUDA(vectB[F], vectB[4]) + K;
			val3 = arriva / abajo;

			dist_M = minCUDA(minCUDA(val1, val2), val3);
			if (dist_M>d)	P++;
		}

		if (P <= (q + 1)) {
			Noise = 255;
		}
		else {
			Noise = 0;
		}
		if (Noise == 255) {
			//D[0]=Magnitud(vectR, vectG, vectB, i, j//i==0 y j==0 no se hace
			D[0] = (Magnitud(vectR, vectG, vectB, 0, 1));
			D[1] = (Magnitud(vectR, vectG, vectB, 0, 2));
			D[2] = (Magnitud(vectR, vectG, vectB, 0, 3));
			D[3] = (Magnitud(vectR, vectG, vectB, 0, 4));
			D[4] = (Magnitud(vectR, vectG, vectB, 0, 5));
			D[5] = (Magnitud(vectR, vectG, vectB, 0, 6));
			D[6] = (Magnitud(vectR, vectG, vectB, 0, 7));
			D[7] = (Magnitud(vectR, vectG, vectB, 0, 8));
			disteucl1[0] = D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7];

			//i=1,j=0 ya esta es D[0]
			//i=1,j=1 No se hace
			D[8] = (Magnitud(vectR, vectG, vectB, 1, 2));
			D[9] = (Magnitud(vectR, vectG, vectB, 1, 3));
			D[10] = (Magnitud(vectR, vectG, vectB, 1, 4));
			D[11] = (Magnitud(vectR, vectG, vectB, 1, 5));
			D[12] = (Magnitud(vectR, vectG, vectB, 1, 6));
			D[13] = (Magnitud(vectR, vectG, vectB, 1, 7));
			D[14] = (Magnitud(vectR, vectG, vectB, 1, 8));
			disteucl1[1] = D[0] + D[8] + D[9] + D[10] + D[11] + D[12] + D[13] + D[14];

			//i=2,j=0 ya esta es D[1]
			//i=2,j=1 ya esta es D[8]
			//i=2,j=2 No se hace
			D[15] = (Magnitud(vectR, vectG, vectB, 2, 3));
			D[16] = (Magnitud(vectR, vectG, vectB, 2, 4));
			D[17] = (Magnitud(vectR, vectG, vectB, 2, 5));
			D[18] = (Magnitud(vectR, vectG, vectB, 2, 6));
			D[19] = (Magnitud(vectR, vectG, vectB, 2, 7));
			D[20] = (Magnitud(vectR, vectG, vectB, 2, 8));
			disteucl1[2] = D[1] + D[8] + D[15] + D[16] + D[17] + D[18] + D[19] + D[20];

			//i=3,j=0 ya esta es D[2]
			//i=3,j=1 ya esta es D[9]
			//i=3,j=2 ya esta es D[15]
			//i=3,j=3 No se hace
			D[21] = (Magnitud(vectR, vectG, vectB, 3, 4));
			D[22] = (Magnitud(vectR, vectG, vectB, 3, 5));
			D[23] = (Magnitud(vectR, vectG, vectB, 3, 6));
			D[24] = (Magnitud(vectR, vectG, vectB, 3, 7));
			D[25] = (Magnitud(vectR, vectG, vectB, 3, 8));
			disteucl1[3] = D[2] + D[9] + D[15] + D[21] + D[22] + D[23] + D[24] + D[25];

			//i=4,j=0 ya esta es D[3]
			//i=4,j=1 ya esta es D[10]
			//i=4,j=2 ya esta es D[16]
			//i=4,j=3 ya esta es D[21]
			//i=4,j=4 No se hace
			D[26] = (Magnitud(vectR, vectG, vectB, 4, 5));
			D[27] = (Magnitud(vectR, vectG, vectB, 4, 6));
			D[28] = (Magnitud(vectR, vectG, vectB, 4, 7));
			D[29] = (Magnitud(vectR, vectG, vectB, 4, 8));
			disteucl1[4] = D[3] + D[10] + D[16] + D[21] + D[26] + D[27] + D[28] + D[29];

			//i=5,j=0 ya esta es D[4]
			//i=5,j=1 ya esta es D[11]
			//i=5,j=2 ya esta es D[17]
			//i=5,j=3 ya esta es D[22]
			//i=5,j=4 ya esta es D[26]
			//i=5,j=5 No se hace
			D[30] = (Magnitud(vectR, vectG, vectB, 5, 6));
			D[31] = (Magnitud(vectR, vectG, vectB, 5, 7));
			D[32] = (Magnitud(vectR, vectG, vectB, 5, 8));
			disteucl1[5] = D[4] + D[11] + D[17] + D[22] + D[26] + D[30] + D[31] + D[32];

			//i=6,j=0 ya esta es D[5]
			//i=6,j=1 ya esta es D[12]
			//i=6,j=2 ya esta es D[18]
			//i=6,j=3 ya esta es D[23]
			//i=6,j=4 ya esta es D[27]
			//i=6,j=5 ya esta es D[30]
			//i=6,j=6 No se hace
			D[33] = (Magnitud(vectR, vectG, vectB, 6, 7));
			D[34] = (Magnitud(vectR, vectG, vectB, 6, 8));
			disteucl1[6] = D[5] + D[12] + D[18] + D[23] + D[27] + D[30] + D[33] + D[34];

			//i=7,j=0 ya esta es D[6]
			//i=7,j=1 ya esta es D[13]
			//i=7,j=2 ya esta es D[19]
			//i=7,j=3 ya esta es D[24]
			//i=7,j=4 ya esta es D[28]
			//i=7,j=5 ya esta es D[31]
			//i=7,j=6 ya esta es D[33]
			//i=7,j=7 No se hace
			D[35] = (Magnitud(vectR, vectG, vectB, 7, 8));
			disteucl1[7] = D[6] + D[13] + D[19] + D[24] + D[28] + D[31] + D[33] + D[35];

			//i=8,j=0 ya esta es D[7]
			//i=8,j=1 ya esta es D[14]
			//i=8,j=2 ya esta es D[20]
			//i=8,j=3 ya esta es D[25]
			//i=8,j=4 ya esta es D[29]
			//i=8,j=5 ya esta es D[32]
			//i=8,j=6 ya esta es D[34]
			//i=8,j=7 ya esta es D[35]
			//i=8,j=8 No se hace
			disteucl1[8] = D[7] + D[14] + D[20] + D[25] + D[29] + D[32] + D[34] + D[35];

			mn = disteucl1[0];
			mx = disteucl1[0];

			posMin = 0;

			for (int i = 0; i<8; i++)
			{
				if (mn>disteucl1[i])
				{
					mn = disteucl1[i];
					posMin = posicion[i];
				}
				else if (mx < disteucl1[i])
				{

				}
			}

			d_Pout[(Row * m + Col) * 3 + 0] = vectR[posMin];
			d_Pout[(Row * m + Col) * 3 + 1] = vectG[posMin];
			d_Pout[(Row * m + Col) * 3 + 2] = vectB[posMin];
		}
		else {

			d_Pout[((Row*m) + Col) * 3 + 0] = vectR[4];
			d_Pout[((Row*m) + Col) * 3 + 1] = vectG[4];
			d_Pout[((Row*m) + Col) * 3 + 2] = vectB[4];
		}

	}//if de Row y Col

}//cierre de funcion



__device__ inline void s(unsigned char* a, unsigned char*b)
{
	int tmp;
	if (*a>*b) {//si a es mayor a b, se intercambian a y b.
		tmp = *b;
		*b = *a;
		*a = tmp;
	}
}

#define min3(a,b,c) s(a, b); s(a,c);
#define max3(a,b,c) s(b, c); s(a,c);

#define minmax3(a,b,c)			max3(a, b, c); s(a,b);
#define minmax4(a,b,c,d)		s(a, b); s(c,d);s(a, c); s(b,d);
#define minmax5(a,b,c,d,e)		s(a, b); s(c,d);min3(a,c,e);max3(b,d,e);

#define minmax6(a,b,c,d,e,f)	s(a,d);s(b,e);s(c,f);min3(a,b,c);max3(d,e,f);

__global__ void MarginalMedianFilter_Global_Forgetfull(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m) {
	//Calculate the row # of the d_Pin and d_Pout element to process 
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	// Calculate the column # of the d_Pin and d_Pout element to process 
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	// each thread computes one element of d_Pout if in range 
	// Se debe de checar si los pixeles esta dentro del intervalo de 8 bits
	int x = 0, c = 0, d = 0, F, canal;

	int i, j;
	unsigned char vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	unsigned char swap;
	unsigned char a0, a1, a2, a3, a4, a5;

	if ((Row < m - 1) && (Col < n - 1)) {

		vectR[0] = d_Pin[((Row - 1) * n + (Col - 1)) * 3 + 0];
		vectG[0] = d_Pin[((Row - 1) * n + (Col - 1)) * 3 + 1];
		vectB[0] = d_Pin[((Row - 1) * n + (Col - 1)) * 3 + 2];

		vectR[1] = d_Pin[((Row - 1) * n + (Col + 0)) * 3 + 0];
		vectG[1] = d_Pin[((Row - 1) * n + (Col + 0)) * 3 + 1];
		vectB[1] = d_Pin[((Row - 1) * n + (Col + 0)) * 3 + 2];

		vectR[2] = d_Pin[((Row - 1) * n + (Col + 1)) * 3 + 0];
		vectG[2] = d_Pin[((Row - 1) * n + (Col + 1)) * 3 + 1];
		vectB[2] = d_Pin[((Row - 1) * n + (Col + 1)) * 3 + 2];

		vectR[3] = d_Pin[((Row + 0) * n + (Col - 1)) * 3 + 0];
		vectG[3] = d_Pin[((Row + 0) * n + (Col - 1)) * 3 + 1];
		vectB[3] = d_Pin[((Row + 0) * n + (Col - 1)) * 3 + 2];

		vectR[4] = d_Pin[((Row + 0) * n + (Col + 0)) * 3 + 0];
		vectG[4] = d_Pin[((Row + 0) * n + (Col + 0)) * 3 + 1];//central
		vectB[4] = d_Pin[((Row + 0) * n + (Col + 0)) * 3 + 2];

		vectR[5] = d_Pin[((Row + 0) * n + (Col + 1)) * 3 + 0];
		vectG[5] = d_Pin[((Row + 0) * n + (Col + 1)) * 3 + 1];
		vectB[5] = d_Pin[((Row + 0) * n + (Col + 1)) * 3 + 2];

		minmax6(&vectR[0], &vectR[1], &vectR[2], &vectR[3], &vectR[4], &vectR[5]);
		minmax6(&vectG[0], &vectG[1], &vectG[2], &vectG[3], &vectG[4], &vectG[5]);
		minmax6(&vectB[0], &vectB[1], &vectB[2], &vectB[3], &vectB[4], &vectB[5]);
		vectR[5] = d_Pin[((Row + 1) * n + (Col - 1)) * 3 + 0];
		vectG[5] = d_Pin[((Row + 1) * n + (Col - 1)) * 3 + 1];
		vectB[5] = d_Pin[((Row + 1) * n + (Col - 1)) * 3 + 2];

		minmax5(&vectR[1], &vectR[2], &vectR[3], &vectR[4], &vectR[5]);
		minmax5(&vectG[1], &vectG[2], &vectG[3], &vectG[4], &vectG[5]);
		minmax5(&vectB[1], &vectB[2], &vectB[3], &vectB[4], &vectB[5]);
		vectR[5] = d_Pin[((Row + 1) * n + (Col + 0)) * 3 + 0];
		vectG[5] = d_Pin[((Row + 1) * n + (Col + 0)) * 3 + 1];
		vectB[5] = d_Pin[((Row + 1) * n + (Col + 0)) * 3 + 2];

		minmax4(&vectR[2], &vectR[3], &vectR[4], &vectR[5]);
		minmax4(&vectG[2], &vectG[3], &vectG[4], &vectG[5]);
		minmax4(&vectB[2], &vectB[3], &vectB[4], &vectB[5]);
		vectR[5] = d_Pin[((Row + 1) * n + (Col + 1)) * 3 + 0];
		vectG[5] = d_Pin[((Row + 1) * n + (Col + 1)) * 3 + 1];
		vectB[5] = d_Pin[((Row + 1) * n + (Col + 1)) * 3 + 2];

		minmax3(&vectR[3], &vectR[4], &vectR[5]);
		minmax3(&vectG[3], &vectG[4], &vectG[5]);
		minmax3(&vectB[3], &vectB[4], &vectB[5]);


		d_Pout[(Row * m + Col) * 3 + 0] = vectR[4]; // ojo aqui va desde 0 a 8
		d_Pout[(Row * m + Col) * 3 + 1] = vectG[4];
		d_Pout[(Row * m + Col) * 3 + 2] = vectB[4];

	}

}


//este es la propuesta
__global__ void FiltradoPropuesta_MMF(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	unsigned char vectR[9], vectG[9], vectB[9];
	float disteucl, disteucl1[9], hold, D[40];
	float  valAngulo = 0.0, r = 0.0;
	float arriva = 0.0, abajo = 0.0, val1, val2, val3, dist_M = 0, Noise = 0.0;
	unsigned int P = 0;
	const unsigned int K = 1024, q = 1;
	const float d = .95;

	float mn, mx;
	int posMin = 0;

	if ((Row < m - 1) && (Col < n - 1)) {
		//hacer el arreglo
		F = 0;
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				vectR[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
				vectG[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
				vectB[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];

				posicion[F] = F;
				F++;
			}
		}
		for (F = 0; F <= 8; F++) {
			arriva = minCUDA(vectR[F], vectR[4]) + K;
			abajo = maxCUDA(vectR[F], vectR[4]) + K;
			val1 = arriva / abajo;

			arriva = minCUDA(vectG[F], vectG[4]) + K;
			abajo = maxCUDA(vectG[F], vectG[4]) + K;
			val2 = arriva / abajo;

			arriva = minCUDA(vectB[F], vectB[4]) + K;
			abajo = maxCUDA(vectB[F], vectB[4]) + K;
			val3 = arriva / abajo;

			dist_M = minCUDA(minCUDA(val1, val2), val3);
			if (dist_M>d)	P++;
		}

		if (P <= (q + 1)) {
			Noise = 255;
		}
		else {
			Noise = 0;
		}
		if (Noise == 255) {

			minmax6(&vectR[0], &vectR[1], &vectR[2], &vectR[3], &vectR[4], &vectR[5]);
			minmax6(&vectG[0], &vectG[1], &vectG[2], &vectG[3], &vectG[4], &vectG[5]);
			minmax6(&vectB[0], &vectB[1], &vectB[2], &vectB[3], &vectB[4], &vectB[5]);
			vectR[5] = vectR[6];
			vectG[5] = vectG[6];
			vectB[5] = vectB[6];

			minmax5(&vectR[1], &vectR[2], &vectR[3], &vectR[4], &vectR[5]);
			minmax5(&vectG[1], &vectG[2], &vectG[3], &vectG[4], &vectG[5]);
			minmax5(&vectB[1], &vectB[2], &vectB[3], &vectB[4], &vectB[5]);
			vectR[5] = vectR[7];
			vectG[5] = vectG[7];
			vectB[5] = vectB[7];

			minmax4(&vectR[2], &vectR[3], &vectR[4], &vectR[5]);
			minmax4(&vectG[2], &vectG[3], &vectG[4], &vectG[5]);
			minmax4(&vectB[2], &vectB[3], &vectB[4], &vectB[5]);
			vectR[5] = vectR[8];
			vectG[5] = vectG[8];
			vectB[5] = vectB[8];

			minmax3(&vectR[3], &vectR[4], &vectR[5]);
			minmax3(&vectG[3], &vectG[4], &vectG[5]);
			minmax3(&vectB[3], &vectB[4], &vectB[5]);


			d_Pout[(Row * m + Col) * 3 + 0] = vectR[4]; // ojo aqui va desde 0 a 8
			d_Pout[(Row * m + Col) * 3 + 1] = vectG[4];
			d_Pout[(Row * m + Col) * 3 + 2] = vectB[4];

		}
		else {

			d_Pout[((Row*m) + Col) * 3 + 0] = vectR[4];
			d_Pout[((Row*m) + Col) * 3 + 1] = vectG[4];
			d_Pout[((Row*m) + Col) * 3 + 2] = vectB[4];
		}

	}//if de Row y Col

}//cierre de funcion
 //propuesta con filtrado VMF
__global__ void FiltradoPropuesta_VMF(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0, i = 0, c = 0;
	unsigned char vectR[9], vectG[9], vectB[9];
	float disteucl, disteucl1[9], hold, D[40];
	float  valAngulo = 0.0, r = 0.0;
	float arriva = 0.0, abajo = 0.0, val1, val2, val3, dist_M = 0, Noise = 0.0;
	unsigned int P = 0;
	const unsigned int K = 1024, q = 1;
	const float d = .95;

	float mn, mx;
	int posMin = 0;

	if ((Row < m - 1) && (Col < n - 1)) {
		//hacer el arreglo
		F = 0;
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				vectR[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
				vectG[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
				vectB[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];

				posicion[F] = F;
				F++;
			}
		}
		for (F = 0; F <= 8; F++) {
			arriva = minCUDA(vectR[F], vectR[4]) + K;
			abajo = maxCUDA(vectR[F], vectR[4]) + K;
			val1 = arriva / abajo;

			arriva = minCUDA(vectG[F], vectG[4]) + K;
			abajo = maxCUDA(vectG[F], vectG[4]) + K;
			val2 = arriva / abajo;

			arriva = minCUDA(vectB[F], vectB[4]) + K;
			abajo = maxCUDA(vectB[F], vectB[4]) + K;
			val3 = arriva / abajo;

			dist_M = minCUDA(minCUDA(val1, val2), val3);
			if (dist_M>d)	P++;
		}

		if (P <= (q + 1)) {
			Noise = 255;
		}
		else {
			Noise = 0;
		}
		if (Noise == 255) {
			disteucl = 0;
			for (F = 0; F <= 8; F++) {
				for (x = 0; x <= 8; x++) {
					//disteucl += abs(vectB[F] - vectB[x]) + abs(vectG[F] - vectG[x]) + abs(vectR[F] - vectR[x]);
					//disteucl += sqrt(pow(vectB[F] - vectB[x], 2) + pow(vectG[F] - vectG[x], 2) + pow(vectR[F] - vectR[x], 2));
					//disteucl +=  (vectB[F]-vectB[x]) * (vectB[F]-vectB[x]) + (vectG[F]-vectG[x]) * (vectG[F]-vectG[x])+(vectR[F]-vectR[x]) * (vectR[F]-vectR[x]);
					//disteucl +=  (vectB[F]-vectB[x]);
					//disteucl += (vectB[F] - vectB[x])*(vectB[F] - vectB[x]) + (vectG[F] - vectG[x])*(vectG[F] - vectG[x]) + (vectR[F] - vectR[x])*(vectR[F] - vectR[x]);
					float distR = abs(vectR[F] - vectR[x]);
					float distG = abs(vectG[F] - vectG[x]);
					float distB = abs(vectB[F] - vectB[x]);
					disteucl += distR + distB + distG;
				}
				disteucl1[F] = disteucl;
				disteucl = 0;
			}

			mn = disteucl1[0];
			mx = disteucl1[0];

			posMin = 0;

			for (i = 0; i<8; i++)
			{
				if (mn>disteucl1[i])
				{
					mn = disteucl1[i];
					posMin = posicion[i];
				}
				else if (mx<disteucl1[i])
				{

				}
			}

			d_Pout[(Row * m + Col) * 3 + 0] = vectR[posMin]; // ojo aqui va desde 0 a 8
			d_Pout[(Row * m + Col) * 3 + 1] = vectG[posMin];
			d_Pout[(Row * m + Col) * 3 + 2] = vectB[posMin];



		}
		else {

			d_Pout[((Row*m) + Col) * 3 + 0] = vectR[4];
			d_Pout[((Row*m) + Col) * 3 + 1] = vectG[4];
			d_Pout[((Row*m) + Col) * 3 + 2] = vectB[4];
		}

	}//if de Row y Col

}//cierre de funcion
__global__ void FiltradoPropuesta_AMF(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0, i = 0, c = 0;
	unsigned char vectR[9], vectG[9], vectB[9];
	float disteucl, disteucl1[9], hold, D[40];
	float  valAngulo = 0.0, r = 0.0;
	float arriva = 0.0, abajo = 0.0, val1, val2, val3, dist_M = 0, Noise = 0.0;
	unsigned int P = 0;
	const unsigned int K = 1024, q = 1;
	const float d = .95;
	unsigned int Div = 0;
	float mn, mx;
	int posMin = 0;
	float sumR = 0.0, sumG = 0.0, sumB = 0.0;

	if ((Row>1) && (Col>1) && (Row < m - 1) && (Col < n - 1)) {
		//hacer el arreglo
		F = 0;
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				vectR[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
				vectG[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
				vectB[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];

				posicion[F] = F;
				F++;
			}
		}
		for (F = 0; F <= 8; F++) {
			arriva = minCUDA(vectR[F], vectR[4]) + K;
			abajo = maxCUDA(vectR[F], vectR[4]) + K;
			val1 = arriva / abajo;

			arriva = minCUDA(vectG[F], vectG[4]) + K;
			abajo = maxCUDA(vectG[F], vectG[4]) + K;
			val2 = arriva / abajo;

			arriva = minCUDA(vectB[F], vectB[4]) + K;
			abajo = maxCUDA(vectB[F], vectB[4]) + K;
			val3 = arriva / abajo;

			dist_M = minCUDA(minCUDA(val1, val2), val3);
			if (dist_M>d)	P++;
		}

		if (P <= (q + 1)) {
			Noise = 255;
		}
		else {
			Noise = 0;
		}
		if (Noise == 255) {

			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {

					sumR += d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
					sumG += d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
					sumB += d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];

				}
			}


			d_Pout[((Row*m) + Col) * 3 + 0] = sumR / 9;
			d_Pout[((Row*m) + Col) * 3 + 1] = sumG / 9;
			d_Pout[((Row*m) + Col) * 3 + 2] = sumB / 9;
		}
		else {

			d_Pout[((Row*m) + Col) * 3 + 0] = vectR[4];
			d_Pout[((Row*m) + Col) * 3 + 1] = vectG[4];
			d_Pout[((Row*m) + Col) * 3 + 2] = vectB[4];
		}

	}//if de Row y Col

}//cierre de funcion


__global__ void VMF_Global_Forgetfull_Reuse(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m)
{
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	unsigned char vectR[9], vectG[9], vectB[9];
	float disteucl = 0.0, disteucl1[9], hold;
	float D[40];
	float mn, mx;
	int posMin = 0;


	if ((Row < m - 1) && (Col < n - 1)) {
		F = 0;

		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				vectR[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
				vectG[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
				vectB[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];

				posicion[F] = F;
				F++;
			}
		}
		//D[0]=Magnitud(vectR, vectG, vectB, i, j//i==0 y j==0 no se hace
		D[0] = (Magnitud(vectR, vectG, vectB, 0, 1));
		D[1] = (Magnitud(vectR, vectG, vectB, 0, 2));
		D[2] = (Magnitud(vectR, vectG, vectB, 0, 3));
		D[3] = (Magnitud(vectR, vectG, vectB, 0, 4));
		D[4] = (Magnitud(vectR, vectG, vectB, 0, 5));
		D[5] = (Magnitud(vectR, vectG, vectB, 0, 6));
		D[6] = (Magnitud(vectR, vectG, vectB, 0, 7));
		D[7] = (Magnitud(vectR, vectG, vectB, 0, 8));
		disteucl1[0] = D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7];

		//i=1,j=0 ya esta es D[0]
		//i=1,j=1 No se hace
		D[8] = (Magnitud(vectR, vectG, vectB, 1, 2));
		D[9] = (Magnitud(vectR, vectG, vectB, 1, 3));
		D[10] = (Magnitud(vectR, vectG, vectB, 1, 4));
		D[11] = (Magnitud(vectR, vectG, vectB, 1, 5));
		D[12] = (Magnitud(vectR, vectG, vectB, 1, 6));
		D[13] = (Magnitud(vectR, vectG, vectB, 1, 7));
		D[14] = (Magnitud(vectR, vectG, vectB, 1, 8));
		disteucl1[1] = D[0] + D[8] + D[9] + D[10] + D[11] + D[12] + D[13] + D[14];

		//i=2,j=0 ya esta es D[1]
		//i=2,j=1 ya esta es D[8]
		//i=2,j=2 No se hace
		D[15] = (Magnitud(vectR, vectG, vectB, 2, 3));
		D[16] = (Magnitud(vectR, vectG, vectB, 2, 4));
		D[17] = (Magnitud(vectR, vectG, vectB, 2, 5));
		D[18] = (Magnitud(vectR, vectG, vectB, 2, 6));
		D[19] = (Magnitud(vectR, vectG, vectB, 2, 7));
		D[20] = (Magnitud(vectR, vectG, vectB, 2, 8));
		disteucl1[2] = D[1] + D[8] + D[15] + D[16] + D[17] + D[18] + D[19] + D[20];

		//i=3,j=0 ya esta es D[2]
		//i=3,j=1 ya esta es D[9]
		//i=3,j=2 ya esta es D[15]
		//i=3,j=3 No se hace
		D[21] = (Magnitud(vectR, vectG, vectB, 3, 4));
		D[22] = (Magnitud(vectR, vectG, vectB, 3, 5));
		D[23] = (Magnitud(vectR, vectG, vectB, 3, 6));
		D[24] = (Magnitud(vectR, vectG, vectB, 3, 7));
		D[25] = (Magnitud(vectR, vectG, vectB, 3, 8));
		disteucl1[3] = D[2] + D[9] + D[15] + D[21] + D[22] + D[23] + D[24] + D[25];

		//i=4,j=0 ya esta es D[3]
		//i=4,j=1 ya esta es D[10]
		//i=4,j=2 ya esta es D[16]
		//i=4,j=3 ya esta es D[21]
		//i=4,j=4 No se hace
		D[26] = (Magnitud(vectR, vectG, vectB, 4, 5));
		D[27] = (Magnitud(vectR, vectG, vectB, 4, 6));
		D[28] = (Magnitud(vectR, vectG, vectB, 4, 7));
		D[29] = (Magnitud(vectR, vectG, vectB, 4, 8));
		disteucl1[4] = D[3] + D[10] + D[16] + D[21] + D[26] + D[27] + D[28] + D[29];

		//i=5,j=0 ya esta es D[4]
		//i=5,j=1 ya esta es D[11]
		//i=5,j=2 ya esta es D[17]
		//i=5,j=3 ya esta es D[22]
		//i=5,j=4 ya esta es D[26]
		//i=5,j=5 No se hace
		D[30] = (Magnitud(vectR, vectG, vectB, 5, 6));
		D[31] = (Magnitud(vectR, vectG, vectB, 5, 7));
		D[32] = (Magnitud(vectR, vectG, vectB, 5, 8));
		disteucl1[5] = D[4] + D[11] + D[17] + D[22] + D[26] + D[30] + D[31] + D[32];

		//i=6,j=0 ya esta es D[5]
		//i=6,j=1 ya esta es D[12]
		//i=6,j=2 ya esta es D[18]
		//i=6,j=3 ya esta es D[23]
		//i=6,j=4 ya esta es D[27]
		//i=6,j=5 ya esta es D[30]
		//i=6,j=6 No se hace
		D[33] = (Magnitud(vectR, vectG, vectB, 6, 7));
		D[34] = (Magnitud(vectR, vectG, vectB, 6, 8));
		disteucl1[6] = D[5] + D[12] + D[18] + D[23] + D[27] + D[30] + D[33] + D[34];

		//i=7,j=0 ya esta es D[6]
		//i=7,j=1 ya esta es D[13]
		//i=7,j=2 ya esta es D[19]
		//i=7,j=3 ya esta es D[24]
		//i=7,j=4 ya esta es D[28]
		//i=7,j=5 ya esta es D[31]
		//i=7,j=6 ya esta es D[33]
		//i=7,j=7 No se hace
		D[35] = (Magnitud(vectR, vectG, vectB, 7, 8));
		disteucl1[7] = D[6] + D[13] + D[19] + D[24] + D[28] + D[31] + D[33] + D[35];

		//i=8,j=0 ya esta es D[7]
		//i=8,j=1 ya esta es D[14]
		//i=8,j=2 ya esta es D[20]
		//i=8,j=3 ya esta es D[25]
		//i=8,j=4 ya esta es D[29]
		//i=8,j=5 ya esta es D[32]
		//i=8,j=6 ya esta es D[34]
		//i=8,j=7 ya esta es D[35]
		//i=8,j=8 No se hace
		disteucl1[8] = D[7] + D[14] + D[20] + D[25] + D[29] + D[32] + D[34] + D[35];

		mn = disteucl1[0];
		mx = disteucl1[0];

		posMin = 0;

		for (int i = 0; i<8; i++)
		{
			if (mn>disteucl1[i])
			{
				mn = disteucl1[i];
				posMin = posicion[i];
			}
			else if (mx<disteucl1[i])
			{

			}
		}

		d_Pout[(Row * m + Col) * 3 + 0] = vectR[posMin];
		d_Pout[(Row * m + Col) * 3 + 1] = vectG[posMin];
		d_Pout[(Row * m + Col) * 3 + 2] = vectB[posMin];
	}
}


/*
__global__ void Idea_VMF_FuzzyPeer(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m)
{
int Row = blockIdx.y*blockDim.y + threadIdx.y;
int Col = blockIdx.x*blockDim.x + threadIdx.x;

int x = 0, posicion[9], hold2 = 0, F = 0;
unsigned char vectR[9], vectG[9], vectB[9];
float disteucl = 0.0, disteucl1[9], hold;
float D[40];
float mn, mx;
int posMin = 0;
float arriva = 0.0, abajo = 0.0, val1, val2, val3, dist_M = 0;

if ((Row < m - 1) && (Col < n - 1)){
F = 0;

for (int i = -1; i <= 1; i++){
for (int j = -1; j <= 1; j++){
vectR[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
vectG[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
vectB[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];

posicion[F] = F;
F++;
}
}
//D[0]=Magnitud(vectR, vectG, vectB, i, j//i==0 y j==0 no se hace
D[0] = (Magnitud(vectR, vectG, vectB, 0, 1));
D[1] = (Magnitud(vectR, vectG, vectB, 0, 2));
D[2] = (Magnitud(vectR, vectG, vectB, 0, 3));
D[3] = (Magnitud(vectR, vectG, vectB, 0, 4));
D[4] = (Magnitud(vectR, vectG, vectB, 0, 5));
D[5] = (Magnitud(vectR, vectG, vectB, 0, 6));
D[6] = (Magnitud(vectR, vectG, vectB, 0, 7));
D[7] = (Magnitud(vectR, vectG, vectB, 0, 8));
disteucl1[0] = D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7];

//i=1,j=0 ya esta es D[0]
//i=1,j=1 No se hace
D[8] = (Magnitud(vectR, vectG, vectB, 1, 2));
D[9] = (Magnitud(vectR, vectG, vectB, 1, 3));
D[10] = (Magnitud(vectR, vectG, vectB, 1, 4));
D[11] = (Magnitud(vectR, vectG, vectB, 1, 5));
D[12] = (Magnitud(vectR, vectG, vectB, 1, 6));
D[13] = (Magnitud(vectR, vectG, vectB, 1, 7));
D[14] = (Magnitud(vectR, vectG, vectB, 1, 8));
disteucl1[1] = D[0] + D[8] + D[9] + D[10] + D[11] + D[12] + D[13] + D[14];

//i=2,j=0 ya esta es D[1]
//i=2,j=1 ya esta es D[8]
//i=2,j=2 No se hace
D[15] = (Magnitud(vectR, vectG, vectB, 2, 3));
D[16] = (Magnitud(vectR, vectG, vectB, 2, 4));
D[17] = (Magnitud(vectR, vectG, vectB, 2, 5));
D[18] = (Magnitud(vectR, vectG, vectB, 2, 6));
D[19] = (Magnitud(vectR, vectG, vectB, 2, 7));
D[20] = (Magnitud(vectR, vectG, vectB, 2, 8));
disteucl1[2] = D[1] + D[8] + D[15] + D[16] + D[17] + D[18] + D[19] + D[20];

//i=3,j=0 ya esta es D[2]
//i=3,j=1 ya esta es D[9]
//i=3,j=2 ya esta es D[15]
//i=3,j=3 No se hace
D[21] = (Magnitud(vectR, vectG, vectB, 3, 4));
D[22] = (Magnitud(vectR, vectG, vectB, 3, 5));
D[23] = (Magnitud(vectR, vectG, vectB, 3, 6));
D[24] = (Magnitud(vectR, vectG, vectB, 3, 7));
D[25] = (Magnitud(vectR, vectG, vectB, 3, 8));
disteucl1[3] = D[2] + D[9] + D[15] + D[21] + D[22] + D[23] + D[24] + D[25];

//i=4,j=0 ya esta es D[3]
//i=4,j=1 ya esta es D[10]
//i=4,j=2 ya esta es D[16]
//i=4,j=3 ya esta es D[21]
//i=4,j=4 No se hace
D[26] = (Magnitud(vectR, vectG, vectB, 4, 5));
D[27] = (Magnitud(vectR, vectG, vectB, 4, 6));
D[28] = (Magnitud(vectR, vectG, vectB, 4, 7));
D[29] = (Magnitud(vectR, vectG, vectB, 4, 8));
disteucl1[4] = D[3] + D[10] + D[16] + D[21] + D[26] + D[27] + D[28] + D[29];

//i=5,j=0 ya esta es D[4]
//i=5,j=1 ya esta es D[11]
//i=5,j=2 ya esta es D[17]
//i=5,j=3 ya esta es D[22]
//i=5,j=4 ya esta es D[26]
//i=5,j=5 No se hace
D[30] = (Magnitud(vectR, vectG, vectB, 5, 6));
D[31] = (Magnitud(vectR, vectG, vectB, 5, 7));
D[32] = (Magnitud(vectR, vectG, vectB, 5, 8));
disteucl1[5] = D[4] + D[11] + D[17] + D[22] + D[26] + D[30] + D[31] + D[32];

//i=6,j=0 ya esta es D[5]
//i=6,j=1 ya esta es D[12]
//i=6,j=2 ya esta es D[18]
//i=6,j=3 ya esta es D[23]
//i=6,j=4 ya esta es D[27]
//i=6,j=5 ya esta es D[30]
//i=6,j=6 No se hace
D[33] = (Magnitud(vectR, vectG, vectB, 6, 7));
D[34] = (Magnitud(vectR, vectG, vectB, 6, 8));
disteucl1[6] = D[5] + D[12] + D[18] + D[23] + D[27] + D[30] + D[33] + D[34];

//i=7,j=0 ya esta es D[6]
//i=7,j=1 ya esta es D[13]
//i=7,j=2 ya esta es D[19]
//i=7,j=3 ya esta es D[24]
//i=7,j=4 ya esta es D[28]
//i=7,j=5 ya esta es D[31]
//i=7,j=6 ya esta es D[33]
//i=7,j=7 No se hace
D[35] = (Magnitud(vectR, vectG, vectB, 7, 8));
disteucl1[7] = D[6] + D[13] + D[19] + D[24] + D[28] + D[31] + D[33] + D[35];

//i=8,j=0 ya esta es D[7]
//i=8,j=1 ya esta es D[14]
//i=8,j=2 ya esta es D[20]
//i=8,j=3 ya esta es D[25]
//i=8,j=4 ya esta es D[29]
//i=8,j=5 ya esta es D[32]
//i=8,j=6 ya esta es D[34]
//i=8,j=7 ya esta es D[35]
//i=8,j=8 No se hace
disteucl1[8] = D[7] + D[14] + D[20] + D[25] + D[29] + D[32] + D[34] + D[35];

mn = disteucl1[0];
mx = disteucl1[0];

posMin = 0;

for (int i = 0; i<8; i++)
{
if (mn>disteucl1[i])
{
mn = disteucl1[i];
posMin = posicion[i];
}
else if (mx<disteucl1[i])
{

}
}
vectR[4] = vectR[posMin];
vectG[4] = vectG[posMin];
vectB[4] = vectB[posMin];
for (F = 0; F <= 8; F++){
arriva = minCUDA(vectR[F], vectR[4]) + K;
abajo = maxCUDA(vectR[F], vectR[4]) + K;
val1 = arriva / abajo;

arriva = minCUDA(vectG[F], vectG[4]) + K;
abajo = maxCUDA(vectG[F], vectG[4]) + K;
val2 = arriva / abajo;

arriva = minCUDA(vectB[F], vectB[4]) + K;
abajo = maxCUDA(vectB[F], vectB[4]) + K;
val3 = arriva / abajo;

dist_M = minCUDA(minCUDA(val1, val2), val3);
if (dist_M>d)	P++;
}

if (P <= (q + 1)){
Noise[(Row * m + Col)] = 255;
}
else{
Noise[(Row * m + Col)] = 0;
}




d_Pout[(Row * m + Col) * 3 + 0] = vectR[posMin];
d_Pout[(Row * m + Col) * 3 + 1] = vectG[posMin];
d_Pout[(Row * m + Col) * 3 + 2] = vectB[posMin];
}
}
*/

__device__ float MagnitudL1(float* VectR, float* VectG, float* VectB, unsigned int i, unsigned int j) {

	float distR = abs(VectR[i] - VectR[j]);
	float distG = abs(VectG[i] - VectG[j]);
	float distB = abs(VectB[i] - VectB[j]);

	//return sqrt((distR)*(distR)+(distG)*(distG)+(distB)*(distB));
	return distR + distB + distG;

}

//Gran
__device__ float S_shape(float Nabla, unsigned int a, unsigned int b) {

	if (Nabla <= a)		return 0;

	if (a <= Nabla && Nabla <= ((a + b) / 2)) {
		float aux = (Nabla - a) / (b - a);
		return 2 * aux*aux;
	}

	if (((a + b) / 2) <= Nabla && Nabla <= b) {
		float aux = ((Nabla - b) / (b - a));
		return 1 - (2 * aux*aux);
	}

	if (Nabla >= b)		return 1;

}
//Peque
__device__ float Z_shape(float Nabla, unsigned int a, unsigned int b) {

	if (Nabla <= a)		return 1;
	if (a <= Nabla && Nabla <= ((a + b) / 2)) {
		float aux = (Nabla - a) / (b - a);
		return 1 - (2 * aux*aux);
	}
	if (((a + b) / 2) <= Nabla && Nabla <= b) {
		float aux = (Nabla - b) / (b - a);
		return 2 * aux*aux;
	}
	if (Nabla >= b)		return 0;
}



__global__ void FTSCF_GPU
(unsigned char* d_Pout, const unsigned char* d_Pin, const unsigned int a,
	const unsigned int b, const unsigned int THS, int n, int m) {

	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], posMin = 6, F = 0, i = 0, j = 0;

	float vectR[25], vectG[25], vectB[25];
	float D[45], disteucl1[9], uGran[3], uPeque[3], rs[9], r = 0;

	float mn, mx;

	posicion[0] = 6; posicion[1] = 7; posicion[2] = 8; posicion[3] = 11; posicion[4] = 12;
	posicion[5] = 13; posicion[6] = 16; posicion[7] = 17; posicion[8] = 18;

	if ((Row < m - 3) && (Col < n - 3)) {
		for (i = -2; i <= 2; i++) {
			for (j = -2; j <= 2; j++) {
				vectR[F] = d_Pin[((Row + i) * n + (Col + j)) * 3 + 0];
				vectG[F] = d_Pin[((Row + i) * n + (Col + j)) * 3 + 1];
				vectB[F] = d_Pin[((Row + i) * n + (Col + j)) * 3 + 2];
				F++;
			}
		}


		//NW
		D[0] = (MagnitudL1(vectR, vectG, vectB, 12, 6));
		uGran[0] = S_shape(D[0], a, b);

		D[1] = (MagnitudL1(vectR, vectG, vectB, 12, 8));
		uGran[1] = S_shape(D[1], a, b);

		D[2] = (MagnitudL1(vectR, vectG, vectB, 12, 16));
		uGran[2] = S_shape(D[2], a, b);

		D[3] = (MagnitudL1(vectR, vectG, vectB, 16, 10));
		uPeque[0] = Z_shape(D[3], a, b);

		D[4] = (MagnitudL1(vectR, vectG, vectB, 2, 8));
		uPeque[1] = Z_shape(D[4], a, b);

		rs[0] = uGran[0] * uGran[1] * uGran[2] * uPeque[0] * uPeque[1];

		//N
		D[5] = (MagnitudL1(vectR, vectG, vectB, 12, 7));
		uGran[0] = S_shape(D[5], a, b);

		D[6] = (MagnitudL1(vectR, vectG, vectB, 12, 13));
		uGran[1] = S_shape(D[6], a, b);

		D[7] = (MagnitudL1(vectR, vectG, vectB, 12, 11));
		uGran[2] = S_shape(D[7], a, b);

		D[8] = (MagnitudL1(vectR, vectG, vectB, 11, 6));
		uPeque[0] = Z_shape(D[8], a, b);

		D[9] = (MagnitudL1(vectR, vectG, vectB, 8, 13));
		uPeque[1] = Z_shape(D[9], a, b);

		rs[1] = uGran[0] * uGran[1] * uGran[2] * uPeque[0] * uPeque[1];

		//NE
		//D[10] = (MagnitudL1(vectR, vectG, vectB, 12, 8));
		// es D[1]
		uGran[0] = S_shape(D[1], a, b);

		//D[11] = (MagnitudL1(vectR, vectG, vectB, 12, 6));
		// es D[0]
		uGran[1] = S_shape(D[0], a, b);

		D[10] = (MagnitudL1(vectR, vectG, vectB, 12, 18));
		uGran[2] = S_shape(D[10], a, b);

		D[11] = (MagnitudL1(vectR, vectG, vectB, 18, 14));
		uPeque[0] = Z_shape(D[11], a, b);

		D[12] = (MagnitudL1(vectR, vectG, vectB, 6, 2));
		uPeque[1] = Z_shape(D[12], a, b);

		rs[2] = uGran[0] * uGran[1] * uGran[2] * uPeque[0] * uPeque[1];

		//E			
		//D[15] = (MagnitudL1(vectR, vectG, vectB, 12, 13));
		//es D[6]
		uGran[0] = S_shape(D[6], a, b);

		//D[16] = (MagnitudL1(vectR, vectG, vectB, 12, 7));
		//es D[5]
		uGran[1] = S_shape(D[5], a, b);

		D[13] = (MagnitudL1(vectR, vectG, vectB, 12, 17));
		uGran[2] = S_shape(D[13], a, b);

		D[14] = (MagnitudL1(vectR, vectG, vectB, 7, 8));
		uPeque[0] = Z_shape(D[14], a, b);

		D[15] = (MagnitudL1(vectR, vectG, vectB, 17, 18));
		uPeque[1] = Z_shape(D[15], a, b);

		rs[3] = uGran[0] * uGran[1] * uGran[2] * uPeque[0] * uPeque[1];

		//SE
		//D[20] = (MagnitudL1(vectR, vectG, vectB, 12, 18));
		//es D[10]
		uGran[0] = S_shape(D[10], a, b);

		//es D[2]
		//D[21] = (MagnitudL1(vectR, vectG, vectB, 12, 16));
		uGran[1] = S_shape(D[2], a, b);

		//es D[1]
		//D[22] = (MagnitudL1(vectR, vectG, vectB, 12, 8));
		uGran[2] = S_shape(D[1], a, b);

		D[16] = (MagnitudL1(vectR, vectG, vectB, 16, 22));
		uPeque[0] = Z_shape(D[16], a, b);

		D[17] = (MagnitudL1(vectR, vectG, vectB, 8, 14));
		uPeque[1] = Z_shape(D[17], a, b);

		rs[4] = uGran[0] * uGran[1] * uGran[2] * uPeque[0] * uPeque[1];

		//S
		//D[18] = (MagnitudL1(vectR, vectG, vectB, 12, 17));
		//es D[13]
		uGran[0] = S_shape(D[13], a, b);
		//es D[7]
		//D[26] = (MagnitudL1(vectR, vectG, vectB, 12, 11));
		uGran[1] = S_shape(D[7], a, b);
		//es D[6]
		//D[27] = (MagnitudL1(vectR, vectG, vectB, 12, 13));
		uGran[2] = S_shape(D[6], a, b);

		D[18] = (MagnitudL1(vectR, vectG, vectB, 11, 16));
		uPeque[0] = Z_shape(D[18], a, b);

		D[19] = (MagnitudL1(vectR, vectG, vectB, 13, 18));
		uPeque[1] = Z_shape(D[19], a, b);

		rs[5] = uGran[0] * uGran[1] * uGran[2] * uPeque[0] * uPeque[1];

		//SW
		//es D[2]
		//D[30] = (MagnitudL1(vectR, vectG, vectB, 12, 16));
		uGran[0] = S_shape(D[2], a, b);
		//es D[0]
		//D[31] = (MagnitudL1(vectR, vectG, vectB, 12, 6));
		uGran[1] = S_shape(D[0], a, b);
		//es D[10]
		//D[32] = (MagnitudL1(vectR, vectG, vectB, 12, 18));
		uGran[2] = S_shape(D[10], a, b);

		D[20] = (MagnitudL1(vectR, vectG, vectB, 6, 10));
		uPeque[0] = Z_shape(D[20], a, b);

		D[21] = (MagnitudL1(vectR, vectG, vectB, 18, 22));
		uPeque[1] = Z_shape(D[21], a, b);

		rs[6] = uGran[0] * uGran[1] * uGran[2] * uPeque[0] * uPeque[1];

		//W
		//Es D[7]
		//D[35] = (MagnitudL1(vectR, vectG, vectB, 12, 11));
		uGran[0] = S_shape(D[7], a, b);
		//Es D[5]
		//D[36] = (MagnitudL1(vectR, vectG, vectB, 12, 7));
		uGran[1] = S_shape(D[5], a, b);
		//es D[13]
		//D[37] = (MagnitudL1(vectR, vectG, vectB, 12, 17));
		uGran[2] = S_shape(D[13], a, b);

		D[21] = (MagnitudL1(vectR, vectG, vectB, 6, 7));
		uPeque[0] = Z_shape(D[21], a, b);

		D[22] = (MagnitudL1(vectR, vectG, vectB, 16, 17));
		uPeque[1] = Z_shape(D[22], a, b);

		rs[7] = uGran[0] * uGran[1] * uGran[2] * uPeque[0] * uPeque[1];

		mn = rs[0];
		r = rs[0];

		for (i = 0; i <= 7; i++)
		{
			if (r<rs[i])
			{
				r = rs[i];

			}
		}
		//Filtro VMF
		if (r > THS) {

			D[23] = (MagnitudL1(vectR, vectG, vectB, 6, 8));
			D[24] = (MagnitudL1(vectR, vectG, vectB, 6, 13));
			D[25] = (MagnitudL1(vectR, vectG, vectB, 6, 16));
			D[26] = (MagnitudL1(vectR, vectG, vectB, 6, 17));
			D[27] = (MagnitudL1(vectR, vectG, vectB, 6, 18));

			disteucl1[0] = D[0] + D[8] + D[21] + D[23] + D[24] + D[25] + D[26] + D[27];

			D[28] = (MagnitudL1(vectR, vectG, vectB, 7, 11));
			D[29] = (MagnitudL1(vectR, vectG, vectB, 7, 13));
			D[30] = (MagnitudL1(vectR, vectG, vectB, 7, 16));
			D[31] = (MagnitudL1(vectR, vectG, vectB, 7, 17));
			D[32] = (MagnitudL1(vectR, vectG, vectB, 7, 18));
			disteucl1[1] = D[21] + D[5] + D[14] + D[28] + D[29] + D[30] + D[31] + D[32];

			//es D[26] D[33] = (MagnitudL1(vectR, vectG, vectB, 8, 6));
			D[33] = (MagnitudL1(vectR, vectG, vectB, 8, 11));
			D[34] = (MagnitudL1(vectR, vectG, vectB, 8, 16));
			D[35] = (MagnitudL1(vectR, vectG, vectB, 8, 17));
			D[36] = (MagnitudL1(vectR, vectG, vectB, 8, 18));
			disteucl1[2] = D[14] + D[1] + D[9] + D[26] + D[33] + D[34] + D[35] + D[36];

			//es D[28]  D[37] = (MagnitudL1(vectR, vectG, vectB, 11, 7));
			//es D[33]     D[38] = (MagnitudL1(vectR, vectG, vectB, 11, 8)); 
			D[37] = (MagnitudL1(vectR, vectG, vectB, 11, 13));
			D[38] = (MagnitudL1(vectR, vectG, vectB, 11, 17));
			D[39] = (MagnitudL1(vectR, vectG, vectB, 11, 18));
			disteucl1[3] = D[7] + D[8] + D[18] + D[28] + D[33] + D[37] + D[38] + D[39];

			//Central ya estan todas las d calculadas
			disteucl1[4] = D[0] + D[5] + D[1] + D[7] + D[6] + D[2] + D[13] + D[10];

			D[40] = (MagnitudL1(vectR, vectG, vectB, 13, 16));
			D[41] = (MagnitudL1(vectR, vectG, vectB, 13, 17));
			disteucl1[5] = D[6] + D[19] + D[9] + D[24] + D[29] + D[37] + D[40] + D[41];

			D[42] = (MagnitudL1(vectR, vectG, vectB, 16, 18));
			disteucl1[6] = D[18] + D[2] + D[22] + D[25] + D[30] + D[34] + D[40] + D[42];

			disteucl1[7] = D[22] + D[13] + D[15] + D[26] + D[31] + D[35] + D[38] + D[41];

			disteucl1[8] = D[19] + D[10] + D[15] + D[27] + D[32] + D[36] + D[39] + D[42];

			posMin = 6;
			mn = disteucl1[0];
			for (i = 0; i <= 7; i++) {
				if (mn>disteucl1[i]) {
					mn = disteucl1[i];
					posMin = posicion[i];
				}
			}

			d_Pout[(Row * m + Col) * 3 + 0] = vectR[posMin];
			d_Pout[(Row * m + Col) * 3 + 1] = vectG[posMin];
			d_Pout[(Row * m + Col) * 3 + 2] = vectB[posMin];
			/*
			d_Pout[(Row * m + Col) * 3 + 0] = 255;
			d_Pout[(Row * m + Col) * 3 + 1] = 255;
			d_Pout[(Row * m + Col) * 3 + 2] = 255;
			*/
		}
		else {
			// si no es ruido la salida el el pixel central de la ventana
			d_Pout[(Row * m + Col) * 3 + 0] = vectR[12];
			d_Pout[(Row * m + Col) * 3 + 1] = vectG[12];
			d_Pout[(Row * m + Col) * 3 + 2] = vectB[12];
		}
	}



}


__global__ void VMF_Global_TwoPixels(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m)
{
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	unsigned char vectR[9], vectG[9], vectB[9];
	float disteucl = 0.0, disteucl1[9], hold;
	float D[40];
	float mn, mx;
	int posMin = 0;


	if ((Row < m - 1) && (Col < n - 1)) {
		F = 0;

		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				vectR[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
				vectG[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
				vectB[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];

				posicion[F] = F;
				F++;
			}
		}
		//D[0]=Magnitud(vectR, vectG, vectB, i, j//i==0 y j==0 no se hace
		D[0] = (Magnitud(vectR, vectG, vectB, 0, 1));
		D[1] = (Magnitud(vectR, vectG, vectB, 0, 2));
		D[2] = (Magnitud(vectR, vectG, vectB, 0, 3));
		D[3] = (Magnitud(vectR, vectG, vectB, 0, 4));
		D[4] = (Magnitud(vectR, vectG, vectB, 0, 5));
		D[5] = (Magnitud(vectR, vectG, vectB, 0, 6));
		D[6] = (Magnitud(vectR, vectG, vectB, 0, 7));
		D[7] = (Magnitud(vectR, vectG, vectB, 0, 8));
		disteucl1[0] = D[0] + D[1] + D[2] + D[3] + D[4] + D[5] + D[6] + D[7];

		//i=1,j=0 ya esta es D[0]
		//i=1,j=1 No se hace
		D[8] = (Magnitud(vectR, vectG, vectB, 1, 2));
		D[9] = (Magnitud(vectR, vectG, vectB, 1, 3));
		D[10] = (Magnitud(vectR, vectG, vectB, 1, 4));
		D[11] = (Magnitud(vectR, vectG, vectB, 1, 5));
		D[12] = (Magnitud(vectR, vectG, vectB, 1, 6));
		D[13] = (Magnitud(vectR, vectG, vectB, 1, 7));
		D[14] = (Magnitud(vectR, vectG, vectB, 1, 8));
		disteucl1[1] = D[0] + D[8] + D[9] + D[10] + D[11] + D[12] + D[13] + D[14];

		//i=2,j=0 ya esta es D[1]
		//i=2,j=1 ya esta es D[8]
		//i=2,j=2 No se hace
		D[15] = (Magnitud(vectR, vectG, vectB, 2, 3));
		D[16] = (Magnitud(vectR, vectG, vectB, 2, 4));
		D[17] = (Magnitud(vectR, vectG, vectB, 2, 5));
		D[18] = (Magnitud(vectR, vectG, vectB, 2, 6));
		D[19] = (Magnitud(vectR, vectG, vectB, 2, 7));
		D[20] = (Magnitud(vectR, vectG, vectB, 2, 8));
		disteucl1[2] = D[1] + D[8] + D[15] + D[16] + D[17] + D[18] + D[19] + D[20];

		//i=3,j=0 ya esta es D[2]
		//i=3,j=1 ya esta es D[9]
		//i=3,j=2 ya esta es D[15]
		//i=3,j=3 No se hace
		D[21] = (Magnitud(vectR, vectG, vectB, 3, 4));
		D[22] = (Magnitud(vectR, vectG, vectB, 3, 5));
		D[23] = (Magnitud(vectR, vectG, vectB, 3, 6));
		D[24] = (Magnitud(vectR, vectG, vectB, 3, 7));
		D[25] = (Magnitud(vectR, vectG, vectB, 3, 8));
		disteucl1[3] = D[2] + D[9] + D[15] + D[21] + D[22] + D[23] + D[24] + D[25];

		//i=4,j=0 ya esta es D[3]
		//i=4,j=1 ya esta es D[10]
		//i=4,j=2 ya esta es D[16]
		//i=4,j=3 ya esta es D[21]
		//i=4,j=4 No se hace
		D[26] = (Magnitud(vectR, vectG, vectB, 4, 5));
		D[27] = (Magnitud(vectR, vectG, vectB, 4, 6));
		D[28] = (Magnitud(vectR, vectG, vectB, 4, 7));
		D[29] = (Magnitud(vectR, vectG, vectB, 4, 8));
		disteucl1[4] = D[3] + D[10] + D[16] + D[21] + D[26] + D[27] + D[28] + D[29];

		//i=5,j=0 ya esta es D[4]
		//i=5,j=1 ya esta es D[11]
		//i=5,j=2 ya esta es D[17]
		//i=5,j=3 ya esta es D[22]
		//i=5,j=4 ya esta es D[26]
		//i=5,j=5 No se hace
		D[30] = (Magnitud(vectR, vectG, vectB, 5, 6));
		D[31] = (Magnitud(vectR, vectG, vectB, 5, 7));
		D[32] = (Magnitud(vectR, vectG, vectB, 5, 8));
		disteucl1[5] = D[4] + D[11] + D[17] + D[22] + D[26] + D[30] + D[31] + D[32];

		//i=6,j=0 ya esta es D[5]
		//i=6,j=1 ya esta es D[12]
		//i=6,j=2 ya esta es D[18]
		//i=6,j=3 ya esta es D[23]
		//i=6,j=4 ya esta es D[27]
		//i=6,j=5 ya esta es D[30]
		//i=6,j=6 No se hace
		D[33] = (Magnitud(vectR, vectG, vectB, 6, 7));
		D[34] = (Magnitud(vectR, vectG, vectB, 6, 8));
		disteucl1[6] = D[5] + D[12] + D[18] + D[23] + D[27] + D[30] + D[33] + D[34];

		//i=7,j=0 ya esta es D[6]
		//i=7,j=1 ya esta es D[13]
		//i=7,j=2 ya esta es D[19]
		//i=7,j=3 ya esta es D[24]
		//i=7,j=4 ya esta es D[28]
		//i=7,j=5 ya esta es D[31]
		//i=7,j=6 ya esta es D[33]
		//i=7,j=7 No se hace
		D[35] = (Magnitud(vectR, vectG, vectB, 7, 8));
		disteucl1[7] = D[6] + D[13] + D[19] + D[24] + D[28] + D[31] + D[33] + D[35];

		//i=8,j=0 ya esta es D[7]
		//i=8,j=1 ya esta es D[14]
		//i=8,j=2 ya esta es D[20]
		//i=8,j=3 ya esta es D[25]
		//i=8,j=4 ya esta es D[29]
		//i=8,j=5 ya esta es D[32]
		//i=8,j=6 ya esta es D[34]
		//i=8,j=7 ya esta es D[35]
		//i=8,j=8 No se hace
		disteucl1[8] = D[7] + D[14] + D[20] + D[25] + D[29] + D[32] + D[34] + D[35];

		mn = disteucl1[0];
		mx = disteucl1[0];

		posMin = 0;

		for (int i = 0; i<8; i++)
		{
			if (mn>disteucl1[i])
			{
				mn = disteucl1[i];
				posMin = posicion[i];
			}
			else if (mx<disteucl1[i])
			{

			}
		}

		d_Pout[(Row * m + Col) * 3 + 0] = vectR[posMin];
		d_Pout[(Row * m + Col) * 3 + 1] = vectG[posMin];
		d_Pout[(Row * m + Col) * 3 + 2] = vectB[posMin];
	}
}


__global__ void VectorUnit_GPU_Global(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m)
{
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	float pixel_UR[9], pixel_UG[9], pixel_UB[9];
	unsigned char vectR[9], vectG[9], vectB[9];// esta comentado por el sqrt
	float disteucl = 0.0, disteucl1[9], hold;
	float valMag;
	float mn, mx, AuxResta = 0, aux1 = 0, aux2 = 0, aux3 = 0;
	int posMin = 0;


	if ((Row>1) && (Col>1) && (Row < m - 1) && (Col < n - 1)) {
		

		F = 0;

		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				vectR[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
				vectG[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
				vectB[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];

				pixel_UR[F] = vectR[F];
				pixel_UG[F] = vectG[F];
				pixel_UB[F] = vectB[F];

				if (pixel_UR[F] == 0 && pixel_UG[F] == 0 && pixel_UB[F] == 0) {
					pixel_UR[F] = 10;
					pixel_UG[F] = 10;
					pixel_UB[F] = 10;
 
				}

				else {
					valMag = sqrtf((pixel_UR[F] * pixel_UR[F]) + (pixel_UG[F] * pixel_UG[F]) + (pixel_UB[F] * pixel_UB[F]));
					pixel_UR[F] = pixel_UR[F] / valMag;
					pixel_UG[F] = pixel_UG[F] / valMag;
					pixel_UB[F] = pixel_UB[F] / valMag;
				}
				posicion[F] = F;

				F++;
			}
		}

		disteucl = 0;
		for (F = 0; F <= 8; F++) {
			for (x = 0; x <= 8; x++) {
				//disteucl += abs(vectB[F]-vectB[x])+abs(vectG[F]-vectG[x])+abs(vectR[F]-vectR[x]);

				//disteucl += sqrtf( powf(pixel_UR[F] - pixel_UR[x],2)
				//			   +  powf(pixel_UG[F] - pixel_UG[x],2)
				//			   +  powf(pixel_UB[F] - pixel_UB[x],2) );

				
				//disteucl += ( fabsf(pixel_UR[F] - pixel_UR[x])
				//+ fabsf(pixel_UG[F] - pixel_UG[x])
				//+ fabsf(pixel_UB[F] - pixel_UB[x]));
				
				aux1 = pixel_UR[F] - pixel_UR[x];
				aux2 = pixel_UG[F] - pixel_UG[x];
				aux3 = pixel_UB[F] - pixel_UB[x];
				disteucl += sqrt((aux1*aux1) + (aux2*aux2) + (aux3*aux3));
				
				
				
				//disteucl += sqrt(pow(pixel_UR[F] - pixel_UR[x], 2)
				//+ pow(pixel_UG[F] - pixel_UG[x], 2)
				//+ pow(pixel_UB[F] - pixel_UB[x], 2));
				
			}
			disteucl1[F] = disteucl;
			disteucl = 0;
		}


		mn = disteucl1[0];
		mx = disteucl1[0];

		posMin = 0;

		for (int i = 0; i<8; i++)
		{
			if (mn>disteucl1[i])
			{
				mn = disteucl1[i];
				posMin = posicion[i];
			}
			else if (mx<disteucl1[i])
			{

			}
		}

		d_Pout[(Row * m + Col) * 3 + 0] = vectR[posMin];
		d_Pout[(Row * m + Col) * 3 + 1] = vectG[posMin];
		d_Pout[(Row * m + Col) * 3 + 2] = vectB[posMin];

		
	}

	

}

__global__ void BVDF_GPU_Global(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m) {
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int x = 0, posicion[9], hold2 = 0, F = 0;
	//double vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	//double disteucl, disteucl1[9], hold;
	//double vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	//double disteucl, disteucl1[9], hold;
	//double	arriva = 0, abajo = 0, valAngulo = 0.0, auxCos = 0;

	float vectR[9], vectG[9], vectB[9]; // si el tipo de dato es double, no ay recursos para la ejecusion)
	float disteucl, disteucl1[9], hold;
	float	arriva = 0, abajo = 0, valAngulo = 0.0, auxCos = 0;


	if ((Row>1) && (Col>1) && (Row < m - 1) && (Col < n - 1)) {
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
		valAngulo = 0;
		for (F = 0; F <= 8; F++) {
			for (x = 0; x <= 8; x++) {

				if ((vectR[F] == 0 && vectG[F] == 0 && vectB[F] == 0) || (vectR[x] == 0 && vectG[x] == 0 && vectB[x] == 0)) {
					// Es pixelZero
					valAngulo += 1000;

				}

				else {
					arriva = (vectR[F] * vectR[x]) + (vectG[F] * vectG[x]) + (vectB[F] * vectB[x]);
					abajo = sqrt((vectR[F] * vectR[F]) + (vectG[F] * vectG[F]) + (vectB[F] * vectB[F])) * sqrt((vectR[x] * vectR[x]) + (vectG[x] * vectG[x]) + (vectB[x] * vectB[x]));

					//if (abajo == 0)		abajo = .01;  //si abajo=o da inf
					//if (arriva == 0)	arriva= .01;  //si abajo=o da inf

					valAngulo += acos(arriva / abajo);
					//valAngulo += __cosf(arriva / abajo);
				}
			

			}

			disteucl1[F] = valAngulo;	
			valAngulo = 0;
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
	
		/*
		d_Pout[(Row * n + Col) * 3 + 0] = 255;
		d_Pout[(Row * n + Col) * 3 + 1] = 255;
		d_Pout[(Row * n + Col) * 3 + 2] = 255;
		*/


	}



}



#define min(a, b) ((a < b) ? a : b)
#define max(a, b) ((a > b) ? a : b) //estas dos funciones estan repetidas con minCUDA y maxCUDA

__global__ void FTSCF_GPU_Original
(unsigned char* d_Pout, const unsigned char* d_Pin, int n, int m) {

	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int M = 0, j = 0, x = 0;
	float vectR[9], vectG[9], vectB[9], hold;

	float gam_small_1[18] = { 0 }, med_1, med_2, var_1, gam_big_1[18] = { 0 };
	float gam_small_2[18] = { 0 }, med1, med2, var1, gam_big_2[18] = { 0 };

	float array_R[25];
	float array_G[25];
	float array_B[25];

	int F = 0, i = 0;

	const int channels = 3;

	if ((Row>1) && (Col>1) && (Row < m - 1) && (Col < n - 1)) {

		
		
			//int tid = omp_get_thread_num();
			//hacer el arreglo
			F = 0;
			
			for (i = -2; i <= 2; i++) {
				for (j = -2; j <= 2; j++) {
					array_R[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
					array_G[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
					array_B[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];
					F++;
				}
			}


			// se copia a continuacion solo los 8-vecinos
			M = 0;
			for (F = 6; F <= 8; F++) {
				vectG[M] = (array_G[F]);
				vectR[M] = (array_R[F]);
				vectB[M] = (array_B[F]);
				M++;
			}
			for (F = 11; F <= 13; F++) {
				vectG[M] = (array_G[F]);
				vectR[M] = (array_R[F]);
				vectB[M] = (array_B[F]);
				M++;
			}
			for (F = 16; F <= 18; F++) {
				vectG[M] = (array_G[F]);
				vectR[M] = (array_R[F]);
				vectB[M] = (array_B[F]);
				M++;
			}

			
			float noreste_C_R, noreste_N1_R, noreste_N2_R, sur_C_R, sur_N1_R, sur_N2_R, noroeste_C_R, noroeste_N1_R, noroeste_N2_R;
			float este_C_R, este_N1_R, este_N2_R, oeste_C_R, oeste_N1_R, oeste_N2_R, sureste_C_R, sureste_N1_R, sureste_N2_R;
			float norte_C_R, norte_N1_R, norte_N2_R, suroeste_C_R, suroeste_N1_R, suroeste_N2_R;
			float suroeste_NW_R, suroeste_SE_R, sur_W_R, sur_E_R, sureste_SW_R, sureste_NE_R, este_S_R, este_N_R, noreste_SE_R, noreste_NW_R;
			float norte_W_R, norte_E_R, noroeste_NE_R, noroeste_SW_R, oeste_S_R, oeste_N_R;
			float noreste_C_G, noreste_N1_G, noreste_N2_G, sur_C_G, sur_N1_G, sur_N2_G, noroeste_C_G, noroeste_N1_G, noroeste_N2_G;
			float este_C_G, este_N1_G, este_N2_G, oeste_C_G, oeste_N1_G, oeste_N2_G, sureste_C_G, sureste_N1_G, sureste_N2_G;
			float norte_C_G, norte_N1_G, norte_N2_G, suroeste_C_G, suroeste_N1_G, suroeste_N2_G;
			float suroeste_NW_G, suroeste_SE_G, sur_W_G, sur_E_G, sureste_SW_G, sureste_NE_G, este_S_G, este_N_G, noreste_SE_G, noreste_NW_G;
			float norte_W_G, norte_E_G, noroeste_NE_G, noroeste_SW_G, oeste_S_G, oeste_N_G;
			float noreste_C_B, noreste_N1_B, noreste_N2_B, sur_C_B, sur_N1_B, sur_N2_B, noroeste_C_B, noroeste_N1_B, noroeste_N2_B;
			float este_C_B, este_N1_B, este_N2_B, oeste_C_B, oeste_N1_B, oeste_N2_B, sureste_C_B, sureste_N1_B, sureste_N2_B;
			float norte_C_B, norte_N1_B, norte_N2_B, suroeste_C_B, suroeste_N1_B, suroeste_N2_B;
			float suroeste_NW_B, suroeste_SE_B, sur_W_B, sur_E_B, sureste_SW_B, sureste_NE_B, este_S_B, este_N_B, noreste_SE_B, noreste_NW_B;
			float norte_W_B, norte_E_B, noroeste_NE_B, noroeste_SW_B, oeste_S_B, oeste_N_B;
			float largo[9], largo_1[9], largo_2[9], LARGO[9], LARGO_1[9], LARGO_2[9];
			float noise_R_R, noise_G_G, noise_B_B;
			int SW_C_B, SW_N1_B, SW_N2_B, SW_NW_B, SW_SE_B, S_C_B, S_N1_B, S_N2_B, S_W_B, S_E_B, SE_C_B, SE_N1_B, SE_N2_B, SE_SW_B, SE_NE_B;
			int E_C_B, E_N1_B, E_N2_B, E_S_B, E_N_B, NE_C_B, NE_N1_B, NE_N2_B, NE_SE_B, NE_NW_B, N_C_B, N_N1_B, N_N2_B, N_W_B, N_E_B;
			int NW_C_B, NW_N1_B, NW_N2_B, NW_NE_B, NW_SW_B, W_C_B, W_N1_B, W_N2_B, W_S_B, W_N_B;
			int SW_C_R, SW_N1_R, SW_N2_R, SW_NW_R, SW_SE_R, S_C_R, S_N1_R, S_N2_R, S_W_R, S_E_R, SE_C_R, SE_N1_R, SE_N2_R, SE_SW_R, SE_NE_R;
			int E_C_R, E_N1_R, E_N2_R, E_S_R, E_N_R, NE_C_R, NE_N1_R, NE_N2_R, NE_SE_R, NE_NW_R, N_C_R, N_N1_R, N_N2_R, N_W_R, N_E_R;
			int NW_C_R, NW_N1_R, NW_N2_R, NW_NE_R, NW_SW_R, W_C_R, W_N1_R, W_N2_R, W_S_R, W_N_R;
			int SW_C_G, SW_N1_G, SW_N2_G, SW_NW_G, SW_SE_G, S_C_G, S_N1_G, S_N2_G, S_W_G, S_E_G, SE_C_G, SE_N1_G, SE_N2_G, SE_SW_G, SE_NE_G;
			int E_C_G, E_N1_G, E_N2_G, E_S_G, E_N_G, NE_C_G, NE_N1_G, NE_N2_G, NE_SE_G, NE_NW_G, N_C_G, N_N1_G, N_N2_G, N_W_G, N_E_G;
			int NW_C_G, NW_N1_G, NW_N2_G, NW_NE_G, NW_SW_G, W_C_G, W_N1_G, W_N2_G, W_S_G, W_N_G;
			float cons1 = 255, cons2 = 255;
			
			
			// blue
			SW_C_B = abs(array_B[6] - array_B[12]);
			SW_N1_B = abs(array_B[10] - array_B[16]);
			SW_N2_B = abs(array_B[2] - array_B[8]);
			SW_NW_B = abs(array_B[12] - array_B[16]);
			SW_SE_B = abs(array_B[12] - array_B[8]);
			S_C_B = abs(array_B[7] - array_B[12]);
			S_N1_B = abs(array_B[6] - array_B[11]);
			S_N2_B = abs(array_B[8] - array_B[13]);
			S_W_B = abs(array_B[12] - array_B[11]);
			S_E_B = abs(array_B[12] - array_B[13]);
			SE_C_B = abs(array_B[8] - array_B[12]);
			SE_N1_B = abs(array_B[2] - array_B[6]);
			SE_N2_B = abs(array_B[14] - array_B[18]);
			SE_SW_B = abs(array_B[12] - array_B[6]);
			SE_NE_B = abs(array_B[12] - array_B[18]);
			E_C_B = abs(array_B[13] - array_B[12]);
			E_N1_B = abs(array_B[8] - array_B[7]);
			E_N2_B = abs(array_B[18] - array_B[17]);
			E_S_B = abs(array_B[12] - array_B[7]);
			E_N_B = abs(array_B[12] - array_B[17]);
			NE_C_B = abs(array_B[18] - array_B[12]);
			NE_N1_B = abs(array_B[14] - array_B[8]);
			NE_N2_B = abs(array_B[22] - array_B[16]);
			NE_SE_B = abs(array_B[12] - array_B[8]);
			NE_NW_B = abs(array_B[12] - array_B[16]);
			N_C_B = abs(array_B[17] - array_B[12]);
			N_N1_B = abs(array_B[18] - array_B[13]);
			N_N2_B = abs(array_B[16] - array_B[11]);
			N_W_B = abs(array_B[12] - array_B[11]);
			N_E_B = abs(array_B[12] - array_B[13]);
			NW_C_B = abs(array_B[16] - array_B[12]);
			NW_N1_B = abs(array_B[22] - array_B[18]);
			NW_N2_B = abs(array_B[10] - array_B[6]);
			NW_NE_B = abs(array_B[12] - array_B[18]);
			NW_SW_B = abs(array_B[12] - array_B[6]);
			W_C_B = abs(array_B[11] - array_B[12]);
			W_N1_B = abs(array_B[16] - array_B[17]);
			W_N2_B = abs(array_B[6] - array_B[7]);
			W_S_B = abs(array_B[12] - array_B[7]);
			W_N_B = abs(array_B[12] - array_B[17]);

			SW_C_G = abs(array_G[6] - array_G[12]);
			SW_N1_G = abs(array_G[10] - array_G[16]);
			SW_N2_G = abs(array_G[2] - array_G[8]);
			SW_NW_G = abs(array_G[12] - array_G[16]);
			SW_SE_G = abs(array_G[12] - array_G[8]);
			S_C_G = abs(array_G[7] - array_G[12]);
			S_N1_G = abs(array_G[6] - array_G[11]);
			S_N2_G = abs(array_G[8] - array_G[13]);
			S_W_G = abs(array_G[12] - array_G[11]);
			S_E_G = abs(array_G[12] - array_G[13]);
			SE_C_G = abs(array_G[8] - array_G[12]);
			SE_N1_G = abs(array_G[2] - array_G[6]);
			SE_N2_G = abs(array_G[14] - array_G[18]);
			SE_SW_G = abs(array_G[12] - array_G[6]);
			SE_NE_G = abs(array_G[12] - array_G[18]);
			E_C_G = abs(array_G[13] - array_G[12]);
			E_N1_G = abs(array_G[8] - array_G[7]);
			E_N2_G = abs(array_G[18] - array_G[17]);
			E_S_G = abs(array_G[12] - array_G[7]);
			E_N_G = abs(array_G[12] - array_G[17]);
			NE_C_G = abs(array_G[18] - array_G[12]);
			NE_N1_G = abs(array_G[14] - array_G[8]);
			NE_N2_G = abs(array_G[22] - array_G[16]);
			NE_SE_G = abs(array_G[12] - array_G[8]);
			NE_NW_G = abs(array_G[12] - array_G[16]);
			N_C_G = abs(array_G[17] - array_G[12]);
			N_N1_G = abs(array_G[18] - array_G[13]);
			N_N2_G = abs(array_G[16] - array_G[11]);
			N_W_G = abs(array_G[12] - array_G[11]);
			N_E_G = abs(array_G[12] - array_G[13]);
			NW_C_G = abs(array_G[16] - array_G[12]);
			NW_N1_G = abs(array_G[22] - array_G[18]);
			NW_N2_G = abs(array_G[10] - array_G[6]);
			NW_NE_G = abs(array_G[12] - array_G[18]);
			NW_SW_G = abs(array_G[12] - array_G[6]);
			W_C_G = abs(array_G[11] - array_G[12]);
			W_N1_G = abs(array_G[16] - array_G[17]);
			W_N2_G = abs(array_G[6] - array_G[7]);
			W_S_G = abs(array_G[12] - array_G[7]);
			W_N_G = abs(array_G[12] - array_G[17]);

			SW_C_R = abs(array_R[6] - array_R[12]);
			SW_N1_R = abs(array_R[10] - array_R[16]);
			SW_N2_R = abs(array_R[2] - array_R[8]);
			SW_NW_R = abs(array_R[12] - array_R[16]);
			SW_SE_R = abs(array_R[12] - array_R[8]);
			S_C_R = abs(array_R[7] - array_R[12]);
			S_N1_R = abs(array_R[6] - array_R[11]);
			S_N2_R = abs(array_R[8] - array_R[13]);
			S_W_R = abs(array_R[12] - array_R[11]);
			S_E_R = abs(array_R[12] - array_R[13]);
			SE_C_R = abs(array_R[8] - array_R[12]);
			SE_N1_R = abs(array_R[2] - array_R[6]);
			SE_N2_R = abs(array_R[14] - array_R[18]);
			SE_SW_R = abs(array_R[12] - array_R[6]);
			SE_NE_R = abs(array_R[12] - array_R[18]);
			E_C_R = abs(array_R[13] - array_R[12]);
			E_N1_R = abs(array_R[8] - array_R[7]);
			E_N2_R = abs(array_R[18] - array_R[17]);
			E_S_R = abs(array_R[12] - array_R[7]);
			E_N_R = abs(array_R[12] - array_R[17]);
			NE_C_R = abs(array_R[18] - array_R[12]);
			NE_N1_R = abs(array_R[14] - array_R[8]);
			NE_N2_R = abs(array_R[22] - array_R[16]);
			NE_SE_R = abs(array_R[12] - array_R[8]);
			NE_NW_R = abs(array_R[12] - array_R[16]);
			N_C_R = abs(array_R[17] - array_R[12]);
			N_N1_R = abs(array_R[18] - array_R[13]);
			N_N2_R = abs(array_R[16] - array_R[11]);
			N_W_R = abs(array_R[12] - array_R[11]);
			N_E_R = abs(array_R[12] - array_R[13]);
			NW_C_R = abs(array_R[16] - array_R[12]);
			NW_N1_R = abs(array_R[22] - array_R[18]);
			NW_N2_R = abs(array_R[10] - array_R[6]);
			NW_NE_R = abs(array_R[12] - array_R[18]);
			NW_SW_R = abs(array_R[12] - array_R[6]);
			W_C_R = abs(array_R[11] - array_R[12]);
			W_N1_R = abs(array_R[16] - array_R[17]);
			W_N2_R = abs(array_R[6] - array_R[7]);
			W_S_R = abs(array_R[12] - array_R[7]);
			W_N_R = abs(array_R[12] - array_R[17]);

			if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[12])) == 0) suroeste_C_R = 0;
			else	suroeste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[10] * array_R[16])) == 0) suroeste_N1_R = 0;
			else   suroeste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[10] * array_R[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[2] * array_R[8])) == 0) suroeste_N2_R = 0;
			else   suroeste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[2] * array_R[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[2], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[16])) == 0) suroeste_NW_R = 0;
			else	suroeste_NW_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[8])) == 0) suroeste_SE_R = 0;
			else	suroeste_SE_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[7] * array_R[12])) == 0) sur_C_R = 0;
			else	sur_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[7] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[11])) == 0) sur_N1_R = 0;
			else	sur_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[13])) == 0) sur_N2_R = 0;
			else   sur_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[11])) == 0) sur_W_R = 0;
			else	sur_W_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[13])) == 0) sur_E_R = 0;
			else	sur_E_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[12])) == 0) sureste_C_R = 0;
			else	sureste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[2])) == 0) sureste_N1_R = 0;
			else	sureste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[2])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[2], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[14] * array_R[18])) == 0) sureste_N2_R = 0;
			else	sureste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[14] * array_R[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[6])) == 0) sureste_SW_R = 0;
			else	sureste_SW_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[6])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[18])) == 0) sureste_NE_R = 0;
			else	sureste_NE_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[13] * array_R[12])) == 0) este_C_R = 0;
			else	este_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[13] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[7])) == 0) este_N1_R = 0;
			else	este_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[17])) == 0) este_N2_R = 0;
			else	este_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[7])) == 0) este_S_R = 0;
			else	este_S_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[17])) == 0) este_N_R = 0;
			else	este_N_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[12])) == 0) noreste_C_R = 0;
			else	noreste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[14] * array_R[8])) == 0) noreste_N1_R = 0;
			else	noreste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[14] * array_R[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[22] * array_R[16])) == 0) noreste_N2_R = 0;
			else	noreste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[22] * array_R[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[8])) == 0) noreste_SE_R = 0;
			else	noreste_SE_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[16])) == 0) noreste_NW_R = 0;
			else	noreste_NW_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[17] * array_R[12])) == 0) norte_C_R = 0;
			else	norte_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[17] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[13])) == 0) norte_N1_R = 0;
			else	norte_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[11])) == 0) norte_N2_R = 0;
			else	norte_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[13])) == 0) norte_E_R = 0;
			else	norte_E_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[11])) == 0) norte_W_R = 0;
			else	norte_W_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[12])) == 0) noroeste_C_R = 0;
			else	noroeste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[22] * array_R[18])) == 0) noroeste_N1_R = 0;
			else	noroeste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[22] * array_R[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[10])) == 0) noroeste_N2_R = 0;
			else	noroeste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[10])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[18])) == 0) noroeste_NE_R = 0;
			else	noroeste_NE_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[12])) == 0) noroeste_SW_R = 0;
			else	noroeste_SW_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[11] * array_R[12])) == 0) oeste_C_R = 0;
			else	oeste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[11] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[17])) == 0) oeste_N1_R = 0;
			else	oeste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[7])) == 0) oeste_N2_R = 0;
			else	oeste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[17])) == 0) oeste_N_R = 0;
			else	oeste_N_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[7])) == 0) oeste_S_R = 0;
			else	oeste_S_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))))));

			if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[12])) == 0) suroeste_C_G = 0;
			else	suroeste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[10] * array_G[16])) == 0) suroeste_N1_G = 0;
			else   suroeste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[10] * array_G[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[2] * array_G[8])) == 0) suroeste_N2_G = 0;
			else   suroeste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[2] * array_G[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[2], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[16])) == 0) suroeste_NW_G = 0;
			else	suroeste_NW_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[8])) == 0) suroeste_SE_G = 0;
			else	suroeste_SE_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[7] * array_G[12])) == 0) sur_C_G = 0;
			else	sur_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[7] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[11])) == 0) sur_N1_G = 0;
			else	sur_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[13])) == 0) sur_N2_G = 0;
			else   sur_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[11])) == 0) sur_W_G = 0;
			else	sur_W_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[13])) == 0) sur_E_G = 0;
			else	sur_E_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[12])) == 0) sureste_C_G = 0;
			else	sureste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[2])) == 0) sureste_N1_G = 0;
			else	sureste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[2])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[2], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[14] * array_G[18])) == 0) sureste_N2_G = 0;
			else	sureste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[14] * array_G[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[6])) == 0) sureste_SW_G = 0;
			else	sureste_SW_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[6])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[18])) == 0) sureste_NE_G = 0;
			else	sureste_NE_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[13] * array_G[12])) == 0) este_C_G = 0;
			else	este_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[13] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[7])) == 0) este_N1_G = 0;
			else	este_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[17])) == 0) este_N2_G = 0;
			else	este_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[7])) == 0) este_S_G = 0;
			else	este_S_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[17])) == 0) este_N_G = 0;
			else	este_N_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[12])) == 0) noreste_C_G = 0;
			else	noreste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[14] * array_G[8])) == 0) noreste_N1_G = 0;
			else	noreste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[14] * array_G[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[22] * array_G[16])) == 0) noreste_N2_G = 0;
			else	noreste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[22] * array_G[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[8])) == 0) noreste_SE_G = 0;
			else	noreste_SE_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[16])) == 0) noreste_NW_G = 0;
			else	noreste_NW_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[17] * array_G[12])) == 0) norte_C_G = 0;
			else	norte_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[17] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[13])) == 0) norte_N1_G = 0;
			else	norte_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[11])) == 0) norte_N2_G = 0;
			else	norte_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[13])) == 0) norte_E_G = 0;
			else	norte_E_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[11])) == 0) norte_W_G = 0;
			else	norte_W_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[12])) == 0) noroeste_C_G = 0;
			else	noroeste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[22] * array_G[18])) == 0) noroeste_N1_G = 0;
			else	noroeste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[22] * array_G[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[10])) == 0) noroeste_N2_G = 0;
			else	noroeste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[10])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[18])) == 0) noroeste_NE_G = 0;
			else	noroeste_NE_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[12])) == 0) noroeste_SW_G = 0;
			else	noroeste_SW_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[11] * array_G[12])) == 0) oeste_C_G = 0;
			else	oeste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[11] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[17])) == 0) oeste_N1_G = 0;
			else	oeste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[7])) == 0) oeste_N2_G = 0;
			else	oeste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[17])) == 0) oeste_N_G = 0;
			else	oeste_N_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[7])) == 0) oeste_S_G = 0;
			else	oeste_S_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))))));

			if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[12])) == 0) suroeste_C_B = 0;
			else	suroeste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[10] * array_B[16])) == 0) suroeste_N1_B = 0;
			else   suroeste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[10] * array_B[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[2] * array_B[8])) == 0) suroeste_N2_B = 0;
			else   suroeste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[2] * array_B[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[2], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[16])) == 0) suroeste_NW_B = 0;
			else	suroeste_NW_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[8])) == 0) suroeste_SE_B = 0;
			else	suroeste_SE_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[7] * array_B[12])) == 0) sur_C_B = 0;
			else	sur_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[7] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2)))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[11])) == 0) sur_N1_B = 0;
			else	sur_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[13])) == 0) sur_N2_B = 0;
			else   sur_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[11])) == 0) sur_W_B = 0;
			else	sur_W_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[13])) == 0) sur_E_B = 0;
			else	sur_E_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[12])) == 0) sureste_C_B = 0;
			else	sureste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[2])) == 0) sureste_N1_B = 0;
			else	sureste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[2])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[2], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[14] * array_B[18])) == 0) sureste_N2_B = 0;
			else	sureste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[14] * array_B[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[6])) == 0) sureste_SW_B = 0;
			else	sureste_SW_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[6])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[18])) == 0) sureste_NE_B = 0;
			else	sureste_NE_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[13] * array_B[12])) == 0) este_C_B = 0;
			else	este_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[13] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[7])) == 0) este_N1_B = 0;
			else	este_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[17])) == 0) este_N2_B = 0;
			else	este_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[7])) == 0) este_S_B = 0;
			else	este_S_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[17])) == 0) este_N_B = 0;
			else	este_N_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[12])) == 0) noreste_C_B = 0;
			else	noreste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[14] * array_B[8])) == 0) noreste_N1_B = 0;
			else	noreste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[14] * array_B[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[22] * array_B[16])) == 0) noreste_N2_B = 0;
			else	noreste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[22] * array_B[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[8])) == 0) noreste_SE_B = 0;
			else	noreste_SE_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[16])) == 0) noreste_NW_B = 0;
			else	noreste_NW_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[17] * array_B[12])) == 0) norte_C_B = 0;
			else	norte_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[17] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[13])) == 0) norte_N1_B = 0;
			else	norte_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[11])) == 0) norte_N2_B = 0;
			else	norte_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[13])) == 0) norte_E_B = 0;
			else	norte_E_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[11])) == 0) norte_W_B = 0;
			else	norte_W_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[12])) == 0) noroeste_C_B = 0;
			else	noroeste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[22] * array_B[18])) == 0) noroeste_N1_B = 0;
			else	noroeste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[22] * array_B[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[10])) == 0) noroeste_N2_B = 0;
			else	noroeste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[10])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[18])) == 0) noroeste_NE_B = 0;
			else	noroeste_NE_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[12])) == 0) noroeste_SW_B = 0;
			else	noroeste_SW_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[11] * array_B[12])) == 0) oeste_C_B = 0;
			else	oeste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[11] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[17])) == 0) oeste_N1_B = 0;
			else	oeste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[7])) == 0) oeste_N2_B = 0;
			else	oeste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[17])) == 0) oeste_N_B = 0;
			else	oeste_N_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))))));
			if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[7])) == 0) oeste_S_B = 0;
			else	oeste_S_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))))));
			//	SUROESTE	

			med_1 = 1, var_1 = 0.8;
			med_2 = 0.1;

			if (suroeste_C_R > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((suroeste_C_R)-med_1), 2) / (2 * var_1))));
			if (suroeste_N1_R < med_2) gam_small_1[0] = 1;
			else 	gam_small_1[0] = (exp(-(pow(((suroeste_N1_R)-med_2), 2) / (2 * var_1))));
			if (suroeste_N2_R < med_2) gam_small_1[1] = 1;
			else 	gam_small_1[1] = (exp(-(pow(((suroeste_N2_R)-med_2), 2) / (2 * var_1))));
			if (suroeste_NW_R > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((suroeste_NW_R)-med_1), 2) / (2 * var_1))));
			if (suroeste_SE_R > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((suroeste_SE_R)-med_1), 2) / (2 * var_1))));
			largo[0] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (sur_C_R > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((sur_C_R)-med_1), 2) / (2 * var_1))));
			if (sur_N1_R < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((sur_N1_R)-med_2), 2) / (2 * var_1))));
			if (sur_N2_R < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((sur_N2_R)-med_2), 2) / (2 * var_1))));
			if (sur_W_R > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((sur_W_R)-med_1), 2) / (2 * var_1))));
			if (sur_E_R > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((sur_E_R)-med_1), 2) / (2 * var_1))));
			largo[1] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (sureste_C_R > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((sureste_C_R)-med_1), 2) / (2 * var_1))));
			if (sureste_N1_R < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((sureste_N1_R)-med_2), 2) / (2 * var_1))));
			if (sureste_N2_R < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((sureste_N2_R)-med_2), 2) / (2 * var_1))));
			if (sureste_NE_R > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((sureste_NE_R)-med_1), 2) / (2 * var_1))));
			if (sureste_SW_R > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((sureste_SW_R)-med_1), 2) / (2 * var_1))));
			largo[2] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (este_C_R > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((este_C_R)-med_1), 2) / (2 * var_1))));
			if (este_N1_R < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((este_N1_R)-med_2), 2) / (2 * var_1))));
			if (este_N2_R < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((este_N2_R)-med_2), 2) / (2 * var_1))));
			if (este_N_R > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((este_N_R)-med_1), 2) / (2 * var_1))));
			if (este_S_R > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((este_S_R)-med_1), 2) / (2 * var_1))));
			largo[3] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (noreste_C_R > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((noreste_C_R)-med_1), 2) / (2 * var_1))));
			if (noreste_N1_R < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((noreste_N1_R)-med_2), 2) / (2 * var_1))));
			if (noreste_N2_R < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((noreste_N2_R)-med_2), 2) / (2 * var_1))));
			if (noreste_NW_R > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((noreste_NW_R)-med_1), 2) / (2 * var_1))));
			if (noreste_SE_R > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((noreste_SE_R)-med_1), 2) / (2 * var_1))));
			largo[4] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (norte_C_R > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((norte_C_R)-med_1), 2) / (2 * var_1))));
			if (norte_N1_R < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((norte_N1_R)-med_2), 2) / (2 * var_1))));
			if (norte_N2_R < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((norte_N2_R)-med_2), 2) / (2 * var_1))));
			if (norte_W_R > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((norte_W_R)-med_1), 2) / (2 * var_1))));
			if (norte_E_R > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((norte_E_R)-med_1), 2) / (2 * var_1))));
			largo[5] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (noroeste_C_R > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((noroeste_C_R)-med_1), 2) / (2 * var_1))));
			if (noroeste_N1_R < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((noroeste_N1_R)-med_2), 2) / (2 * var_1))));
			if (noroeste_N2_R < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((noroeste_N2_R)-med_2), 2) / (2 * var_1))));
			if (noroeste_NE_R > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((noroeste_NE_R)-med_1), 2) / (2 * var_1))));
			if (noroeste_SW_R > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((noroeste_SW_R)-med_1), 2) / (2 * var_1))));
			largo[6] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (oeste_C_R > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((oeste_C_R)-med_1), 2) / (2 * var_1))));
			if (oeste_N1_R < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((oeste_N1_R)-med_2), 2) / (2 * var_1))));
			if (oeste_N2_R < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((oeste_N2_R)-med_2), 2) / (2 * var_1))));
			if (oeste_N_R > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((oeste_N_R)-med_1), 2) / (2 * var_1))));
			if (oeste_S_R > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((oeste_S_R)-med_1), 2) / (2 * var_1))));
			largo[7] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (suroeste_C_G > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((suroeste_C_G)-med_1), 2) / (2 * var_1))));
			if (suroeste_N1_G < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((suroeste_N1_G)-med_2), 2) / (2 * var_1))));
			if (suroeste_N2_G < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((suroeste_N2_G)-med_2), 2) / (2 * var_1))));
			if (suroeste_NW_G > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((suroeste_NW_G)-med_1), 2) / (2 * var_1))));
			if (suroeste_SE_G > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((suroeste_SE_G)-med_1), 2) / (2 * var_1))));
			largo_1[0] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (sur_C_G > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((sur_C_G)-med_1), 2) / (2 * var_1))));
			if (sur_N1_G < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((sur_N1_G)-med_2), 2) / (2 * var_1))));
			if (sur_N2_G < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((sur_N2_G)-med_2), 2) / (2 * var_1))));
			if (sur_W_G > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((sur_W_G)-med_1), 2) / (2 * var_1))));
			if (sur_E_G > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((sur_E_G)-med_1), 2) / (2 * var_1))));
			largo_1[1] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (sureste_C_G > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((sureste_C_G)-med_1), 2) / (2 * var_1))));
			if (sureste_N1_G < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((sureste_N1_G)-med_2), 2) / (2 * var_1))));
			if (sureste_N2_G < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((sureste_N2_G)-med_2), 2) / (2 * var_1))));
			if (sureste_NE_G > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((sureste_NE_G)-med_1), 2) / (2 * var_1))));
			if (sureste_SW_G > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((sureste_SW_G)-med_1), 2) / (2 * var_1))));
			largo_1[2] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (este_C_G > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((este_C_G)-med_1), 2) / (2 * var_1))));
			if (este_N1_G < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((este_N1_G)-med_2), 2) / (2 * var_1))));
			if (este_N2_G < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((este_N2_G)-med_2), 2) / (2 * var_1))));
			if (este_N_G > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((este_N_G)-med_1), 2) / (2 * var_1))));
			if (este_S_G > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((este_S_G)-med_1), 2) / (2 * var_1))));
			largo_1[3] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (noreste_C_G > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((noreste_C_G)-med_1), 2) / (2 * var_1))));
			if (noreste_N1_G < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((noreste_N1_G)-med_2), 2) / (2 * var_1))));
			if (noreste_N2_G < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((noreste_N2_G)-med_2), 2) / (2 * var_1))));
			if (noreste_NW_G > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((noreste_NW_G)-med_1), 2) / (2 * var_1))));
			if (noreste_SE_G > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((noreste_SE_G)-med_1), 2) / (2 * var_1))));
			largo_1[4] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (norte_C_G > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((norte_C_G)-med_1), 2) / (2 * var_1))));
			if (norte_N1_G < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((norte_N1_G)-med_2), 2) / (2 * var_1))));
			if (norte_N2_G < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((norte_N2_G)-med_2), 2) / (2 * var_1))));
			if (norte_W_G > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((norte_W_G)-med_1), 2) / (2 * var_1))));
			if (norte_E_G > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((norte_E_G)-med_1), 2) / (2 * var_1))));
			largo_1[5] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (noroeste_C_G > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((noroeste_C_G)-med_1), 2) / (2 * var_1))));
			if (noroeste_N1_G < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((noroeste_N1_G)-med_2), 2) / (2 * var_1))));
			if (noroeste_N2_G < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((noroeste_N2_G)-med_2), 2) / (2 * var_1))));
			if (noroeste_NE_G > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((noroeste_NE_G)-med_1), 2) / (2 * var_1))));
			if (noroeste_SW_G > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((noroeste_SW_G)-med_1), 2) / (2 * var_1))));
			largo_1[6] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (oeste_C_G > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((oeste_C_G)-med_1), 2) / (2 * var_1))));
			if (oeste_N1_G < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((oeste_N1_G)-med_2), 2) / (2 * var_1))));
			if (oeste_N2_G < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((oeste_N2_G)-med_2), 2) / (2 * var_1))));
			if (oeste_N_G > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((oeste_N_G)-med_1), 2) / (2 * var_1))));
			if (oeste_S_G > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((oeste_S_G)-med_1), 2) / (2 * var_1))));
			largo_1[7] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (suroeste_C_B > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((suroeste_C_B)-med_1), 2) / (2 * var_1))));
			if (suroeste_N1_B < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((suroeste_N1_B)-med_2), 2) / (2 * var_1))));
			if (suroeste_N2_B < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((suroeste_N2_B)-med_2), 2) / (2 * var_1))));
			if (suroeste_NW_B > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((suroeste_NW_B)-med_1), 2) / (2 * var_1))));
			if (suroeste_SE_B > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((suroeste_SE_B)-med_1), 2) / (2 * var_1))));
			largo_2[0] = (gam_big_2[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_2[2]);
			if (sur_C_B > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((sur_C_B)-med_1), 2) / (2 * var_1))));
			if (sur_N1_B < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((sur_N1_B)-med_2), 2) / (2 * var_1))));
			if (sur_N2_B < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((sur_N2_B)-med_2), 2) / (2 * var_1))));
			if (sur_W_B > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((sur_W_B)-med_1), 2) / (2 * var_1))));
			if (sur_E_B > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((sur_E_B)-med_1), 2) / (2 * var_1))));
			largo_2[1] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (sureste_C_B > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((sureste_C_B)-med_1), 2) / (2 * var_1))));
			if (sureste_N1_B < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((sureste_N1_B)-med_2), 2) / (2 * var_1))));
			if (sureste_N2_B < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((sureste_N2_B)-med_2), 2) / (2 * var_1))));
			if (sureste_NE_B > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((sureste_NE_B)-med_1), 2) / (2 * var_1))));
			if (sureste_SW_B > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((sureste_SW_B)-med_1), 2) / (2 * var_1))));
			largo_2[2] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (este_C_B > med_1) gam_big_1[0] = 1;

			else	gam_big_1[0] = (exp(-(pow(((este_C_B)-med_1), 2) / (2 * var_1))));
			if (este_N1_B < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((este_N1_B)-med_2), 2) / (2 * var_1))));
			if (este_N2_B < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((este_N2_B)-med_2), 2) / (2 * var_1))));
			if (este_N_B > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((este_N_B)-med_1), 2) / (2 * var_1))));
			if (este_S_B > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((este_S_B)-med_1), 2) / (2 * var_1))));
			largo_2[3] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (noreste_C_B > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((noreste_C_B)-med_1), 2) / (2 * var_1))));
			if (noreste_N1_B < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((noreste_N1_B)-med_2), 2) / (2 * var_1))));
			if (noreste_N2_B < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((noreste_N2_B)-med_2), 2) / (2 * var_1))));
			if (noreste_NW_B > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((noreste_NW_B)-med_1), 2) / (2 * var_1))));
			if (noreste_SE_B > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((noreste_SE_B)-med_1), 2) / (2 * var_1))));
			largo_2[4] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (norte_C_B > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((norte_C_B)-med_1), 2) / (2 * var_1))));
			if (norte_N1_B < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((norte_N1_B)-med_2), 2) / (2 * var_1))));
			if (norte_N2_B < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((norte_N2_B)-med_2), 2) / (2 * var_1))));
			if (norte_W_B > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((norte_W_B)-med_1), 2) / (2 * var_1))));
			if (norte_E_B > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((norte_E_B)-med_1), 2) / (2 * var_1))));
			largo_2[5] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (noroeste_C_B > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((noroeste_C_B)-med_1), 2) / (2 * var_1))));
			if (noroeste_N1_B < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((noroeste_N1_B)-med_2), 2) / (2 * var_1))));
			if (noroeste_N2_B < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((noroeste_N2_B)-med_2), 2) / (2 * var_1))));
			if (noroeste_NE_B > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((noroeste_NE_B)-med_1), 2) / (2 * var_1))));
			if (noroeste_SW_B > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((noroeste_SW_B)-med_1), 2) / (2 * var_1))));
			largo_2[6] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
			if (oeste_C_B > med_1) gam_big_1[0] = 1;
			else	gam_big_1[0] = (exp(-(pow(((oeste_C_B)-med_1), 2) / (2 * var_1))));
			if (oeste_N1_B < med_2) gam_small_1[0] = 1;
			else	gam_small_1[0] = (exp(-(pow(((oeste_N1_B)-med_2), 2) / (2 * var_1))));
			if (oeste_N2_B < med_2) gam_small_1[1] = 1;
			else	gam_small_1[1] = (exp(-(pow(((oeste_N2_B)-med_2), 2) / (2 * var_1))));
			if (oeste_N_B > med_1) gam_big_1[1] = 1;
			else	gam_big_1[1] = (exp(-(pow(((oeste_N_B)-med_1), 2) / (2 * var_1))));
			if (oeste_S_B > med_1) gam_big_1[2] = 1;
			else	gam_big_1[2] = (exp(-(pow(((oeste_S_B)-med_1), 2) / (2 * var_1))));
			largo_2[7] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);

			med1 = 60;
			med2 = 10;
			var1 = 1000;
			if (SW_C_R > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((SW_C_R)-med1), 2) / (2 * var1))));
			if (SW_N1_R < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((SW_N1_R)-med2), 2) / (2 * var1))));
			if (SW_N2_R < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((SW_N2_R)-med2), 2) / (2 * var1))));
			if (SW_NW_R > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((SW_NW_R)-med1), 2) / (2 * var1))));
			if (SW_SE_R > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((SW_SE_R)-med1), 2) / (2 * var1))));
			LARGO[0] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (S_C_R > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((S_C_R)-med1), 2) / (2 * var1))));
			if (S_N1_R < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((S_N1_R)-med2), 2) / (2 * var1))));
			if (S_N2_R < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((S_N2_R)-med2), 2) / (2 * var1))));
			if (S_W_R > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((S_W_R)-med1), 2) / (2 * var1))));
			if (S_E_R > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((S_E_R)-med1), 2) / (2 * var1))));
			LARGO[1] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (SE_C_R > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((SE_C_R)-med1), 2) / (2 * var1))));
			if (SE_N1_R < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((SE_N1_R)-med2), 2) / (2 * var1))));
			if (SE_N2_R < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((SE_N2_R)-med2), 2) / (2 * var1))));
			if (SE_NE_R > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((SE_NE_R)-med1), 2) / (2 * var1))));
			if (SE_SW_R > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((SE_SW_R)-med1), 2) / (2 * var1))));
			LARGO[2] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (E_C_R > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((E_C_R)-med1), 2) / (2 * var1))));
			if (E_N1_R < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((E_N1_R)-med2), 2) / (2 * var1))));
			if (E_N2_R < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((E_N2_R)-med2), 2) / (2 * var1))));
			if (E_N_R > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((E_N_R)-med1), 2) / (2 * var1))));
			if (E_S_R > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((E_S_R)-med1), 2) / (2 * var1))));
			LARGO[3] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (NE_C_R > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((NE_C_R)-med1), 2) / (2 * var1))));
			if (NE_N1_R < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((NE_N1_R)-med2), 2) / (2 * var1))));
			if (NE_N2_R < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((NE_N2_R)-med2), 2) / (2 * var1))));
			if (NE_NW_R > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((NE_NW_R)-med1), 2) / (2 * var1))));
			if (NE_SE_R > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((NE_SE_R)-med1), 2) / (2 * var1))));
			LARGO[4] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (N_C_R > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((N_C_R)-med1), 2) / (2 * var1))));
			if (N_N1_R < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((N_N1_R)-med2), 2) / (2 * var1))));
			if (N_N2_R < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((N_N2_R)-med2), 2) / (2 * var1))));
			if (N_W_R > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((N_W_R)-med1), 2) / (2 * var1))));
			if (N_E_R > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((N_E_R)-med1), 2) / (2 * var1))));
			LARGO[5] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (NW_C_R > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((NW_C_R)-med1), 2) / (2 * var1))));
			if (NW_N1_R < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((NW_N1_R)-med2), 2) / (2 * var1))));
			if (NW_N2_R < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((NW_N2_R)-med2), 2) / (2 * var1))));
			if (NW_NE_R > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((NW_NE_R)-med1), 2) / (2 * var1))));
			if (NW_SW_R > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((NW_SW_R)-med1), 2) / (2 * var1))));
			LARGO[6] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (W_C_R > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((W_C_R)-med1), 2) / (2 * var1))));
			if (W_N1_R < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((W_N1_R)-med2), 2) / (2 * var1))));
			if (W_N2_R < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((W_N2_R)-med2), 2) / (2 * var1))));
			if (W_N_R > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((W_N_R)-med1), 2) / (2 * var1))));
			if (W_S_R > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((W_S_R)-med1), 2) / (2 * var1))));
			LARGO[7] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (SW_C_G > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((SW_C_G)-med1), 2) / (2 * var1))));
			if (SW_N1_G < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((SW_N1_G)-med2), 2) / (2 * var1))));
			if (SW_N2_G < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((SW_N2_G)-med2), 2) / (2 * var1))));
			if (SW_NW_G > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((SW_NW_G)-med1), 2) / (2 * var1))));
			if (SW_SE_G > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((SW_SE_G)-med1), 2) / (2 * var1))));
			LARGO_1[0] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (S_C_G > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((S_C_G)-med1), 2) / (2 * var1))));
			if (S_N1_G < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((S_N1_G)-med2), 2) / (2 * var1))));
			if (S_N2_G < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((S_N2_G)-med2), 2) / (2 * var1))));
			if (S_W_G > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((S_W_G)-med1), 2) / (2 * var1))));
			if (S_E_G > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((S_E_G)-med1), 2) / (2 * var1))));
			LARGO_1[1] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (SE_C_G > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((SE_C_G)-med1), 2) / (2 * var1))));
			if (SE_N1_G < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((SE_N1_G)-med2), 2) / (2 * var1))));
			if (SE_N2_G < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((SE_N2_G)-med2), 2) / (2 * var1))));
			if (SE_NE_G > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((SE_NE_G)-med1), 2) / (2 * var1))));
			if (SE_SW_G > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((SE_SW_G)-med1), 2) / (2 * var1))));
			LARGO_1[2] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (E_C_G > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((E_C_G)-med1), 2) / (2 * var1))));
			if (E_N1_G < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((E_N1_G)-med2), 2) / (2 * var1))));
			if (E_N2_G < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((E_N2_G)-med2), 2) / (2 * var1))));
			if (E_N_G > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((E_N_G)-med1), 2) / (2 * var1))));
			if (E_S_G > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((E_S_G)-med1), 2) / (2 * var1))));
			LARGO_1[3] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (NE_C_G > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((NE_C_G)-med1), 2) / (2 * var1))));
			if (NE_N1_G < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((NE_N1_G)-med2), 2) / (2 * var1))));
			if (NE_N2_G < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((NE_N2_G)-med2), 2) / (2 * var1))));
			if (NE_NW_G > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((NE_NW_G)-med1), 2) / (2 * var1))));
			if (NE_SE_G > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((NE_SE_G)-med1), 2) / (2 * var1))));
			LARGO_1[4] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (N_C_G > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((N_C_G)-med1), 2) / (2 * var1))));
			if (N_N1_G < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((N_N1_G)-med2), 2) / (2 * var1))));
			if (N_N2_G < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((N_N2_G)-med2), 2) / (2 * var1))));
			if (N_W_G > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((N_W_G)-med1), 2) / (2 * var1))));
			if (N_E_G > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((N_E_G)-med1), 2) / (2 * var1))));
			LARGO_1[5] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (NW_C_G > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((NW_C_G)-med1), 2) / (2 * var1))));
			if (NW_N1_G < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((NW_N1_G)-med2), 2) / (2 * var1))));
			if (NW_N2_G < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((NW_N2_G)-med2), 2) / (2 * var1))));
			if (NW_NE_G > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((NW_NE_G)-med1), 2) / (2 * var1))));
			if (NW_SW_G > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((NW_SW_G)-med1), 2) / (2 * var1))));
			LARGO_1[6] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (W_C_G > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((W_C_G)-med1), 2) / (2 * var1))));
			if (W_N1_G < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((W_N1_G)-med2), 2) / (2 * var1))));
			if (W_N2_G < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((W_N2_G)-med2), 2) / (2 * var1))));
			if (W_N_G > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((W_N_G)-med1), 2) / (2 * var1))));
			if (W_S_G > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((W_S_G)-med1), 2) / (2 * var1))));
			LARGO_1[7] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (SW_C_G > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((SW_C_B)-med1), 2) / (2 * var1))));
			if (SW_N1_B < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((SW_N1_B)-med2), 2) / (2 * var1))));
			if (SW_N2_B < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((SW_N2_B)-med2), 2) / (2 * var1))));
			if (SW_NW_B > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((SW_NW_B)-med1), 2) / (2 * var1))));
			if (SW_SE_B > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((SW_SE_B)-med1), 2) / (2 * var1))));
			LARGO_2[0] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (S_C_B > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((S_C_B)-med1), 2) / (2 * var1))));
			if (S_N1_B < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((S_N1_B)-med2), 2) / (2 * var1))));
			if (S_N2_B < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((S_N2_B)-med2), 2) / (2 * var1))));
			if (S_W_B > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((S_W_B)-med1), 2) / (2 * var1))));
			if (S_E_B > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((S_E_B)-med1), 2) / (2 * var1))));
			LARGO_2[1] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (SE_C_B > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((SE_C_B)-med1), 2) / (2 * var1))));
			if (SE_N1_B < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((SE_N1_B)-med2), 2) / (2 * var1))));
			if (SE_N2_B < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((SE_N2_B)-med2), 2) / (2 * var1))));
			if (SE_NE_B > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((SE_NE_B)-med1), 2) / (2 * var1))));
			if (SE_SW_B > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((SE_SW_B)-med1), 2) / (2 * var1))));
			LARGO_2[2] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (E_C_B > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((E_C_B)-med1), 2) / (2 * var1))));
			if (E_N1_B < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((E_N1_B)-med2), 2) / (2 * var1))));
			if (E_N2_B < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((E_N2_B)-med2), 2) / (2 * var1))));
			if (E_N_B > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((E_N_B)-med1), 2) / (2 * var1))));
			if (E_S_B > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((E_S_B)-med1), 2) / (2 * var1))));
			LARGO_2[3] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (NE_C_B > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((NE_C_B)-med1), 2) / (2 * var1))));
			if (NE_N1_B < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((NE_N1_B)-med2), 2) / (2 * var1))));
			if (NE_N2_B < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((NE_N2_B)-med2), 2) / (2 * var1))));
			if (NE_NW_B > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((NE_NW_B)-med1), 2) / (2 * var1))));
			if (NE_SE_B > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((NE_SE_B)-med1), 2) / (2 * var1))));
			LARGO_2[4] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (N_C_B > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((N_C_B)-med1), 2) / (2 * var1))));
			if (N_N1_B < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((N_N1_B)-med2), 2) / (2 * var1))));
			if (N_N2_B < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((N_N2_B)-med2), 2) / (2 * var1))));
			if (N_W_B > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((N_W_B)-med1), 2) / (2 * var1))));
			if (N_E_B > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((N_E_B)-med1), 2) / (2 * var1))));
			LARGO_2[5] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (NW_C_B > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((NW_C_B)-med1), 2) / (2 * var1))));
			if (NW_N1_B < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((NW_N1_B)-med2), 2) / (2 * var1))));
			if (NW_N2_B < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((NW_N2_B)-med2), 2) / (2 * var1))));
			if (NW_NE_B > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((NW_NE_B)-med1), 2) / (2 * var1))));
			if (NW_SW_B > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((NW_SW_B)-med1), 2) / (2 * var1))));
			LARGO_2[6] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
			if (W_C_B > med1) gam_big_2[0] = 1;
			else	gam_big_2[0] = (exp(-(pow(((W_C_B)-med1), 2) / (2 * var1))));
			if (W_N1_B < med2) gam_small_2[0] = 1;
			else	gam_small_2[0] = (exp(-(pow(((W_N1_B)-med2), 2) / (2 * var1))));
			if (W_N2_B < med2) gam_small_2[1] = 1;
			else	gam_small_2[1] = (exp(-(pow(((W_N2_B)-med2), 2) / (2 * var1))));
			if (W_N_B > med1) gam_big_2[1] = 1;
			else	gam_big_2[1] = (exp(-(pow(((W_N_B)-med1), 2) / (2 * var1))));
			if (W_S_B > med1) gam_big_2[2] = 1;
			else	gam_big_2[2] = (exp(-(pow(((W_S_B)-med1), 2) / (2 * var1))));
			LARGO_2[7] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);

			float	mu_R_R[8], mu_G_G[8], mu_B_B[8];

			mu_R_R[0] = min(largo[0], LARGO[0]);
			mu_R_R[1] = min(largo[1], LARGO[1]);
			mu_R_R[2] = min(largo[2], LARGO[2]);
			mu_R_R[3] = min(largo[3], LARGO[3]);
			mu_R_R[4] = min(largo[4], LARGO[4]);
			mu_R_R[5] = min(largo[5], LARGO[5]);
			mu_R_R[6] = min(largo[6], LARGO[6]);
			mu_R_R[7] = min(largo[7], LARGO[7]);

			mu_G_G[0] = min(largo_1[0], LARGO_1[0]);
			mu_G_G[1] = min(largo_1[1], LARGO_1[1]);
			mu_G_G[2] = min(largo_1[2], LARGO_1[2]);
			mu_G_G[3] = min(largo_1[3], LARGO_1[3]);
			mu_G_G[4] = min(largo_1[4], LARGO_1[4]);
			mu_G_G[5] = min(largo_1[5], LARGO_1[5]);
			mu_G_G[6] = min(largo_1[6], LARGO_1[6]);
			mu_G_G[7] = min(largo_1[7], LARGO_1[7]);

			mu_B_B[0] = min(largo_2[0], LARGO_2[0]);
			mu_B_B[1] = min(largo_2[1], LARGO_2[1]);
			mu_B_B[2] = min(largo_2[2], LARGO_2[2]);
			mu_B_B[3] = min(largo_2[3], LARGO_2[3]);
			mu_B_B[4] = min(largo_2[4], LARGO_2[4]);
			mu_B_B[5] = min(largo_2[5], LARGO_2[5]);
			mu_B_B[6] = min(largo_2[6], LARGO_2[6]);
			mu_B_B[7] = min(largo_2[7], LARGO_2[7]);

			noise_R_R = max(max(max(max(max(max(max(mu_R_R[0], mu_R_R[1]), mu_R_R[2]), mu_R_R[3]), mu_R_R[4]), mu_R_R[5]), mu_R_R[6]), mu_R_R[7]);
			noise_G_G = max(max(max(max(max(max(max(mu_G_G[0], mu_G_G[1]), mu_G_G[2]), mu_G_G[3]), mu_G_G[4]), mu_G_G[5]), mu_G_G[6]), mu_G_G[7]);
			noise_B_B = max(max(max(max(max(max(max(mu_B_B[0], mu_B_B[1]), mu_B_B[2]), mu_B_B[3]), mu_B_B[4]), mu_B_B[5]), mu_B_B[6]), mu_B_B[7]);

			//printf( "%f",noise_B_B);

			if ((noise_B_B >= 0.3))
			{

				float weights[9], sum_weights = 0, hold2, suma = 0;
				for (j = 0; j <= 7; j++)
				{
					sum_weights += (1 - mu_B_B[j]);
				}
				sum_weights = (sum_weights + 3 * sqrt(1 - noise_B_B)) / 2;
				weights[0] = (1 - mu_B_B[0]);
				weights[1] = (1 - mu_B_B[1]);
				weights[2] = (1 - mu_B_B[2]);
				weights[3] = (1 - mu_B_B[7]);
				weights[4] = 3 * sqrt(1 - noise_B_B);
				weights[5] = (1 - mu_B_B[3]);
				weights[6] = (1 - mu_B_B[6]);
				weights[7] = (1 - mu_B_B[5]);
				weights[8] = (1 - mu_B_B[4]);

				for (j = 0; j <= 8; j++)
				{
					for (x = 0; x <= 7; x++)
					{
						if (vectB[x] > vectB[x + 1])
						{
							hold = vectB[x];
							hold2 = weights[x];
							vectB[x] = vectB[x + 1];
							weights[x] = weights[x + 1];
							vectB[x + 1] = hold;
							weights[x + 1] = hold2;
						}
					}
				}
				for (j = 8; j >= 0; j--)
				{
					suma += weights[j];
					if (suma >= sum_weights)
					{
						if (j < 2)
						{
							sum_weights = sum_weights - (weights[0] + weights[1]);
							sum_weights = sum_weights / 2;
							suma = 0;
							for (F = 8; F >= 2; F--)
							{
								suma += weights[F];
								if (suma > sum_weights)
								{
									d_Pout[(Row * m + Col) * channels + 2] = vectB[F];
									F = -1;
								}
							}
							j = -1;
						}
						else
						{
							d_Pout[(Row * m + Col) * channels + 2] = vectB[j];
							//d_Pout[(Row * m + Col) * channels + 0] = d_Pout[(Row * m + Col) * channels + 0];
							j = -1;
						}
						suma = -1;
					}
				}
				//		fwrite (&CCC, 1, 1, header_file);
			}
			else
			{
				d_Pout[(Row * m + Col) * channels + 2] = vectB[4];
				//d_Pout[(Row * m + Col) * channels + 0] = 0;

				//		fwrite (&CCC, 1, 1, header_file);
			}

			if (noise_G_G >= 0.3)
			{

				float weights[9], sum_weights = 0, hold2, suma = 0;
				for (j = 0; j <= 7; j++)
				{
					sum_weights += (1 - mu_G_G[j]);
				}
				sum_weights = (sum_weights + 3 * sqrt(1 - noise_G_G)) / 2;
				weights[0] = (1 - mu_G_G[0]);
				weights[1] = (1 - mu_G_G[1]);
				weights[2] = (1 - mu_G_G[2]);
				weights[3] = (1 - mu_G_G[7]);
				weights[4] = 3 * sqrt(1 - noise_G_G);
				weights[5] = (1 - mu_G_G[3]);
				weights[6] = (1 - mu_G_G[6]);
				weights[7] = (1 - mu_G_G[5]);
				weights[8] = (1 - mu_G_G[4]);
				for (j = 0; j <= 8; j++)
				{
					for (x = 0; x <= 7; x++)
					{
						if (vectG[x] > vectG[x + 1])
						{
							hold = vectG[x];
							hold2 = weights[x];
							vectG[x] = vectG[x + 1];
							weights[x] = weights[x + 1];
							vectG[x + 1] = hold;
							weights[x + 1] = hold2;
						}
					}
				}
				for (j = 8; j >= 0; j--)
				{
					suma += weights[j];
					if (suma >= sum_weights)
					{
						if (j < 2)
						{
							sum_weights = sum_weights - (weights[0] + weights[1]);
							sum_weights = sum_weights / 2;
							suma = 0;
							for (F = 8; F >= 2; F--)
							{
								suma += weights[F];
								if (suma >= sum_weights)
								{
									d_Pout[(Row * m + Col) * channels + 1] = vectG[F];
									F = -1;
								}
							}
							j = -1;
						}
						else
						{
							d_Pout[(Row * m + Col) * channels + 1] = vectG[j];
							j = -1;
						}
						suma = -1;
					}
				}
				//		fwrite (&BBB, 1, 1, header_file);
			}
			else
			{
				d_Pout[(Row * m + Col) * channels + 1] = vectG[4];
				//		fwrite (&BBB, 1, 1, header_file);
			}

			if (noise_R_R >= 0.3)
			{

				float weights[9], sum_weights = 0, hold2, suma = 0;
				for (j = 0; j <= 7; j++)
				{
					sum_weights += (1 - mu_R_R[j]);
				}
				sum_weights = (sum_weights + 3 * sqrt(1 - noise_R_R)) / 2;
				weights[0] = (1 - mu_R_R[0]);
				weights[1] = (1 - mu_R_R[1]);
				weights[2] = (1 - mu_R_R[2]);
				weights[3] = (1 - mu_R_R[7]);
				weights[4] = 3 * sqrt(1 - noise_R_R);
				weights[5] = (1 - mu_R_R[3]);
				weights[6] = (1 - mu_R_R[6]);
				weights[7] = (1 - mu_R_R[5]);
				weights[8] = (1 - mu_R_R[4]);
				for (j = 0; j <= 8; j++)
				{
					for (x = 0; x <= 7; x++)
					{
						if (vectR[x] > vectR[x + 1])
						{
							hold = vectR[x];
							hold2 = weights[x];
							vectR[x] = vectR[x + 1];
							weights[x] = weights[x + 1];
							vectR[x + 1] = hold;
							weights[x + 1] = hold2;
						}
					}
				}
				for (j = 8; j >= 0; j--)
				{
					suma += weights[j];
					if (suma >= sum_weights)
					{
						if (j < 2)
						{
							sum_weights = sum_weights - (weights[0] + weights[1]);
							sum_weights = sum_weights / 2;
							suma = 0;
							for (F = 8; F >= 2; F--)
							{
								suma += weights[F];
								if (suma > sum_weights)
								{
									d_Pout[(Row * m + Col) * channels + 0] = vectR[F];
									F = -1;
								}
							}
							j = -1;
						}
						else
						{
							d_Pout[(Row * m + Col) * channels + 0] = vectR[j];
							j = -1;
						}
						suma = -1;
					}
				}
				//      fwrite (&AAA, 1, 1, header_file);
			}
			else
			{
				d_Pout[(Row * m + Col) * channels + 0] = vectR[4];
				//d_Pout[(Row * m + Col) * channels + 0] = 255;
				//		fwrite (&AAA, 1, 1, header_file);
			}
			
			
			//d_Pout[(Row * m + Col) * channels + 0] = 255;
		}
		
}

__global__ void FTSCF_GPU_Original_Params
(unsigned char* d_Pout, const unsigned char* d_Pin, int n, int m,
	
	float med_1, float var_1, float med_2, float med1, float med2, float var1, float THS) {

	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	int M = 0, j = 0, x = 0;
	float vectR[9], vectG[9], vectB[9], hold;

	float gam_small_1[18] = { 0 }, gam_big_1[18] = { 0 };
	float gam_small_2[18] = { 0 }, gam_big_2[18] = { 0 };

	float array_R[25];
	float array_G[25];
	float array_B[25];

	int F = 0, i = 0;

	const int channels = 3;

	if ((Row>1) && (Col>1) && (Row < m - 1) && (Col < n - 1)) {



		//int tid = omp_get_thread_num();
		//hacer el arreglo
		F = 0;

		for (i = -2; i <= 2; i++) {
			for (j = -2; j <= 2; j++) {
				array_R[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 0];
				array_G[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 1];
				array_B[F] = d_Pin[((Row + i) * m + (Col + j)) * 3 + 2];
				F++;
			}
		}


		// se copia a continuacion solo los 8-vecinos
		M = 0;
		for (F = 6; F <= 8; F++) {
			vectG[M] = (array_G[F]);
			vectR[M] = (array_R[F]);
			vectB[M] = (array_B[F]);
			M++;
		}
		for (F = 11; F <= 13; F++) {
			vectG[M] = (array_G[F]);
			vectR[M] = (array_R[F]);
			vectB[M] = (array_B[F]);
			M++;
		}
		for (F = 16; F <= 18; F++) {
			vectG[M] = (array_G[F]);
			vectR[M] = (array_R[F]);
			vectB[M] = (array_B[F]);
			M++;
		}


		float noreste_C_R, noreste_N1_R, noreste_N2_R, sur_C_R, sur_N1_R, sur_N2_R, noroeste_C_R, noroeste_N1_R, noroeste_N2_R;
		float este_C_R, este_N1_R, este_N2_R, oeste_C_R, oeste_N1_R, oeste_N2_R, sureste_C_R, sureste_N1_R, sureste_N2_R;
		float norte_C_R, norte_N1_R, norte_N2_R, suroeste_C_R, suroeste_N1_R, suroeste_N2_R;
		float suroeste_NW_R, suroeste_SE_R, sur_W_R, sur_E_R, sureste_SW_R, sureste_NE_R, este_S_R, este_N_R, noreste_SE_R, noreste_NW_R;
		float norte_W_R, norte_E_R, noroeste_NE_R, noroeste_SW_R, oeste_S_R, oeste_N_R;
		float noreste_C_G, noreste_N1_G, noreste_N2_G, sur_C_G, sur_N1_G, sur_N2_G, noroeste_C_G, noroeste_N1_G, noroeste_N2_G;
		float este_C_G, este_N1_G, este_N2_G, oeste_C_G, oeste_N1_G, oeste_N2_G, sureste_C_G, sureste_N1_G, sureste_N2_G;
		float norte_C_G, norte_N1_G, norte_N2_G, suroeste_C_G, suroeste_N1_G, suroeste_N2_G;
		float suroeste_NW_G, suroeste_SE_G, sur_W_G, sur_E_G, sureste_SW_G, sureste_NE_G, este_S_G, este_N_G, noreste_SE_G, noreste_NW_G;
		float norte_W_G, norte_E_G, noroeste_NE_G, noroeste_SW_G, oeste_S_G, oeste_N_G;
		float noreste_C_B, noreste_N1_B, noreste_N2_B, sur_C_B, sur_N1_B, sur_N2_B, noroeste_C_B, noroeste_N1_B, noroeste_N2_B;
		float este_C_B, este_N1_B, este_N2_B, oeste_C_B, oeste_N1_B, oeste_N2_B, sureste_C_B, sureste_N1_B, sureste_N2_B;
		float norte_C_B, norte_N1_B, norte_N2_B, suroeste_C_B, suroeste_N1_B, suroeste_N2_B;
		float suroeste_NW_B, suroeste_SE_B, sur_W_B, sur_E_B, sureste_SW_B, sureste_NE_B, este_S_B, este_N_B, noreste_SE_B, noreste_NW_B;
		float norte_W_B, norte_E_B, noroeste_NE_B, noroeste_SW_B, oeste_S_B, oeste_N_B;
		float largo[9], largo_1[9], largo_2[9], LARGO[9], LARGO_1[9], LARGO_2[9];
		float noise_R_R, noise_G_G, noise_B_B;
		int SW_C_B, SW_N1_B, SW_N2_B, SW_NW_B, SW_SE_B, S_C_B, S_N1_B, S_N2_B, S_W_B, S_E_B, SE_C_B, SE_N1_B, SE_N2_B, SE_SW_B, SE_NE_B;
		int E_C_B, E_N1_B, E_N2_B, E_S_B, E_N_B, NE_C_B, NE_N1_B, NE_N2_B, NE_SE_B, NE_NW_B, N_C_B, N_N1_B, N_N2_B, N_W_B, N_E_B;
		int NW_C_B, NW_N1_B, NW_N2_B, NW_NE_B, NW_SW_B, W_C_B, W_N1_B, W_N2_B, W_S_B, W_N_B;
		int SW_C_R, SW_N1_R, SW_N2_R, SW_NW_R, SW_SE_R, S_C_R, S_N1_R, S_N2_R, S_W_R, S_E_R, SE_C_R, SE_N1_R, SE_N2_R, SE_SW_R, SE_NE_R;
		int E_C_R, E_N1_R, E_N2_R, E_S_R, E_N_R, NE_C_R, NE_N1_R, NE_N2_R, NE_SE_R, NE_NW_R, N_C_R, N_N1_R, N_N2_R, N_W_R, N_E_R;
		int NW_C_R, NW_N1_R, NW_N2_R, NW_NE_R, NW_SW_R, W_C_R, W_N1_R, W_N2_R, W_S_R, W_N_R;
		int SW_C_G, SW_N1_G, SW_N2_G, SW_NW_G, SW_SE_G, S_C_G, S_N1_G, S_N2_G, S_W_G, S_E_G, SE_C_G, SE_N1_G, SE_N2_G, SE_SW_G, SE_NE_G;
		int E_C_G, E_N1_G, E_N2_G, E_S_G, E_N_G, NE_C_G, NE_N1_G, NE_N2_G, NE_SE_G, NE_NW_G, N_C_G, N_N1_G, N_N2_G, N_W_G, N_E_G;
		int NW_C_G, NW_N1_G, NW_N2_G, NW_NE_G, NW_SW_G, W_C_G, W_N1_G, W_N2_G, W_S_G, W_N_G;
		float cons1 = 255, cons2 = 255;


		// blue
		SW_C_B = abs(array_B[6] - array_B[12]);
		SW_N1_B = abs(array_B[10] - array_B[16]);
		SW_N2_B = abs(array_B[2] - array_B[8]);
		SW_NW_B = abs(array_B[12] - array_B[16]);
		SW_SE_B = abs(array_B[12] - array_B[8]);
		S_C_B = abs(array_B[7] - array_B[12]);
		S_N1_B = abs(array_B[6] - array_B[11]);
		S_N2_B = abs(array_B[8] - array_B[13]);
		S_W_B = abs(array_B[12] - array_B[11]);
		S_E_B = abs(array_B[12] - array_B[13]);
		SE_C_B = abs(array_B[8] - array_B[12]);
		SE_N1_B = abs(array_B[2] - array_B[6]);
		SE_N2_B = abs(array_B[14] - array_B[18]);
		SE_SW_B = abs(array_B[12] - array_B[6]);
		SE_NE_B = abs(array_B[12] - array_B[18]);
		E_C_B = abs(array_B[13] - array_B[12]);
		E_N1_B = abs(array_B[8] - array_B[7]);
		E_N2_B = abs(array_B[18] - array_B[17]);
		E_S_B = abs(array_B[12] - array_B[7]);
		E_N_B = abs(array_B[12] - array_B[17]);
		NE_C_B = abs(array_B[18] - array_B[12]);
		NE_N1_B = abs(array_B[14] - array_B[8]);
		NE_N2_B = abs(array_B[22] - array_B[16]);
		NE_SE_B = abs(array_B[12] - array_B[8]);
		NE_NW_B = abs(array_B[12] - array_B[16]);
		N_C_B = abs(array_B[17] - array_B[12]);
		N_N1_B = abs(array_B[18] - array_B[13]);
		N_N2_B = abs(array_B[16] - array_B[11]);
		N_W_B = abs(array_B[12] - array_B[11]);
		N_E_B = abs(array_B[12] - array_B[13]);
		NW_C_B = abs(array_B[16] - array_B[12]);
		NW_N1_B = abs(array_B[22] - array_B[18]);
		NW_N2_B = abs(array_B[10] - array_B[6]);
		NW_NE_B = abs(array_B[12] - array_B[18]);
		NW_SW_B = abs(array_B[12] - array_B[6]);
		W_C_B = abs(array_B[11] - array_B[12]);
		W_N1_B = abs(array_B[16] - array_B[17]);
		W_N2_B = abs(array_B[6] - array_B[7]);
		W_S_B = abs(array_B[12] - array_B[7]);
		W_N_B = abs(array_B[12] - array_B[17]);

		SW_C_G = abs(array_G[6] - array_G[12]);
		SW_N1_G = abs(array_G[10] - array_G[16]);
		SW_N2_G = abs(array_G[2] - array_G[8]);
		SW_NW_G = abs(array_G[12] - array_G[16]);
		SW_SE_G = abs(array_G[12] - array_G[8]);
		S_C_G = abs(array_G[7] - array_G[12]);
		S_N1_G = abs(array_G[6] - array_G[11]);
		S_N2_G = abs(array_G[8] - array_G[13]);
		S_W_G = abs(array_G[12] - array_G[11]);
		S_E_G = abs(array_G[12] - array_G[13]);
		SE_C_G = abs(array_G[8] - array_G[12]);
		SE_N1_G = abs(array_G[2] - array_G[6]);
		SE_N2_G = abs(array_G[14] - array_G[18]);
		SE_SW_G = abs(array_G[12] - array_G[6]);
		SE_NE_G = abs(array_G[12] - array_G[18]);
		E_C_G = abs(array_G[13] - array_G[12]);
		E_N1_G = abs(array_G[8] - array_G[7]);
		E_N2_G = abs(array_G[18] - array_G[17]);
		E_S_G = abs(array_G[12] - array_G[7]);
		E_N_G = abs(array_G[12] - array_G[17]);
		NE_C_G = abs(array_G[18] - array_G[12]);
		NE_N1_G = abs(array_G[14] - array_G[8]);
		NE_N2_G = abs(array_G[22] - array_G[16]);
		NE_SE_G = abs(array_G[12] - array_G[8]);
		NE_NW_G = abs(array_G[12] - array_G[16]);
		N_C_G = abs(array_G[17] - array_G[12]);
		N_N1_G = abs(array_G[18] - array_G[13]);
		N_N2_G = abs(array_G[16] - array_G[11]);
		N_W_G = abs(array_G[12] - array_G[11]);
		N_E_G = abs(array_G[12] - array_G[13]);
		NW_C_G = abs(array_G[16] - array_G[12]);
		NW_N1_G = abs(array_G[22] - array_G[18]);
		NW_N2_G = abs(array_G[10] - array_G[6]);
		NW_NE_G = abs(array_G[12] - array_G[18]);
		NW_SW_G = abs(array_G[12] - array_G[6]);
		W_C_G = abs(array_G[11] - array_G[12]);
		W_N1_G = abs(array_G[16] - array_G[17]);
		W_N2_G = abs(array_G[6] - array_G[7]);
		W_S_G = abs(array_G[12] - array_G[7]);
		W_N_G = abs(array_G[12] - array_G[17]);

		SW_C_R = abs(array_R[6] - array_R[12]);
		SW_N1_R = abs(array_R[10] - array_R[16]);
		SW_N2_R = abs(array_R[2] - array_R[8]);
		SW_NW_R = abs(array_R[12] - array_R[16]);
		SW_SE_R = abs(array_R[12] - array_R[8]);
		S_C_R = abs(array_R[7] - array_R[12]);
		S_N1_R = abs(array_R[6] - array_R[11]);
		S_N2_R = abs(array_R[8] - array_R[13]);
		S_W_R = abs(array_R[12] - array_R[11]);
		S_E_R = abs(array_R[12] - array_R[13]);
		SE_C_R = abs(array_R[8] - array_R[12]);
		SE_N1_R = abs(array_R[2] - array_R[6]);
		SE_N2_R = abs(array_R[14] - array_R[18]);
		SE_SW_R = abs(array_R[12] - array_R[6]);
		SE_NE_R = abs(array_R[12] - array_R[18]);
		E_C_R = abs(array_R[13] - array_R[12]);
		E_N1_R = abs(array_R[8] - array_R[7]);
		E_N2_R = abs(array_R[18] - array_R[17]);
		E_S_R = abs(array_R[12] - array_R[7]);
		E_N_R = abs(array_R[12] - array_R[17]);
		NE_C_R = abs(array_R[18] - array_R[12]);
		NE_N1_R = abs(array_R[14] - array_R[8]);
		NE_N2_R = abs(array_R[22] - array_R[16]);
		NE_SE_R = abs(array_R[12] - array_R[8]);
		NE_NW_R = abs(array_R[12] - array_R[16]);
		N_C_R = abs(array_R[17] - array_R[12]);
		N_N1_R = abs(array_R[18] - array_R[13]);
		N_N2_R = abs(array_R[16] - array_R[11]);
		N_W_R = abs(array_R[12] - array_R[11]);
		N_E_R = abs(array_R[12] - array_R[13]);
		NW_C_R = abs(array_R[16] - array_R[12]);
		NW_N1_R = abs(array_R[22] - array_R[18]);
		NW_N2_R = abs(array_R[10] - array_R[6]);
		NW_NE_R = abs(array_R[12] - array_R[18]);
		NW_SW_R = abs(array_R[12] - array_R[6]);
		W_C_R = abs(array_R[11] - array_R[12]);
		W_N1_R = abs(array_R[16] - array_R[17]);
		W_N2_R = abs(array_R[6] - array_R[7]);
		W_S_R = abs(array_R[12] - array_R[7]);
		W_N_R = abs(array_R[12] - array_R[17]);

		if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[12])) == 0) suroeste_C_R = 0;
		else	suroeste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[10] * array_R[16])) == 0) suroeste_N1_R = 0;
		else   suroeste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[10] * array_R[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[2] * array_R[8])) == 0) suroeste_N2_R = 0;
		else   suroeste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[2] * array_R[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[2], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[16])) == 0) suroeste_NW_R = 0;
		else	suroeste_NW_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[8])) == 0) suroeste_SE_R = 0;
		else	suroeste_SE_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[7] * array_R[12])) == 0) sur_C_R = 0;
		else	sur_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[7] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[11])) == 0) sur_N1_R = 0;
		else	sur_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[13])) == 0) sur_N2_R = 0;
		else   sur_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[11])) == 0) sur_W_R = 0;
		else	sur_W_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[13])) == 0) sur_E_R = 0;
		else	sur_E_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[12])) == 0) sureste_C_R = 0;
		else	sureste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[2])) == 0) sureste_N1_R = 0;
		else	sureste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[2])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[2], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[14] * array_R[18])) == 0) sureste_N2_R = 0;
		else	sureste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[14] * array_R[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[6])) == 0) sureste_SW_R = 0;
		else	sureste_SW_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[6])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[18])) == 0) sureste_NE_R = 0;
		else	sureste_NE_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[13] * array_R[12])) == 0) este_C_R = 0;
		else	este_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[13] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[7])) == 0) este_N1_R = 0;
		else	este_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[8] * array_R[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[17])) == 0) este_N2_R = 0;
		else	este_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[7])) == 0) este_S_R = 0;
		else	este_S_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[17])) == 0) este_N_R = 0;
		else	este_N_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[12])) == 0) noreste_C_R = 0;
		else	noreste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[14] * array_R[8])) == 0) noreste_N1_R = 0;
		else	noreste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[14] * array_R[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[22] * array_R[16])) == 0) noreste_N2_R = 0;
		else	noreste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[22] * array_R[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[8])) == 0) noreste_SE_R = 0;
		else	noreste_SE_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[16])) == 0) noreste_NW_R = 0;
		else	noreste_NW_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[17] * array_R[12])) == 0) norte_C_R = 0;
		else	norte_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[17] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[13])) == 0) norte_N1_R = 0;
		else	norte_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[18] * array_R[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[11])) == 0) norte_N2_R = 0;
		else	norte_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[13])) == 0) norte_E_R = 0;
		else	norte_E_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[13], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[11])) == 0) norte_W_R = 0;
		else	norte_W_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[12])) == 0) noroeste_C_R = 0;
		else	noroeste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[22] * array_R[18])) == 0) noroeste_N1_R = 0;
		else	noroeste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[22] * array_R[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[10])) == 0) noroeste_N2_R = 0;
		else	noroeste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[10])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[18])) == 0) noroeste_NE_R = 0;
		else	noroeste_NE_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[12])) == 0) noroeste_SW_R = 0;
		else	noroeste_SW_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[11] * array_R[12])) == 0) oeste_C_R = 0;
		else	oeste_C_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[11] * array_R[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[11], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[17])) == 0) oeste_N1_R = 0;
		else	oeste_N1_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[16] * array_R[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[7])) == 0) oeste_N2_R = 0;
		else	oeste_N2_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[6] * array_R[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[17])) == 0) oeste_N_R = 0;
		else	oeste_N_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[17], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[7])) == 0) oeste_S_R = 0;
		else	oeste_S_R = acos(((cons1 + cons1) + (cons2*cons2) + (array_R[12] * array_R[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_R[12], 2))))));

		if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[12])) == 0) suroeste_C_G = 0;
		else	suroeste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[10] * array_G[16])) == 0) suroeste_N1_G = 0;
		else   suroeste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[10] * array_G[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[2] * array_G[8])) == 0) suroeste_N2_G = 0;
		else   suroeste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[2] * array_G[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[2], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[16])) == 0) suroeste_NW_G = 0;
		else	suroeste_NW_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[8])) == 0) suroeste_SE_G = 0;
		else	suroeste_SE_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[7] * array_G[12])) == 0) sur_C_G = 0;
		else	sur_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[7] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[11])) == 0) sur_N1_G = 0;
		else	sur_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[13])) == 0) sur_N2_G = 0;
		else   sur_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[11])) == 0) sur_W_G = 0;
		else	sur_W_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[13])) == 0) sur_E_G = 0;
		else	sur_E_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[12])) == 0) sureste_C_G = 0;
		else	sureste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[2])) == 0) sureste_N1_G = 0;
		else	sureste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[2])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[2], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[14] * array_G[18])) == 0) sureste_N2_G = 0;
		else	sureste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[14] * array_G[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[6])) == 0) sureste_SW_G = 0;
		else	sureste_SW_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[6])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[18])) == 0) sureste_NE_G = 0;
		else	sureste_NE_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[13] * array_G[12])) == 0) este_C_G = 0;
		else	este_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[13] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[7])) == 0) este_N1_G = 0;
		else	este_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[8] * array_G[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[17])) == 0) este_N2_G = 0;
		else	este_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[7])) == 0) este_S_G = 0;
		else	este_S_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[17])) == 0) este_N_G = 0;
		else	este_N_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[12])) == 0) noreste_C_G = 0;
		else	noreste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[14] * array_G[8])) == 0) noreste_N1_G = 0;
		else	noreste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[14] * array_G[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[22] * array_G[16])) == 0) noreste_N2_G = 0;
		else	noreste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[22] * array_G[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[8])) == 0) noreste_SE_G = 0;
		else	noreste_SE_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[16])) == 0) noreste_NW_G = 0;
		else	noreste_NW_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[17] * array_G[12])) == 0) norte_C_G = 0;
		else	norte_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[17] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[13])) == 0) norte_N1_G = 0;
		else	norte_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[18] * array_G[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[11])) == 0) norte_N2_G = 0;
		else	norte_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[13])) == 0) norte_E_G = 0;
		else	norte_E_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[13], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[11])) == 0) norte_W_G = 0;
		else	norte_W_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[12])) == 0) noroeste_C_G = 0;
		else	noroeste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[22] * array_G[18])) == 0) noroeste_N1_G = 0;
		else	noroeste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[22] * array_G[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[10])) == 0) noroeste_N2_G = 0;
		else	noroeste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[10])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[18])) == 0) noroeste_NE_G = 0;
		else	noroeste_NE_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[12])) == 0) noroeste_SW_G = 0;
		else	noroeste_SW_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[11] * array_G[12])) == 0) oeste_C_G = 0;
		else	oeste_C_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[11] * array_G[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[11], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[17])) == 0) oeste_N1_G = 0;
		else	oeste_N1_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[16] * array_G[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[7])) == 0) oeste_N2_G = 0;
		else	oeste_N2_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[6] * array_G[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[17])) == 0) oeste_N_G = 0;
		else	oeste_N_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[17], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[7])) == 0) oeste_S_G = 0;
		else	oeste_S_G = acos(((cons1 + cons1) + (cons2*cons2) + (array_G[12] * array_G[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_G[12], 2))))));

		if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[12])) == 0) suroeste_C_B = 0;
		else	suroeste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[10] * array_B[16])) == 0) suroeste_N1_B = 0;
		else   suroeste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[10] * array_B[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[2] * array_B[8])) == 0) suroeste_N2_B = 0;
		else   suroeste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[2] * array_B[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[2], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[16])) == 0) suroeste_NW_B = 0;
		else	suroeste_NW_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[8])) == 0) suroeste_SE_B = 0;
		else	suroeste_SE_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[7] * array_B[12])) == 0) sur_C_B = 0;
		else	sur_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[7] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2)))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2)))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[11])) == 0) sur_N1_B = 0;
		else	sur_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[13])) == 0) sur_N2_B = 0;
		else   sur_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[11])) == 0) sur_W_B = 0;
		else	sur_W_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[13])) == 0) sur_E_B = 0;
		else	sur_E_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[12])) == 0) sureste_C_B = 0;
		else	sureste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[2])) == 0) sureste_N1_B = 0;
		else	sureste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[2])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[2], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[14] * array_B[18])) == 0) sureste_N2_B = 0;
		else	sureste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[14] * array_B[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[6])) == 0) sureste_SW_B = 0;
		else	sureste_SW_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[6])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[18])) == 0) sureste_NE_B = 0;
		else	sureste_NE_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[13] * array_B[12])) == 0) este_C_B = 0;
		else	este_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[13] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[7])) == 0) este_N1_B = 0;
		else	este_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[8] * array_B[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[17])) == 0) este_N2_B = 0;
		else	este_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[7])) == 0) este_S_B = 0;
		else	este_S_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[17])) == 0) este_N_B = 0;
		else	este_N_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[12])) == 0) noreste_C_B = 0;
		else	noreste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[14] * array_B[8])) == 0) noreste_N1_B = 0;
		else	noreste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[14] * array_B[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[14], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[22] * array_B[16])) == 0) noreste_N2_B = 0;
		else	noreste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[22] * array_B[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[8])) == 0) noreste_SE_B = 0;
		else	noreste_SE_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[8])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[8], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[16])) == 0) noreste_NW_B = 0;
		else	noreste_NW_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[16])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[17] * array_B[12])) == 0) norte_C_B = 0;
		else	norte_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[17] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[13])) == 0) norte_N1_B = 0;
		else	norte_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[18] * array_B[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[11])) == 0) norte_N2_B = 0;
		else	norte_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[13])) == 0) norte_E_B = 0;
		else	norte_E_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[13])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[13], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[11])) == 0) norte_W_B = 0;
		else	norte_W_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[11])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[12])) == 0) noroeste_C_B = 0;
		else	noroeste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[22] * array_B[18])) == 0) noroeste_N1_B = 0;
		else	noroeste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[22] * array_B[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[22], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[10])) == 0) noroeste_N2_B = 0;
		else	noroeste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[10])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[10], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[18])) == 0) noroeste_NE_B = 0;
		else	noroeste_NE_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[18])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[18], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[12])) == 0) noroeste_SW_B = 0;
		else	noroeste_SW_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[11] * array_B[12])) == 0) oeste_C_B = 0;
		else	oeste_C_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[11] * array_B[12])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[11], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[17])) == 0) oeste_N1_B = 0;
		else	oeste_N1_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[16] * array_B[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[16], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[7])) == 0) oeste_N2_B = 0;
		else	oeste_N2_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[6] * array_B[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[6], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[17])) == 0) oeste_N_B = 0;
		else	oeste_N_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[17])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[17], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))))));
		if (((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[7])) == 0) oeste_S_B = 0;
		else	oeste_S_B = acos(((cons1 + cons1) + (cons2*cons2) + (array_B[12] * array_B[7])) / ((sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[7], 2))*(sqrt(pow(cons1, 2) + pow(cons1, 2) + pow(array_B[12], 2))))));
		//	SUROESTE	

		/*
		med_1 = 1, var_1 = 0.8;
		med_2 = 0.1;
		*/
		if (suroeste_C_R > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((suroeste_C_R)-med_1), 2) / (2 * var_1))));
		if (suroeste_N1_R < med_2) gam_small_1[0] = 1;
		else 	gam_small_1[0] = (exp(-(pow(((suroeste_N1_R)-med_2), 2) / (2 * var_1))));
		if (suroeste_N2_R < med_2) gam_small_1[1] = 1;
		else 	gam_small_1[1] = (exp(-(pow(((suroeste_N2_R)-med_2), 2) / (2 * var_1))));
		if (suroeste_NW_R > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((suroeste_NW_R)-med_1), 2) / (2 * var_1))));
		if (suroeste_SE_R > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((suroeste_SE_R)-med_1), 2) / (2 * var_1))));
		largo[0] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (sur_C_R > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((sur_C_R)-med_1), 2) / (2 * var_1))));
		if (sur_N1_R < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((sur_N1_R)-med_2), 2) / (2 * var_1))));
		if (sur_N2_R < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((sur_N2_R)-med_2), 2) / (2 * var_1))));
		if (sur_W_R > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((sur_W_R)-med_1), 2) / (2 * var_1))));
		if (sur_E_R > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((sur_E_R)-med_1), 2) / (2 * var_1))));
		largo[1] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (sureste_C_R > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((sureste_C_R)-med_1), 2) / (2 * var_1))));
		if (sureste_N1_R < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((sureste_N1_R)-med_2), 2) / (2 * var_1))));
		if (sureste_N2_R < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((sureste_N2_R)-med_2), 2) / (2 * var_1))));
		if (sureste_NE_R > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((sureste_NE_R)-med_1), 2) / (2 * var_1))));
		if (sureste_SW_R > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((sureste_SW_R)-med_1), 2) / (2 * var_1))));
		largo[2] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (este_C_R > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((este_C_R)-med_1), 2) / (2 * var_1))));
		if (este_N1_R < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((este_N1_R)-med_2), 2) / (2 * var_1))));
		if (este_N2_R < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((este_N2_R)-med_2), 2) / (2 * var_1))));
		if (este_N_R > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((este_N_R)-med_1), 2) / (2 * var_1))));
		if (este_S_R > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((este_S_R)-med_1), 2) / (2 * var_1))));
		largo[3] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (noreste_C_R > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((noreste_C_R)-med_1), 2) / (2 * var_1))));
		if (noreste_N1_R < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((noreste_N1_R)-med_2), 2) / (2 * var_1))));
		if (noreste_N2_R < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((noreste_N2_R)-med_2), 2) / (2 * var_1))));
		if (noreste_NW_R > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((noreste_NW_R)-med_1), 2) / (2 * var_1))));
		if (noreste_SE_R > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((noreste_SE_R)-med_1), 2) / (2 * var_1))));
		largo[4] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (norte_C_R > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((norte_C_R)-med_1), 2) / (2 * var_1))));
		if (norte_N1_R < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((norte_N1_R)-med_2), 2) / (2 * var_1))));
		if (norte_N2_R < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((norte_N2_R)-med_2), 2) / (2 * var_1))));
		if (norte_W_R > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((norte_W_R)-med_1), 2) / (2 * var_1))));
		if (norte_E_R > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((norte_E_R)-med_1), 2) / (2 * var_1))));
		largo[5] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (noroeste_C_R > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((noroeste_C_R)-med_1), 2) / (2 * var_1))));
		if (noroeste_N1_R < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((noroeste_N1_R)-med_2), 2) / (2 * var_1))));
		if (noroeste_N2_R < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((noroeste_N2_R)-med_2), 2) / (2 * var_1))));
		if (noroeste_NE_R > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((noroeste_NE_R)-med_1), 2) / (2 * var_1))));
		if (noroeste_SW_R > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((noroeste_SW_R)-med_1), 2) / (2 * var_1))));
		largo[6] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (oeste_C_R > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((oeste_C_R)-med_1), 2) / (2 * var_1))));
		if (oeste_N1_R < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((oeste_N1_R)-med_2), 2) / (2 * var_1))));
		if (oeste_N2_R < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((oeste_N2_R)-med_2), 2) / (2 * var_1))));
		if (oeste_N_R > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((oeste_N_R)-med_1), 2) / (2 * var_1))));
		if (oeste_S_R > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((oeste_S_R)-med_1), 2) / (2 * var_1))));
		largo[7] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (suroeste_C_G > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((suroeste_C_G)-med_1), 2) / (2 * var_1))));
		if (suroeste_N1_G < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((suroeste_N1_G)-med_2), 2) / (2 * var_1))));
		if (suroeste_N2_G < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((suroeste_N2_G)-med_2), 2) / (2 * var_1))));
		if (suroeste_NW_G > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((suroeste_NW_G)-med_1), 2) / (2 * var_1))));
		if (suroeste_SE_G > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((suroeste_SE_G)-med_1), 2) / (2 * var_1))));
		largo_1[0] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (sur_C_G > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((sur_C_G)-med_1), 2) / (2 * var_1))));
		if (sur_N1_G < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((sur_N1_G)-med_2), 2) / (2 * var_1))));
		if (sur_N2_G < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((sur_N2_G)-med_2), 2) / (2 * var_1))));
		if (sur_W_G > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((sur_W_G)-med_1), 2) / (2 * var_1))));
		if (sur_E_G > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((sur_E_G)-med_1), 2) / (2 * var_1))));
		largo_1[1] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (sureste_C_G > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((sureste_C_G)-med_1), 2) / (2 * var_1))));
		if (sureste_N1_G < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((sureste_N1_G)-med_2), 2) / (2 * var_1))));
		if (sureste_N2_G < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((sureste_N2_G)-med_2), 2) / (2 * var_1))));
		if (sureste_NE_G > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((sureste_NE_G)-med_1), 2) / (2 * var_1))));
		if (sureste_SW_G > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((sureste_SW_G)-med_1), 2) / (2 * var_1))));
		largo_1[2] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (este_C_G > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((este_C_G)-med_1), 2) / (2 * var_1))));
		if (este_N1_G < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((este_N1_G)-med_2), 2) / (2 * var_1))));
		if (este_N2_G < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((este_N2_G)-med_2), 2) / (2 * var_1))));
		if (este_N_G > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((este_N_G)-med_1), 2) / (2 * var_1))));
		if (este_S_G > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((este_S_G)-med_1), 2) / (2 * var_1))));
		largo_1[3] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (noreste_C_G > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((noreste_C_G)-med_1), 2) / (2 * var_1))));
		if (noreste_N1_G < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((noreste_N1_G)-med_2), 2) / (2 * var_1))));
		if (noreste_N2_G < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((noreste_N2_G)-med_2), 2) / (2 * var_1))));
		if (noreste_NW_G > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((noreste_NW_G)-med_1), 2) / (2 * var_1))));
		if (noreste_SE_G > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((noreste_SE_G)-med_1), 2) / (2 * var_1))));
		largo_1[4] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (norte_C_G > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((norte_C_G)-med_1), 2) / (2 * var_1))));
		if (norte_N1_G < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((norte_N1_G)-med_2), 2) / (2 * var_1))));
		if (norte_N2_G < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((norte_N2_G)-med_2), 2) / (2 * var_1))));
		if (norte_W_G > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((norte_W_G)-med_1), 2) / (2 * var_1))));
		if (norte_E_G > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((norte_E_G)-med_1), 2) / (2 * var_1))));
		largo_1[5] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (noroeste_C_G > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((noroeste_C_G)-med_1), 2) / (2 * var_1))));
		if (noroeste_N1_G < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((noroeste_N1_G)-med_2), 2) / (2 * var_1))));
		if (noroeste_N2_G < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((noroeste_N2_G)-med_2), 2) / (2 * var_1))));
		if (noroeste_NE_G > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((noroeste_NE_G)-med_1), 2) / (2 * var_1))));
		if (noroeste_SW_G > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((noroeste_SW_G)-med_1), 2) / (2 * var_1))));
		largo_1[6] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (oeste_C_G > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((oeste_C_G)-med_1), 2) / (2 * var_1))));
		if (oeste_N1_G < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((oeste_N1_G)-med_2), 2) / (2 * var_1))));
		if (oeste_N2_G < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((oeste_N2_G)-med_2), 2) / (2 * var_1))));
		if (oeste_N_G > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((oeste_N_G)-med_1), 2) / (2 * var_1))));
		if (oeste_S_G > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((oeste_S_G)-med_1), 2) / (2 * var_1))));
		largo_1[7] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (suroeste_C_B > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((suroeste_C_B)-med_1), 2) / (2 * var_1))));
		if (suroeste_N1_B < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((suroeste_N1_B)-med_2), 2) / (2 * var_1))));
		if (suroeste_N2_B < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((suroeste_N2_B)-med_2), 2) / (2 * var_1))));
		if (suroeste_NW_B > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((suroeste_NW_B)-med_1), 2) / (2 * var_1))));
		if (suroeste_SE_B > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((suroeste_SE_B)-med_1), 2) / (2 * var_1))));
		largo_2[0] = (gam_big_2[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_2[2]);
		if (sur_C_B > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((sur_C_B)-med_1), 2) / (2 * var_1))));
		if (sur_N1_B < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((sur_N1_B)-med_2), 2) / (2 * var_1))));
		if (sur_N2_B < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((sur_N2_B)-med_2), 2) / (2 * var_1))));
		if (sur_W_B > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((sur_W_B)-med_1), 2) / (2 * var_1))));
		if (sur_E_B > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((sur_E_B)-med_1), 2) / (2 * var_1))));
		largo_2[1] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (sureste_C_B > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((sureste_C_B)-med_1), 2) / (2 * var_1))));
		if (sureste_N1_B < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((sureste_N1_B)-med_2), 2) / (2 * var_1))));
		if (sureste_N2_B < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((sureste_N2_B)-med_2), 2) / (2 * var_1))));
		if (sureste_NE_B > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((sureste_NE_B)-med_1), 2) / (2 * var_1))));
		if (sureste_SW_B > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((sureste_SW_B)-med_1), 2) / (2 * var_1))));
		largo_2[2] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (este_C_B > med_1) gam_big_1[0] = 1;

		else	gam_big_1[0] = (exp(-(pow(((este_C_B)-med_1), 2) / (2 * var_1))));
		if (este_N1_B < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((este_N1_B)-med_2), 2) / (2 * var_1))));
		if (este_N2_B < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((este_N2_B)-med_2), 2) / (2 * var_1))));
		if (este_N_B > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((este_N_B)-med_1), 2) / (2 * var_1))));
		if (este_S_B > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((este_S_B)-med_1), 2) / (2 * var_1))));
		largo_2[3] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (noreste_C_B > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((noreste_C_B)-med_1), 2) / (2 * var_1))));
		if (noreste_N1_B < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((noreste_N1_B)-med_2), 2) / (2 * var_1))));
		if (noreste_N2_B < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((noreste_N2_B)-med_2), 2) / (2 * var_1))));
		if (noreste_NW_B > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((noreste_NW_B)-med_1), 2) / (2 * var_1))));
		if (noreste_SE_B > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((noreste_SE_B)-med_1), 2) / (2 * var_1))));
		largo_2[4] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (norte_C_B > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((norte_C_B)-med_1), 2) / (2 * var_1))));
		if (norte_N1_B < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((norte_N1_B)-med_2), 2) / (2 * var_1))));
		if (norte_N2_B < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((norte_N2_B)-med_2), 2) / (2 * var_1))));
		if (norte_W_B > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((norte_W_B)-med_1), 2) / (2 * var_1))));
		if (norte_E_B > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((norte_E_B)-med_1), 2) / (2 * var_1))));
		largo_2[5] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (noroeste_C_B > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((noroeste_C_B)-med_1), 2) / (2 * var_1))));
		if (noroeste_N1_B < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((noroeste_N1_B)-med_2), 2) / (2 * var_1))));
		if (noroeste_N2_B < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((noroeste_N2_B)-med_2), 2) / (2 * var_1))));
		if (noroeste_NE_B > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((noroeste_NE_B)-med_1), 2) / (2 * var_1))));
		if (noroeste_SW_B > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((noroeste_SW_B)-med_1), 2) / (2 * var_1))));
		largo_2[6] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);
		if (oeste_C_B > med_1) gam_big_1[0] = 1;
		else	gam_big_1[0] = (exp(-(pow(((oeste_C_B)-med_1), 2) / (2 * var_1))));
		if (oeste_N1_B < med_2) gam_small_1[0] = 1;
		else	gam_small_1[0] = (exp(-(pow(((oeste_N1_B)-med_2), 2) / (2 * var_1))));
		if (oeste_N2_B < med_2) gam_small_1[1] = 1;
		else	gam_small_1[1] = (exp(-(pow(((oeste_N2_B)-med_2), 2) / (2 * var_1))));
		if (oeste_N_B > med_1) gam_big_1[1] = 1;
		else	gam_big_1[1] = (exp(-(pow(((oeste_N_B)-med_1), 2) / (2 * var_1))));
		if (oeste_S_B > med_1) gam_big_1[2] = 1;
		else	gam_big_1[2] = (exp(-(pow(((oeste_S_B)-med_1), 2) / (2 * var_1))));
		largo_2[7] = (gam_big_1[0] * gam_small_1[0] * gam_small_1[1] * gam_big_1[1] * gam_big_1[2]);

		/*
		med1 = 60;
		med2 = 10;
		var1 = 1000;
		*/

		if (SW_C_R > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((SW_C_R)-med1), 2) / (2 * var1))));
		if (SW_N1_R < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((SW_N1_R)-med2), 2) / (2 * var1))));
		if (SW_N2_R < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((SW_N2_R)-med2), 2) / (2 * var1))));
		if (SW_NW_R > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((SW_NW_R)-med1), 2) / (2 * var1))));
		if (SW_SE_R > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((SW_SE_R)-med1), 2) / (2 * var1))));
		LARGO[0] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (S_C_R > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((S_C_R)-med1), 2) / (2 * var1))));
		if (S_N1_R < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((S_N1_R)-med2), 2) / (2 * var1))));
		if (S_N2_R < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((S_N2_R)-med2), 2) / (2 * var1))));
		if (S_W_R > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((S_W_R)-med1), 2) / (2 * var1))));
		if (S_E_R > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((S_E_R)-med1), 2) / (2 * var1))));
		LARGO[1] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (SE_C_R > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((SE_C_R)-med1), 2) / (2 * var1))));
		if (SE_N1_R < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((SE_N1_R)-med2), 2) / (2 * var1))));
		if (SE_N2_R < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((SE_N2_R)-med2), 2) / (2 * var1))));
		if (SE_NE_R > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((SE_NE_R)-med1), 2) / (2 * var1))));
		if (SE_SW_R > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((SE_SW_R)-med1), 2) / (2 * var1))));
		LARGO[2] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (E_C_R > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((E_C_R)-med1), 2) / (2 * var1))));
		if (E_N1_R < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((E_N1_R)-med2), 2) / (2 * var1))));
		if (E_N2_R < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((E_N2_R)-med2), 2) / (2 * var1))));
		if (E_N_R > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((E_N_R)-med1), 2) / (2 * var1))));
		if (E_S_R > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((E_S_R)-med1), 2) / (2 * var1))));
		LARGO[3] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (NE_C_R > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((NE_C_R)-med1), 2) / (2 * var1))));
		if (NE_N1_R < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((NE_N1_R)-med2), 2) / (2 * var1))));
		if (NE_N2_R < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((NE_N2_R)-med2), 2) / (2 * var1))));
		if (NE_NW_R > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((NE_NW_R)-med1), 2) / (2 * var1))));
		if (NE_SE_R > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((NE_SE_R)-med1), 2) / (2 * var1))));
		LARGO[4] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (N_C_R > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((N_C_R)-med1), 2) / (2 * var1))));
		if (N_N1_R < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((N_N1_R)-med2), 2) / (2 * var1))));
		if (N_N2_R < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((N_N2_R)-med2), 2) / (2 * var1))));
		if (N_W_R > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((N_W_R)-med1), 2) / (2 * var1))));
		if (N_E_R > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((N_E_R)-med1), 2) / (2 * var1))));
		LARGO[5] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (NW_C_R > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((NW_C_R)-med1), 2) / (2 * var1))));
		if (NW_N1_R < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((NW_N1_R)-med2), 2) / (2 * var1))));
		if (NW_N2_R < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((NW_N2_R)-med2), 2) / (2 * var1))));
		if (NW_NE_R > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((NW_NE_R)-med1), 2) / (2 * var1))));
		if (NW_SW_R > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((NW_SW_R)-med1), 2) / (2 * var1))));
		LARGO[6] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (W_C_R > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((W_C_R)-med1), 2) / (2 * var1))));
		if (W_N1_R < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((W_N1_R)-med2), 2) / (2 * var1))));
		if (W_N2_R < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((W_N2_R)-med2), 2) / (2 * var1))));
		if (W_N_R > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((W_N_R)-med1), 2) / (2 * var1))));
		if (W_S_R > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((W_S_R)-med1), 2) / (2 * var1))));
		LARGO[7] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (SW_C_G > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((SW_C_G)-med1), 2) / (2 * var1))));
		if (SW_N1_G < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((SW_N1_G)-med2), 2) / (2 * var1))));
		if (SW_N2_G < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((SW_N2_G)-med2), 2) / (2 * var1))));
		if (SW_NW_G > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((SW_NW_G)-med1), 2) / (2 * var1))));
		if (SW_SE_G > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((SW_SE_G)-med1), 2) / (2 * var1))));
		LARGO_1[0] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (S_C_G > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((S_C_G)-med1), 2) / (2 * var1))));
		if (S_N1_G < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((S_N1_G)-med2), 2) / (2 * var1))));
		if (S_N2_G < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((S_N2_G)-med2), 2) / (2 * var1))));
		if (S_W_G > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((S_W_G)-med1), 2) / (2 * var1))));
		if (S_E_G > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((S_E_G)-med1), 2) / (2 * var1))));
		LARGO_1[1] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (SE_C_G > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((SE_C_G)-med1), 2) / (2 * var1))));
		if (SE_N1_G < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((SE_N1_G)-med2), 2) / (2 * var1))));
		if (SE_N2_G < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((SE_N2_G)-med2), 2) / (2 * var1))));
		if (SE_NE_G > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((SE_NE_G)-med1), 2) / (2 * var1))));
		if (SE_SW_G > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((SE_SW_G)-med1), 2) / (2 * var1))));
		LARGO_1[2] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (E_C_G > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((E_C_G)-med1), 2) / (2 * var1))));
		if (E_N1_G < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((E_N1_G)-med2), 2) / (2 * var1))));
		if (E_N2_G < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((E_N2_G)-med2), 2) / (2 * var1))));
		if (E_N_G > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((E_N_G)-med1), 2) / (2 * var1))));
		if (E_S_G > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((E_S_G)-med1), 2) / (2 * var1))));
		LARGO_1[3] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (NE_C_G > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((NE_C_G)-med1), 2) / (2 * var1))));
		if (NE_N1_G < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((NE_N1_G)-med2), 2) / (2 * var1))));
		if (NE_N2_G < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((NE_N2_G)-med2), 2) / (2 * var1))));
		if (NE_NW_G > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((NE_NW_G)-med1), 2) / (2 * var1))));
		if (NE_SE_G > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((NE_SE_G)-med1), 2) / (2 * var1))));
		LARGO_1[4] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (N_C_G > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((N_C_G)-med1), 2) / (2 * var1))));
		if (N_N1_G < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((N_N1_G)-med2), 2) / (2 * var1))));
		if (N_N2_G < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((N_N2_G)-med2), 2) / (2 * var1))));
		if (N_W_G > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((N_W_G)-med1), 2) / (2 * var1))));
		if (N_E_G > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((N_E_G)-med1), 2) / (2 * var1))));
		LARGO_1[5] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (NW_C_G > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((NW_C_G)-med1), 2) / (2 * var1))));
		if (NW_N1_G < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((NW_N1_G)-med2), 2) / (2 * var1))));
		if (NW_N2_G < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((NW_N2_G)-med2), 2) / (2 * var1))));
		if (NW_NE_G > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((NW_NE_G)-med1), 2) / (2 * var1))));
		if (NW_SW_G > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((NW_SW_G)-med1), 2) / (2 * var1))));
		LARGO_1[6] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (W_C_G > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((W_C_G)-med1), 2) / (2 * var1))));
		if (W_N1_G < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((W_N1_G)-med2), 2) / (2 * var1))));
		if (W_N2_G < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((W_N2_G)-med2), 2) / (2 * var1))));
		if (W_N_G > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((W_N_G)-med1), 2) / (2 * var1))));
		if (W_S_G > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((W_S_G)-med1), 2) / (2 * var1))));
		LARGO_1[7] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (SW_C_G > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((SW_C_B)-med1), 2) / (2 * var1))));
		if (SW_N1_B < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((SW_N1_B)-med2), 2) / (2 * var1))));
		if (SW_N2_B < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((SW_N2_B)-med2), 2) / (2 * var1))));
		if (SW_NW_B > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((SW_NW_B)-med1), 2) / (2 * var1))));
		if (SW_SE_B > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((SW_SE_B)-med1), 2) / (2 * var1))));
		LARGO_2[0] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (S_C_B > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((S_C_B)-med1), 2) / (2 * var1))));
		if (S_N1_B < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((S_N1_B)-med2), 2) / (2 * var1))));
		if (S_N2_B < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((S_N2_B)-med2), 2) / (2 * var1))));
		if (S_W_B > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((S_W_B)-med1), 2) / (2 * var1))));
		if (S_E_B > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((S_E_B)-med1), 2) / (2 * var1))));
		LARGO_2[1] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (SE_C_B > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((SE_C_B)-med1), 2) / (2 * var1))));
		if (SE_N1_B < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((SE_N1_B)-med2), 2) / (2 * var1))));
		if (SE_N2_B < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((SE_N2_B)-med2), 2) / (2 * var1))));
		if (SE_NE_B > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((SE_NE_B)-med1), 2) / (2 * var1))));
		if (SE_SW_B > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((SE_SW_B)-med1), 2) / (2 * var1))));
		LARGO_2[2] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (E_C_B > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((E_C_B)-med1), 2) / (2 * var1))));
		if (E_N1_B < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((E_N1_B)-med2), 2) / (2 * var1))));
		if (E_N2_B < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((E_N2_B)-med2), 2) / (2 * var1))));
		if (E_N_B > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((E_N_B)-med1), 2) / (2 * var1))));
		if (E_S_B > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((E_S_B)-med1), 2) / (2 * var1))));
		LARGO_2[3] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (NE_C_B > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((NE_C_B)-med1), 2) / (2 * var1))));
		if (NE_N1_B < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((NE_N1_B)-med2), 2) / (2 * var1))));
		if (NE_N2_B < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((NE_N2_B)-med2), 2) / (2 * var1))));
		if (NE_NW_B > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((NE_NW_B)-med1), 2) / (2 * var1))));
		if (NE_SE_B > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((NE_SE_B)-med1), 2) / (2 * var1))));
		LARGO_2[4] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (N_C_B > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((N_C_B)-med1), 2) / (2 * var1))));
		if (N_N1_B < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((N_N1_B)-med2), 2) / (2 * var1))));
		if (N_N2_B < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((N_N2_B)-med2), 2) / (2 * var1))));
		if (N_W_B > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((N_W_B)-med1), 2) / (2 * var1))));
		if (N_E_B > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((N_E_B)-med1), 2) / (2 * var1))));
		LARGO_2[5] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (NW_C_B > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((NW_C_B)-med1), 2) / (2 * var1))));
		if (NW_N1_B < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((NW_N1_B)-med2), 2) / (2 * var1))));
		if (NW_N2_B < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((NW_N2_B)-med2), 2) / (2 * var1))));
		if (NW_NE_B > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((NW_NE_B)-med1), 2) / (2 * var1))));
		if (NW_SW_B > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((NW_SW_B)-med1), 2) / (2 * var1))));
		LARGO_2[6] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);
		if (W_C_B > med1) gam_big_2[0] = 1;
		else	gam_big_2[0] = (exp(-(pow(((W_C_B)-med1), 2) / (2 * var1))));
		if (W_N1_B < med2) gam_small_2[0] = 1;
		else	gam_small_2[0] = (exp(-(pow(((W_N1_B)-med2), 2) / (2 * var1))));
		if (W_N2_B < med2) gam_small_2[1] = 1;
		else	gam_small_2[1] = (exp(-(pow(((W_N2_B)-med2), 2) / (2 * var1))));
		if (W_N_B > med1) gam_big_2[1] = 1;
		else	gam_big_2[1] = (exp(-(pow(((W_N_B)-med1), 2) / (2 * var1))));
		if (W_S_B > med1) gam_big_2[2] = 1;
		else	gam_big_2[2] = (exp(-(pow(((W_S_B)-med1), 2) / (2 * var1))));
		LARGO_2[7] = (gam_big_2[0] * gam_small_2[0] * gam_small_2[1] * gam_big_2[1] * gam_big_2[2]);

		float	mu_R_R[8], mu_G_G[8], mu_B_B[8];

		mu_R_R[0] = min(largo[0], LARGO[0]);
		mu_R_R[1] = min(largo[1], LARGO[1]);
		mu_R_R[2] = min(largo[2], LARGO[2]);
		mu_R_R[3] = min(largo[3], LARGO[3]);
		mu_R_R[4] = min(largo[4], LARGO[4]);
		mu_R_R[5] = min(largo[5], LARGO[5]);
		mu_R_R[6] = min(largo[6], LARGO[6]);
		mu_R_R[7] = min(largo[7], LARGO[7]);

		mu_G_G[0] = min(largo_1[0], LARGO_1[0]);
		mu_G_G[1] = min(largo_1[1], LARGO_1[1]);
		mu_G_G[2] = min(largo_1[2], LARGO_1[2]);
		mu_G_G[3] = min(largo_1[3], LARGO_1[3]);
		mu_G_G[4] = min(largo_1[4], LARGO_1[4]);
		mu_G_G[5] = min(largo_1[5], LARGO_1[5]);
		mu_G_G[6] = min(largo_1[6], LARGO_1[6]);
		mu_G_G[7] = min(largo_1[7], LARGO_1[7]);

		mu_B_B[0] = min(largo_2[0], LARGO_2[0]);
		mu_B_B[1] = min(largo_2[1], LARGO_2[1]);
		mu_B_B[2] = min(largo_2[2], LARGO_2[2]);
		mu_B_B[3] = min(largo_2[3], LARGO_2[3]);
		mu_B_B[4] = min(largo_2[4], LARGO_2[4]);
		mu_B_B[5] = min(largo_2[5], LARGO_2[5]);
		mu_B_B[6] = min(largo_2[6], LARGO_2[6]);
		mu_B_B[7] = min(largo_2[7], LARGO_2[7]);

		noise_R_R = max(max(max(max(max(max(max(mu_R_R[0], mu_R_R[1]), mu_R_R[2]), mu_R_R[3]), mu_R_R[4]), mu_R_R[5]), mu_R_R[6]), mu_R_R[7]);
		noise_G_G = max(max(max(max(max(max(max(mu_G_G[0], mu_G_G[1]), mu_G_G[2]), mu_G_G[3]), mu_G_G[4]), mu_G_G[5]), mu_G_G[6]), mu_G_G[7]);
		noise_B_B = max(max(max(max(max(max(max(mu_B_B[0], mu_B_B[1]), mu_B_B[2]), mu_B_B[3]), mu_B_B[4]), mu_B_B[5]), mu_B_B[6]), mu_B_B[7]);

		//printf( "%f",noise_B_B);

		if ((noise_B_B >= THS))
		{

			float weights[9], sum_weights = 0, hold2, suma = 0;
			for (j = 0; j <= 7; j++)
			{
				sum_weights += (1 - mu_B_B[j]);
			}
			sum_weights = (sum_weights + 3 * sqrt(1 - noise_B_B)) / 2;
			weights[0] = (1 - mu_B_B[0]);
			weights[1] = (1 - mu_B_B[1]);
			weights[2] = (1 - mu_B_B[2]);
			weights[3] = (1 - mu_B_B[7]);
			weights[4] = 3 * sqrt(1 - noise_B_B);
			weights[5] = (1 - mu_B_B[3]);
			weights[6] = (1 - mu_B_B[6]);
			weights[7] = (1 - mu_B_B[5]);
			weights[8] = (1 - mu_B_B[4]);

			for (j = 0; j <= 8; j++)
			{
				for (x = 0; x <= 7; x++)
				{
					if (vectB[x] > vectB[x + 1])
					{
						hold = vectB[x];
						hold2 = weights[x];
						vectB[x] = vectB[x + 1];
						weights[x] = weights[x + 1];
						vectB[x + 1] = hold;
						weights[x + 1] = hold2;
					}
				}
			}
			for (j = 8; j >= 0; j--)
			{
				suma += weights[j];
				if (suma >= sum_weights)
				{
					if (j < 2)
					{
						sum_weights = sum_weights - (weights[0] + weights[1]);
						sum_weights = sum_weights / 2;
						suma = 0;
						for (F = 8; F >= 2; F--)
						{
							suma += weights[F];
							if (suma > sum_weights)
							{
								d_Pout[(Row * m + Col) * channels + 2] = vectB[F];
								F = -1;
							}
						}
						j = -1;
					}
					else
					{
						d_Pout[(Row * m + Col) * channels + 2] = vectB[j];
						//d_Pout[(Row * m + Col) * channels + 0] = d_Pout[(Row * m + Col) * channels + 0];
						j = -1;
					}
					suma = -1;
				}
			}
			//		fwrite (&CCC, 1, 1, header_file);
		}
		else
		{
			d_Pout[(Row * m + Col) * channels + 2] = vectB[4];
			//d_Pout[(Row * m + Col) * channels + 0] = 0;

			//		fwrite (&CCC, 1, 1, header_file);
		}

		if (noise_G_G >= THS)
		{

			float weights[9], sum_weights = 0, hold2, suma = 0;
			for (j = 0; j <= 7; j++)
			{
				sum_weights += (1 - mu_G_G[j]);
			}
			sum_weights = (sum_weights + 3 * sqrt(1 - noise_G_G)) / 2;
			weights[0] = (1 - mu_G_G[0]);
			weights[1] = (1 - mu_G_G[1]);
			weights[2] = (1 - mu_G_G[2]);
			weights[3] = (1 - mu_G_G[7]);
			weights[4] = 3 * sqrt(1 - noise_G_G);
			weights[5] = (1 - mu_G_G[3]);
			weights[6] = (1 - mu_G_G[6]);
			weights[7] = (1 - mu_G_G[5]);
			weights[8] = (1 - mu_G_G[4]);
			for (j = 0; j <= 8; j++)
			{
				for (x = 0; x <= 7; x++)
				{
					if (vectG[x] > vectG[x + 1])
					{
						hold = vectG[x];
						hold2 = weights[x];
						vectG[x] = vectG[x + 1];
						weights[x] = weights[x + 1];
						vectG[x + 1] = hold;
						weights[x + 1] = hold2;
					}
				}
			}
			for (j = 8; j >= 0; j--)
			{
				suma += weights[j];
				if (suma >= sum_weights)
				{
					if (j < 2)
					{
						sum_weights = sum_weights - (weights[0] + weights[1]);
						sum_weights = sum_weights / 2;
						suma = 0;
						for (F = 8; F >= 2; F--)
						{
							suma += weights[F];
							if (suma >= sum_weights)
							{
								d_Pout[(Row * m + Col) * channels + 1] = vectG[F];
								F = -1;
							}
						}
						j = -1;
					}
					else
					{
						d_Pout[(Row * m + Col) * channels + 1] = vectG[j];
						j = -1;
					}
					suma = -1;
				}
			}
			//		fwrite (&BBB, 1, 1, header_file);
		}
		else
		{
			d_Pout[(Row * m + Col) * channels + 1] = vectG[4];
			//		fwrite (&BBB, 1, 1, header_file);
		}

		if (noise_R_R >= THS)
		{

			float weights[9], sum_weights = 0, hold2, suma = 0;
			for (j = 0; j <= 7; j++)
			{
				sum_weights += (1 - mu_R_R[j]);
			}
			sum_weights = (sum_weights + 3 * sqrt(1 - noise_R_R)) / 2;
			weights[0] = (1 - mu_R_R[0]);
			weights[1] = (1 - mu_R_R[1]);
			weights[2] = (1 - mu_R_R[2]);
			weights[3] = (1 - mu_R_R[7]);
			weights[4] = 3 * sqrt(1 - noise_R_R);
			weights[5] = (1 - mu_R_R[3]);
			weights[6] = (1 - mu_R_R[6]);
			weights[7] = (1 - mu_R_R[5]);
			weights[8] = (1 - mu_R_R[4]);
			for (j = 0; j <= 8; j++)
			{
				for (x = 0; x <= 7; x++)
				{
					if (vectR[x] > vectR[x + 1])
					{
						hold = vectR[x];
						hold2 = weights[x];
						vectR[x] = vectR[x + 1];
						weights[x] = weights[x + 1];
						vectR[x + 1] = hold;
						weights[x + 1] = hold2;
					}
				}
			}
			for (j = 8; j >= 0; j--)
			{
				suma += weights[j];
				if (suma >= sum_weights)
				{
					if (j < 2)
					{
						sum_weights = sum_weights - (weights[0] + weights[1]);
						sum_weights = sum_weights / 2;
						suma = 0;
						for (F = 8; F >= 2; F--)
						{
							suma += weights[F];
							if (suma > sum_weights)
							{
								d_Pout[(Row * m + Col) * channels + 0] = vectR[F];
								F = -1;
							}
						}
						j = -1;
					}
					else
					{
						d_Pout[(Row * m + Col) * channels + 0] = vectR[j];
						j = -1;
					}
					suma = -1;
				}
			}
			//      fwrite (&AAA, 1, 1, header_file);
		}
		else
		{
			d_Pout[(Row * m + Col) * channels + 0] = vectR[4];
			//d_Pout[(Row * m + Col) * channels + 0] = 255;
			//		fwrite (&AAA, 1, 1, header_file);
		}


		//d_Pout[(Row * m + Col) * channels + 0] = 255;
	}

}

