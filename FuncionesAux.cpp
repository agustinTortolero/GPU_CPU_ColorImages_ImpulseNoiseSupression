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

using namespace std;

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
}

void ObtenerPath(char* Dir, int i) {

	char numeroDeImagen[5];
	_itoa_s(i, numeroDeImagen, 10);


	strcat_s(Dir, 200, numeroDeImagen);

	strcat_s(Dir, 200, ".bmp");

}


void ObtenerPathWrite(char* Dir2, int i) {

	char numeroDeImagen[5];
	int aux = i;
	_itoa_s(aux, numeroDeImagen, 5,10);
	strcat_s(Dir2, 200, numeroDeImagen);
	strcat_s(Dir2, 200, ".bmp");

}
