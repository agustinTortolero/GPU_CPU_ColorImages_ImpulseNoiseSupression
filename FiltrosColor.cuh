__global__ void VMF_GPU_GLOBAL(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);

__global__ void PeerGroup(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);

__global__ void Detection_FuzzyMetric(unsigned char* Noise, const unsigned char* d_Pin, int n, int m);
__global__ void Detection_Euclidean(unsigned char* Noise, const unsigned char* d_Pin, int n, int m);
__global__ void AMF_Filtering(unsigned char* image_out, const unsigned char* image_in, unsigned char* Noise, int n, int m);
__global__ void VMF_Filtering(unsigned char* image_out, const unsigned char* image_in, unsigned char* Noise, int n, int m);
__global__ void FiltradoPropuesta(unsigned char* image_out, const unsigned char* image_in, unsigned char* Noise, int n, int m);

__global__ void MarginalMedianFilter_Global_Forgetfull(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);
__global__ void VMF_Global_Forgetfull_Reuse(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);


__global__ void FiltradoPropuesta2(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);//implementacion mejorada

__global__ void FiltradoPropuesta_MMF(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);//implementacion mejorada
__global__ void FiltradoPropuesta_VMF(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);
__global__ void FiltradoPropuesta_AMF(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);

__device__ float Magnitud(unsigned char* VectR, unsigned char* VectG, unsigned char* VectB, unsigned int i, unsigned int j);

__global__ void VectorUnit_GPU_Global(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);
__global__ void BVDF_GPU_Global(unsigned char* d_Pout, unsigned char* d_Pin, int n, int m);



__global__ void FTSCF_GPU
(unsigned char* image_out,
	const unsigned char* image_in,
	const unsigned int a,
	const unsigned int b,
	const unsigned int THS,
	int n, int m);

__global__ void FTSCF_GPU_Original
(unsigned char* image_out,
	const unsigned char* image_in,
	int n, int m);

__global__ void FTSCF_GPU_Original_Params
(unsigned char* image_out,
	const unsigned char* image_in,
	int n, int m, 
	float med_1, float var_1, float med_2, float med1, float med2, float var1, float THS);
