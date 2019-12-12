/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

#define PHIMAG_BLOCKDIM 512
#define COMPUTEQ_BLOCKDIM 256
#define COMPUTEQ_K_ELEMS_PER_GRID 1024

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};
/* Values in the k-space coordinate system are stored in constant memory
 * on the GPU */
__constant__ __device__ kValues KT[COMPUTEQ_K_ELEMS_PER_GRID];

__global__ void ComputePhiMagGPU(float* phiR, float* phiI, float* phiMag, int numK)
{
	int index = blockIdx.x*PHIMAG_BLOCKDIM + threadIdx.x;
	if (index < numK) 
	{
		float real = phiR[index];
		float imag = phiI[index];
		phiMag[index] = real * real + imag * imag;
	}
}

__global__ void ComputeQGPU(int numK, int KBaseIndex, float* x, float* y, float* z, float* Qr, float* Qi)
{
	__shared__ float sx, sy, sz, sQr, sQi;
	float expArg;

	int Xindex = blockIdx.x * PHIMAG_BLOCKDIM + threadIdx.x;

	sx = x[Xindex];
	sy = y[Xindex];
	sz = z[Xindex];
	sQr = Qr[Xindex];
	sQi = Qi[Xindex];

	for (int Kindex=0; Kindex < K_ELEMS_PER_GRID && KBaseIndex < numK; Kindex++)
	{
	  expArg = PIx2 * (KT[Kindex].Kx * sx +
					  KT[Kindex].Ky * sy +
					  KT[Kindex].Kz * sz);
	  sQr += KT[Kindex].PhiMag * cosf(expArg);
	  sQi += KT[Kindex].PhiMag * sinf(expArg);
	  KBaseIndex++;
	}
	Qr[Xindex] = sQr;
	Qi[Xindex] = sQi;
}

void 
ComputePhiMagCPU(int numK, 
				 float* phiR_d, float* phiI_d, 
				 float* phiMag_d) {
	int phiMagBlocks = numK / PHIMAG_BLOCKDIM;
	if (numK % PHIMAG_BLOCKDIM)
		phiMagBlocks++;
	dim3 DimBlock(PHIMAG_BLOCKDIM, 1, 1);
	dim3 DimGrid(phiMagBlocks, 1, 1);

	ComputePhiMagGPU <<< DimGrid, DimBlock >>> (phiR_d, phiI_d, phiMag_d, numK);
}

void
ComputeQCPU(int numK, int numX,
			float* x_d, float* y_d, float* z_d,
			kValues* kVals,
			float* Qr_d, float* Qi_d) {
	int Gridnum = numK / COMPUTEQ_K_ELEMS_PER_GRID;				//tile
	int Griddim = numX / COMPUTEQ_BLOCKDIM;

	if (numK % COMPUTEQ_K_ELEMS_PER_GRID)
		Gridnum++;
	if (numX % COMPUTEQ_BLOCKDIM)
		Griddim++;

	dim3 DimBlock(COMPUTEQ_BLOCKDIM, 1, 1);
	dim3 DimGrid(Griddim, 1, 1);

	for (int Grid = 0; Grid < Gridnum; Grid++) {
		// Put the tile of K values into constant mem
		int Base = Grid * COMPUTEQ_K_ELEMS_PER_GRID;
		kValues* kValsTile = kVals + Base;
		int num = MIN(COMPUTEQ_K_ELEMS_PER_GRID, numK - Base);

		cudaMemcpyToSymbol(KT, kValsTile, num * sizeof(kValues), 0);

		ComputeQGPU <<< DimGrid, DimBlock >>> (numK, Base, x_d, y_d, z_d, Qr_d, Qi_d);
	}
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}
