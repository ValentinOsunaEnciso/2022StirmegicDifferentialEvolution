/*Differential Evolution, version DE/best/1/bin; Valentin Osuna-Enciso, DIC/2020, UDG*/
#include "interface.h"

//MANY LOCAL MINIMA:
__device__ double ackley(double* x) {
	double sum1 = 0.0, sum2 = 0.0, a = 20.0, b = 0.2, c = 2 * PI;
	for (int i = 0; i < D; i++) {
		sum1 += pow(x[i], 2.0);
		sum2 += cos(c * x[i]);
	}
	return -a * exp(-b * sqrt(sum1 / D)) - exp(sum2 / D) + a + exp(1.0);
}

__device__ double* acklLims() {
	double limites[] = { -32.768, 32.768 };
	return limites;
}

__device__ double griewank(double* x) {
	double sum = 0.0, prod = 1.0;
	for (int i = 0; i < D; i++) {
		sum += pow(x[i], 2.0) / 4000.0;
		prod = prod * cos(x[i] / sqrt(i + 1.0));
	}
	return sum - prod + 1.0;
}

__device__ double* grieLims() {
	double limites[] = { -600.0, 600.0 };
	return limites;
}

__device__ double levi(double* x) {
	double sum = 0.0, term1, term3, prod = 1.0, w[D];
	int i;
	for (i = 0; i < D; i++)
		w[i] = 1.0 + (x[i] - 1.0) / 4.0;
	term1 = pow(sin(PI * w[0]), 2.0);
	term3 = pow(w[D - 1] - 1.0, 2.0) * (1.0 + pow(sin(2.0 * PI * w[D - 1]), 2.0));
	for (i = 0; i < D - 1; i++)
		sum += pow(w[i] - 1, 2.0) * (1.0 + 10.0 * pow(sin(PI * w[i] + 1.0), 2.0));
	return term1 + sum + term3;
}

__device__ double* leviLims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}

__device__ double rastrigin(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D; i++)
		sum += pow(x[i], 2.0) - 10.0 * cos(2.0 * PI * x[i]);
	return sum + 10.0 * D;
}

__device__ double* rastLims() {
	double limites[] = { -5.12, 5.12 };
	return limites;
}

__device__ double schwefel(double* x) {
	double f = 0.0;
	for (int i = 0; i < D; i++)
		f -= x[i] * sin(sqrt(fabs(x[i])));
	f += 418.9829 * D;
	return f;
}

__device__ double* schwLims() {
	double limites[] = { -500.0, 500.0 };
	return limites;
}
//BOWL SHAPED:

__device__ double perm(double* x) {
	double outer = 0.0, inner;
	for (int i = 0; i < D; i++) {
		inner = 0.0;
		for (int j = 0; j < D; j++)
			inner += (double(j) + 1.0 + 10.0) * (pow(x[j], double(i) + 1.0) - pow(1.0 / (double(j) + 1.0), double(i) + 1.0));
		outer += pow(inner, 2.0);
	}
	return outer;
}

__device__ double* permLims() {
	double limites[] = { -double(D), double(D) };
	return limites;
}

__device__ double hyperellipsoid(double* x) {
	double outer = 0.0, inner;
	for (int i = 0; i < D; i++) {
		inner = 0.0;
		for (int j = 0; j < D; j++)
			inner += pow(x[j], 2.0);
		outer += inner;
	}
	return outer;
}

__device__ double* hypeLims() {
	double limites[] = { -65.536, 65.536 };
	return limites;
}

__device__ double sphere(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D; i++)
		sum += pow(x[i], 2.0);
	return sum;
}

__device__ double* spheLims() {
	double limites[] = { -5.12, 5.12 };
	return limites;
}

__device__ double sumpow(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D; i++)
		sum += pow(fabs(x[i]), double(i) + 2.0);
	return sum;
}

__device__ double* sumpLims() {
	double limites[] = { -1.0, 1.0 };
	return limites;
}

__device__ double sumsquares(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D; i++)
		sum += double(i) * pow(x[i], 2.0);
	return sum;
}

__device__ double* sumsLims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}

__device__ double trid(double* x) {
	double sum1 = pow(x[0] - 1, 2.0), sum2 = 0.0;
	for (int i = 1; i < D; i++) {
		sum1 += pow(x[i] - 1, 2.0);
		sum2 += x[i] * x[i - 1];
	}
	return sum1 - sum2;
}

__device__ double* tridLims() {
	double limites[] = { -pow(double(D),2.0), pow(double(D),2.0) };
	return limites;
}

//PLATE SHAPED:

__device__ double zakharov(double* x) {
	double sum1 = 0.0, sum2 = 0.0;
	for (int i = 0; i < D; i++) {
		sum1 += pow(x[i], 2.0);
		sum2 += 0.5 * (double(i) + 1.0) * x[i];
	}
	return sum1 + pow(sum2, 2.0) + pow(sum2, 4.0);
}

__device__ double* zakhLims() {
	double limites[] = { -5.0, 10.0 };
	return limites;
}

//VALLEY SHAPED:

__device__ double dixonprice(double* x) {
	double sum = 0.0;
	for (int i = 1; i < D; i++)
		sum += (double(i) + 1.0) * pow(2 * pow(x[i], 2.0) - x[i - 1], 2.0);
	return sum + pow(x[0] - 1.0, 2.0);
}

__device__ double* dixoLims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}

__device__ double rosenbrock(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D - 1; i++)
		sum += 100.0 * pow(x[i + 1] - pow(x[i], 2.0), 2.0) + pow(x[i] - 1.0, 2.0);
	return sum;
}

__device__ double* roseLims() {
	double limites[] = { -5.0, 10.0 };
	return limites;
}

__device__ int* shuffleR1R2(double* x, int indiceHilo) {
	int r1 = Np * x[indiceHilo * 2], r2 = Np * x[indiceHilo * 2 + 1], cont = 2;
	while ((r1 == r2) && cont < 2 * Np) {
		r1 = __double2int_ru(Np * x[indiceHilo + cont + 1]);
		r2 = __double2int_ru(Np * x[indiceHilo + cont++]);
	}
	int limites[] = { r1, r2 };
	return limites;
}

//Lineas clave para llamadas dinamicas a funciones:
typedef double (*llamadaaFunciones)(double*);
__device__ llamadaaFunciones funciones[] = { ackley, griewank, levi, rastrigin, schwefel,  sphere, sumpow, sumsquares, trid, zakharov, dixonprice, rosenbrock };
typedef double* (*llamadaaFunciones2)();
__device__ llamadaaFunciones2 limites[] = { acklLims, grieLims, leviLims, rastLims, schwLims,  spheLims, sumpLims, sumsLims, tridLims, zakhLims, dixoLims, roseLims };

__global__ void evaluaFuncion(double* x, double* out, int funcion) {
	//evalua funciones; version global memory.
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int id = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	out[id] = funciones[funcion](&x[id * D]);
	__syncthreads();
}

__global__ void f1b(double* x, double* out) {
	//Schwefel; f*=-41898.29; lims=[-500,500]; x*=[420.9687,...]
	//version shared memory.
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int i, id = blockId * blockDim.x + threadIdx.x;
	__shared__ double compartida[D];
	for (i = 0; i < D; i++) compartida[id + i] = x[id * D + i];
	//__syncthreads();
	double resultado = 0.0;
	for (i = id; i < id + D; i++)
		resultado -= compartida[i] * sin(sqrt(fabs(compartida[i])));
	out[id] = resultado;
}

__global__ void iniciaX(double* X, double* A, double* inferior, double* superior) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int id = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	X[id] = inferior[id] + (superior[id] - inferior[id]) * A[id];
}

__global__ void iniciaL(double* inferior, double* superior, int funcion) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int id = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	double* x;
	x = limites[funcion]();
	inferior[id] = x[0];
	superior[id] = x[1];
}

//############### Busca el minimo global:  #################################################
//##(Taken from: http://supercomputingblog.com/cuda/cuda-tutorial-3-thread-communication/)##
__global__ void getMinGlobal(double* F, double* fbest, double* xbest, double* X, int* indices) {
	__shared__ double min[Np];		// Declare array to be in shared memory.
	__shared__ int indexes[Np];		//pInd is array of indexs where a min is found
	// Calculate which element this thread reads from memory	
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int arrayIndex = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x, i;
	min[threadIdx.x] = F[arrayIndex];
	indexes[threadIdx.x] = indices[arrayIndex];
	__syncthreads();
	int nTotalThreads = blockDim.x;	// Total number of active threads
	//printf("HILO: %d\n", threadIdx.x);
	while (nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < halfPoint) {
			// Get the shared value stored by another thread
			double temp = min[threadIdx.x + halfPoint];
			//printf("temp: %f\n", temp);
			int temp1 = indexes[threadIdx.x + halfPoint];
			//printf("hilo: %d; medio: %d, %f y %f; %d\n", threadIdx.x, halfPoint, min[threadIdx.x], temp, blockDim.x);
			if (temp < min[threadIdx.x]) {
				min[threadIdx.x] = temp;
				indexes[threadIdx.x] = temp1;
			}
		}
		__syncthreads();
		nTotalThreads = (nTotalThreads >> 1);	// divide by two.
	}
	// At this point in time, thread zero has the min
	// It's time for thread zero to write it's final results.
	// Note that the address structure of pResults is different, because
	// there is only one value for every thread block.
	if (threadIdx.x == 0) {
		fbest[blockIdx.y + blockIdx.x] = min[0];
		for (i = 0; i < D; i++) { xbest[i] = X[(indexes[0]) * D + i]; }
	}
	__syncthreads();
}

__global__ void generaEvaluaU(double* aleat1, double* aleat2, double* X, double* U, double* FU, double* xbest, double f, double cr, int funcion, double* li, double* ls) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int arrayIndex = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	//int r1 = Np * aleat1[arrayIndex * 3], r2 = Np * aleat1[arrayIndex * 3 + 1], jrand = D * aleat1[arrayIndex * 3 + 2];
	int r1, r2, i, jrand = __double2int_ru(D * aleat1[arrayIndex * 3 + 2]);
	int* x;
	x = shuffleR1R2(aleat1, arrayIndex);
	r1 = x[0];
	r2 = x[1];
	//printf("HILO: %d; r1: %d, r2: %d; jrand: %d\n", arrayIndex, r1, r2, jrand);
	for (i = 0; i < D; i++) {
		if (aleat2[arrayIndex * D + i] < cr || i == jrand) {
			U[arrayIndex * D + i] = xbest[i] + f * (X[r1 * D + i] - X[r2 * D + i]);
			if (U[arrayIndex * D + i] > ls[arrayIndex * D + i])
				U[arrayIndex * D + i] = ls[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
			if (U[arrayIndex * D + i] < li[arrayIndex * D + i])
				U[arrayIndex * D + i] = li[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
		}
		else {
			U[arrayIndex * D + i] = X[arrayIndex * D + i];
		}
		//printf("%f\t", xbest[i]);
	}
	FU[arrayIndex] = funciones[funcion](&U[arrayIndex * D]);
	__syncthreads();
}

__global__ void comparaFUyF(double* X, double* U, double* F, double* FU) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int arrayIndex = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x, i;
	if (FU[arrayIndex] < F[arrayIndex]) {
		for (i = 0; i < D; i++)
			X[arrayIndex * D + i] = U[arrayIndex * D + i];
		F[arrayIndex] = FU[arrayIndex];
	}
	__syncthreads();
}

extern "C" void generaEvaluaMutantes(double* d_aleat3, double* d_aleat, double* d_X, double* d_U, double* d_FU, double* d_xbest, float F, float Cr, int funcion, double* d_liminf, double* d_limsup) {
	dim3 blocksXgridX(Np, 1);  		//Bloques en grid;Hilos por bloque
	generaEvaluaU << < 1, blocksXgridX >> > (d_aleat3, d_aleat, d_X, d_U, d_FU, d_xbest, F, Cr, funcion, d_liminf, d_limsup);
}

extern "C" void obtieneMinimoGlobal(double* d_F, double* d_fbest, double* d_xbest, double* d_X, int* d_ind) {
	dim3 blocksXgridX(Np, 1);  		//Bloques en grid;Hilos por bloque
	getMinGlobal << <1, blocksXgridX >> > (d_F, d_fbest, d_xbest, d_X, d_ind);
}

extern "C" void evaluaFitnessPoblacion(double* d_X, double* d_F, int funcion) {
	dim3 blocksXgridX(Np, 1);  		//Bloques en grid;Hilos por bloque
	evaluaFuncion << < blocksXgridX, 1 >> > (d_X, d_F, funcion);
}

extern "C" void iniciaPoblacionyLimites(double* d_X, double* d_aleat, double* d_liminf, double* d_limsup, int funcion) {
	dim3 blocksXgridX(Np, 1), hilosXblockX(D, 1);  		//Bloques en grid;Hilos por bloque
	iniciaL << <blocksXgridX, hilosXblockX >> > (d_liminf, d_limsup, funcion);
	iniciaX << <blocksXgridX, hilosXblockX >> > (d_X, d_aleat, d_liminf, d_limsup);
}

extern "C" void comparaFitnessUyX(double* d_X, double* d_U, double* d_F, double* d_FU) {
	dim3 blocksXgridX(Np, 1), hilosXblockX(D, 1);  		//Bloques en grid;Hilos por bloque
	comparaFUyF << < 1, blocksXgridX >> > (d_X, d_U, d_F, d_FU);
}

