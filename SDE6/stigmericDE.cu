/*Crear desde host un vector de numeros aleatorios con distribución uniforme.*/
#define D	100		//Number of dimensions 
#define Np	512	//Poblacion; 2,4,8,16,32,64,128,... 
#define PI 3.1415926535


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_profiler_api.h>
#include<cuda_runtime.h>
#include<ctime>


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

__global__ void stigmeric(double* aleat1, double* aleat2, double* X, double* F, double* U, double* FU, double f, double cr, int funcion, double* li, double* ls) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int arrayIndex = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int r1, r2, i, jrand = __double2int_ru(D * aleat1[arrayIndex * 3 + 2]), ind_supe;
	double u[D], caso1;
	if (arrayIndex > 0 && arrayIndex < Np - 1) {
		// COMPARING TRACES: ----------------------------------------------------
		if (F[arrayIndex - 1] < F[arrayIndex]) {
			r1 = arrayIndex - 1;
			r2 = arrayIndex;
		}
		else {
			r1 = arrayIndex;
			r2 = arrayIndex - 1;
		}
		if (F[r1] < F[arrayIndex + 1]) {
			ind_supe = r1;
			r1 = r2;
			r2 = arrayIndex + 1;
		}
		else {
			ind_supe = arrayIndex + 1;
			i = r2;
			r2 = r1;
			r1 = i;
		}
		//printf("HILO: %d; r1: %d, r2: %d; jrand: %d\n", arrayIndex, r1, r2, jrand);
		// ACTION (MUTANT VECTOR, TRIAL VECTOR): ++++++++++++++++++++++++++++++++
		for (i = 0; i < D; i++) {
			if (aleat2[arrayIndex * D + i] < cr || i == jrand) {
				caso1 = X[ind_supe * D + i] + f * (X[r1 * D + i] - X[r2 * D + i]);
				if (caso1 < ls[arrayIndex * D + i] && caso1 > li[arrayIndex * D + i])
					U[arrayIndex * D + i] = caso1;
				else
					U[arrayIndex * D + i] = ls[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
				/*U[arrayIndex * D + i] = X[ind_supe * D + i] + f * (X[r1 * D + i] - X[r2 * D + i]);
				if (U[arrayIndex * D + i] > ls[arrayIndex * D + i]) {
					U[arrayIndex * D + i] = (X[ind_supe * D + i] + f * (X[r1 * D + i] - X[r2 * D + i])) - ls[arrayIndex * D + i];
					//U[arrayIndex * D + i] = ls[arrayIndex * D + i] * aleat2[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
					//U[arrayIndex * D + i] = X[r1 * D + i] * aleat2[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
					//U[arrayIndex * D + i] =  (X[ind_supe * D + i] + f * (X[r1 * D + i] - X[r2 * D + i]))* aleat2[arrayIndex * D + i]* aleat2[arrayIndex * D + i];
					//U[arrayIndex * D + i] = li[arrayIndex * D + i] + (ls[arrayIndex * D + i] - li[arrayIndex * D + i]) * aleat2[arrayIndex * D + i];
					//U[arrayIndex * D + i] = li[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
				}
				if (U[arrayIndex * D + i] < li[arrayIndex * D + i]) {
					U[arrayIndex * D + i] = ls[arrayIndex * D + i] * aleat2[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
					//U[arrayIndex * D + i] = li[arrayIndex * D + i] + (ls[arrayIndex * D + i] - li[arrayIndex * D + i]) * aleat2[arrayIndex * D + i];
					//U[arrayIndex * D + i] = X[r1 * D + i] * aleat2[arrayIndex * D + i];
					//U[arrayIndex * D + i] = X[ind_supe * D + i] * aleat2[arrayIndex * D + i];
					//U[arrayIndex * D + i] = X[r2 * D + i] * aleat2[arrayIndex * D + i];
					//U[arrayIndex * D + i] = li[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
				}*/
			}
			else {
				U[arrayIndex * D + i] = X[arrayIndex * D + i];
			}
			//printf("%f\t", xbest[i]);
		}
		// TRACE UPDATE: ********************************************************
		FU[arrayIndex] = funciones[funcion](&U[arrayIndex * D]);
		if (FU[arrayIndex] < F[ind_supe]) {
			for (i = 0; i < D; i++) {
				X[ind_supe * D + i] = U[arrayIndex * D + i];
				__syncthreads();
			}
			F[ind_supe] = FU[arrayIndex];
			__syncthreads();
		}
		else if (FU[arrayIndex] < F[r1]) {
			for (i = 0; i < D; i++) {
				X[r1 * D + i] = U[arrayIndex * D + i];
				__syncthreads();
			}
			F[r1] = FU[arrayIndex];
			__syncthreads();
		}
		else if (FU[arrayIndex] < F[r2]) {
			for (i = 0; i < D; i++) {
				X[r2 * D + i] = U[arrayIndex * D + i];
				__syncthreads();
			}
			F[r2] = FU[arrayIndex];
			__syncthreads();
		}
	}
	// DIVERSITY KEEPING: *******************************************************
	/*if (arrayIndex == Np - 1) {
		for (i = 0; i < D; i++) {
			U[arrayIndex * D + i] = li[arrayIndex * D + i] + (ls[arrayIndex * D + i]- li[arrayIndex * D + i]) * aleat2[arrayIndex * D + i];
		}
		// TRACE UPDATE: ********************************************************
		FU[arrayIndex] = funciones[funcion](&U[arrayIndex * D]);
		if (FU[arrayIndex] < F[arrayIndex]) {
			F[arrayIndex] = FU[arrayIndex];
			__syncthreads();
			for (i = 0; i < D; i++) {
				X[arrayIndex * D + i] = U[arrayIndex * D + i];
				__syncthreads();
			}
		}
	}*/
}

__global__ void stigmeric2(double* aleat1, double* aleat2, double* X, double* F, double* U, double* FU, double f, double cr, int funcion, double* li, double* ls) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int arrayIndex = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int r1, r2, i, jrand = __double2int_ru(D * aleat1[arrayIndex * 3 + 2]), ind_supe;
	double u[D], fu;
	if (arrayIndex > 0 && arrayIndex < Np - 1) {
		if (F[arrayIndex - 1] < F[arrayIndex]) {
			r1 = arrayIndex - 1;
			r2 = arrayIndex;
		}
		else {
			r1 = arrayIndex;
			r2 = arrayIndex - 1;
		}
		if (F[r1] < F[arrayIndex + 1]) {
			ind_supe = r1;
			r1 = r2;
			r2 = arrayIndex + 1;
		}
		else {
			ind_supe = arrayIndex + 1;
			i = r2;
			r2 = r1;
			r1 = i;
		}
		//printf("HILO: %d; r1: %d, r2: %d; jrand: %d\n", arrayIndex, r1, r2, jrand);
		for (i = 0; i < D; i++) {
			if (aleat2[arrayIndex * D + i] < cr || i == jrand) {
				u[i] = X[ind_supe * D + i] + f * (X[r1 * D + i] - X[r2 * D + i]);
				if (u[i] > ls[arrayIndex * D + i])
					u[i] = li[arrayIndex * D + i] + (ls[arrayIndex * D + i] - li[arrayIndex * D + i]) * aleat2[arrayIndex * D + i];
				//u[i] = ls[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
				if (u[i] < li[arrayIndex * D + i])
					u[i] = X[r2 * D + i] * aleat2[arrayIndex * D + i];
				//u[i] = li[arrayIndex * D + i] * aleat2[arrayIndex * D + i];
			}
			else {
				u[i] = X[arrayIndex * D + i];
			}
			//printf("%f\t", xbest[i]);
		}
		fu = funciones[funcion](&U[arrayIndex * D]);
		if (fu < F[ind_supe]) {
			for (i = 0; i < D; i++) {
				X[ind_supe * D + i] = u[i];
				__syncthreads();
			}
			F[ind_supe] = fu;
			__syncthreads();
		}
		else if (fu < F[r1]) {
			for (i = 0; i < D; i++) {
				X[r1 * D + i] = u[i];
				__syncthreads();
			}
			F[r1] = fu;
			__syncthreads();
		}
		else if (fu < F[r2]) {
			for (i = 0; i < D; i++) {
				X[r2 * D + i] = u[i];
				__syncthreads();
			}
			F[r2] = fu;
			__syncthreads();
		}
	}
}

int main(void) {
	double* d_aleat, * d_X, * d_F, * d_FU, * d_limsup, * d_liminf, * d_xbest, * d_fbest, * d_U, * d_aleat3;//Apuntador a datos en device
	dim3 blocksXgridX(Np, 1), hilosXblockX(D, 1);  		//Bloques en grid;Hilos por bloque				
	size_t tamanoDoble = sizeof(double), tamanoInt = sizeof(int);
	int iteracion, funcion = 0, corridas, * h_ind, * d_ind, i, j, cont;
	curandGenerator_t gen;								//Crea variable generador 
	double* h_Data, * h_F, * h_fbest, * h_xbest, * h_aleat3;//Apuntador a datos en host
	float tiempo;

	//RESERVA MEMORIA E INICIALIZACIONES EN HOST Y EN DEVICE:
	h_Data = (double*)calloc(Np * D, tamanoDoble);		//[Np x D]	
	h_F = (double*)calloc(Np, tamanoDoble);				//[Np x 1]	
	h_fbest = (double*)calloc(1, tamanoDoble);			//[1 x 1 ]	
	h_xbest = (double*)calloc(D, tamanoDoble);			//[1 x D ]	
	h_ind = (int*)calloc(Np, tamanoInt);				//[Np x 1]
	h_aleat3 = (double*)calloc(Np * 3, tamanoDoble);	//[Np x 3]

	cudaMalloc((void**)&d_ind, Np * tamanoInt);			//[Np x 1] indices para buscar minimo
	cudaMalloc((void**)&d_aleat, Np * D * tamanoDoble);	//[Np x D] aleatorios para modificar individuos
	cudaMalloc((void**)&d_aleat3, Np * 3 * tamanoDoble);//[Np x 3] aleatorios para generar indices padres
	cudaMalloc((void**)&d_X, Np * D * tamanoDoble);		//[Np x D] poblacion soluciones candidatas
	cudaMalloc((void**)&d_U, Np * D * tamanoDoble);		//[Np x D] poblacion mutada
	cudaMalloc((void**)&d_liminf, Np * D * tamanoDoble);//[Np x D] limites superiores espacio busqueda
	cudaMalloc((void**)&d_limsup, Np * D * tamanoDoble);//[Np x D] limites inferiores espacio busqueda
	cudaMalloc((void**)&d_F, Np * tamanoDoble);			//[Np x 1] evaluacion de soluciones candidatas
	cudaMalloc((void**)&d_FU, Np * tamanoDoble);		//[Np x 1] evaluacion de soluciones mutadas
	cudaMalloc((void**)&d_fbest, 1 * tamanoDoble);		//[1 x 1 ] mejor evaluacion de solucion cand
	cudaMalloc((void**)&d_xbest, D * tamanoDoble);		//[1 x D ] mejor solucion candidata
	for (int i = 0; i < Np; i++) { h_ind[i] = i; }
	cudaMemcpy(d_ind, h_ind, Np * tamanoInt, cudaMemcpyHostToDevice);
	FILE* fPtr;										//Apuntador a archivo de texto
	fopen_s(&fPtr, "stigmergy6.txt", "a");			//Crear y abrir archivo en modo append
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);	//Crea generador	
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));	//Pone semilla en generador

	for (funcion = 0; funcion < 1; funcion++) {
		//fprintf(fPtr, "\nFunción %d; tiempo en S:\n", funcion);
		for (corridas = 0; corridas < 30; corridas++) {			
			//EXPERIMENTO 2: ###########################################################
			curandGenerateUniformDouble(gen, d_aleat, Np * D);		//Genera aleatorios en device
			iniciaL << <blocksXgridX, hilosXblockX >> > (d_liminf, d_limsup, funcion);
			iniciaX << <blocksXgridX, hilosXblockX >> > (d_X, d_aleat, d_liminf, d_limsup);
			evaluaFuncion << < blocksXgridX, 1 >> > (d_X, d_F, funcion);
			iteracion = 0;
			do {
				curandGenerateUniformDouble(gen, d_aleat3, Np * 3);			//Genera aleatorios en device
				curandGenerateUniformDouble(gen, d_aleat, Np * D);			//Genera aleatorios en device
				stigmeric << < 1, blocksXgridX >> > (d_aleat3, d_aleat, d_X, d_F, d_U, d_FU, 0.8, 0.3, funcion, d_liminf, d_limsup);
				++iteracion;
				cudaMemcpy(h_F, d_F, tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host
				cudaDeviceSynchronize();
				printf("Funcion: %d; iteraciones: %d, fitness: %.4f \n", funcion, iteracion, h_F[0]);
			} while (iteracion < 20000 && h_F[0]>0.01);
			fprintf(fPtr, "%f, %d, ", h_F[0], iteracion);
			printf("Funcion: %d; Corrida: %d; iteraciones: %d, fitness: %.4f \n", funcion, corridas, iteracion, h_F[0]);
		}
		fprintf(fPtr, "\n");
	}
	fclose(fPtr);

	/*printf("\nTiempo funcion stigmeric: %f mS\n\n", tiempo);

	cudaMemcpy(h_Data, d_X, Np * D * tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host
	cont = 0;
	for (i = 0; i < 1; i++) {
		for (j = 0; j < D; j++)
			printf("%f  ", h_Data[cont++]);
		printf("\n");
	}

	cudaMemcpy(h_F, d_F, Np * tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host
	cont = 0;
	for(j=0;j<Np;j++)
		printf("%f \n", h_F[j]);*/



		//cudaEvent_t start, stop;
		//cudaEventCreate(&start); cudaEventCreate(&stop);
		//cudaEventRecord(start, 0);
		//getMinGlobal << <1, blocksXgridX >> > (d_F, d_fbest, d_xbest, d_X, d_ind);
		//cudaEventRecord(stop, 0);
		//cudaEventSynchronize(stop);
		//cudaDeviceSynchronize();
		//cudaEventElapsedTime(&tiempo, start, stop);
		//printf("\nTiempo funcion getMinGlobal: %f mS", tiempo*5000);
		/*for (iteracion = 0; iteracion <5000; iteracion++) {
			//GENERA Y EVALUA MATRIZ DE VECTORES MUTADOS:
			curandGenerateUniformDouble(gen, d_aleat3, Np * 3);			//Genera aleatorios en device
			curandGenerateUniformDouble(gen, d_aleat, Np * D);			//Genera aleatorios en device
			//cudaDeviceSynchronize();
			generaEvaluaU << < 1, blocksXgridX >> > (d_aleat3, d_aleat, d_X, d_U, d_FU, d_xbest, 0.8, 0.3, funcion, d_liminf, d_limsup);
			//cudaDeviceSynchronize();
			//COMPARA FU Y F; OBTIENE fbest Y xbest:
			/*cudaMemcpy(h_F, d_F, Np * tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host
			cont = 0;
			for (i = 0; i < Np; i++) {
				printf("%f \n", h_F[cont++]);
			}
			printf("\n");
			cudaMemcpy(h_F, d_FU, Np * tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host
			cont = 0;
			for (i = 0; i < Np; i++) {
				printf("%f  \n", h_F[cont++]);
			}*/
			/*comparaFUyF << < 1, blocksXgridX >> > (d_X, d_U, d_F, d_FU);
			//cudaDeviceSynchronize();
			getMinGlobal << <1, blocksXgridX >> > (d_F, d_fbest, d_xbest, d_X, d_ind);
			//cudaDeviceSynchronize();
			//MUESTRA MEJOR FUNCION OBJETIVO EN CADA ITERACION:
			//cudaMemcpy(h_fbest, d_fbest, tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host
			cudaMemcpy(h_Data, d_U, Np * D * tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host
			/*cont = 0;
			for (i = 0; i < Np; i++) {
				for (j = 0; j < D; j++)
					printf("%f  ", h_Data[cont++]);
				printf("\n");
			}*/


			/*printf("Iteracion %d: Mejor fitness: %.4e \n \n", iteracion, h_fbest[0]);
		}*/
		/*cudaMemcpy(h_xbest, d_xbest, D * tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host
		for (i = 0; i < D; i++) printf("%f   ", h_xbest[i]);*/

	curandDestroyGenerator(gen);	//Destruye generador de aleatorios
	cudaFree(d_liminf);				//Libera memoria device
	cudaFree(d_limsup);
	cudaFree(d_aleat);
	cudaFree(d_aleat3);
	cudaFree(d_X);
	cudaFree(d_U);
	cudaFree(d_F);
	cudaFree(d_FU);
	cudaFree(d_fbest);
	cudaFree(d_xbest);
	cudaFree(d_ind);

	free(h_Data);					//Libera memoria de host
	free(h_F);
	free(h_fbest);
	free(h_xbest);
	free(h_ind);
	free(h_aleat3);
	return EXIT_SUCCESS;
}