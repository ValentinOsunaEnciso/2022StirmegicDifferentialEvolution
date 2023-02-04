/*Differential Evolution, version DE/best/1/bin; Valentin Osuna-Enciso, DIC/2020, UDG*/
#include "interface.h"

//FUNCIONES NUEVAS; SE NOMBRAN COMO APARECEN EN [2015Dogon]:
//Stepint; D>2; limits={-5.12,5.12}; f*=25-6*D; x*=(-5.12<=x_i<-5); Unimodal; Separable.
__device__ double F1(double* x) {
	double suma = 0.0;
	for (int i = 0; i < D; i++) {
		suma += ceil(x[i]);
	}
	return 25 + suma;
}
__device__ double* F1lims() {
	double limites[] = { -5.12, 5.12 };
	return limites;
}
//Step; D>2; limits={-100,100}; f*=0; Unimodal; Separable.
__device__ double F2(double* x) {
	double suma = 0.0;
	for (int i = 0; i < D; i++) {
		suma += pow(floor(x[i] + 0.5), 2.0);
	}
	return suma;
}
__device__ double* F2lims() {
	double limites[] = { -100.0, 100.0 };
	return limites;
}
//Sphere; D>2; limits={-100,100}; f*=0; Unimodal; Separable.
__device__ double F3(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D; i++)
		sum += pow(x[i], 2.0);
	return sum;
}
__device__ double* F3lims() {
	double limites[] = { -5.12, 5.12 };
	return limites;
}
//Sumsquares; D>2; limits={-10,10}; f*=0; Unimodal; Separable.
__device__ double F4(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D; i++)
		sum += double(i) * pow(x[i], 2.0);
	return sum;
}
__device__ double* F4lims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}
//Trid; D>2; limits={-D^2,D^2}; f*=-D(D+4)(D-1)/6; Unimodal; No separable.
__device__ double F5(double* x) {
	double sum1 = pow(x[0] - 1, 2.0), sum2 = 0.0;
	for (int i = 1; i < D; i++) {
		sum1 += pow(x[i] - 1, 2.0);
		sum2 += x[i] * x[i - 1];
	}
	return sum1 - sum2;
}
__device__ double* F5lims() {
	double limites[] = { -pow(double(D),2.0), pow(double(D),2.0) };
	return limites;
}
//Zakharov; D>2; limites={-5,10}; f*=0; Unimodal; No separable.
__device__ double F6(double* x) {
	double sum1 = 0.0, sum2 = 0.0;
	for (int i = 0; i < D; i++) {
		sum1 += pow(x[i], 2.0);
		sum2 += 0.5 * (double(i) + 1.0) * x[i];
	}
	return sum1 + pow(sum2, 2.0) + pow(sum2, 4.0);
}
__device__ double* F6lims() {
	double limites[] = { -5.0, 10.0 };
	return limites;
}
//Schwefel 2.22; D>2; limits={-10,10}; f*=0; Unimodal; No separable.
__device__ double F7(double* x) {
	double suma = 0.0, multiplicacion = 1.0;
	for (int i = 0; i < D; i++) {
		suma += abs(x[i]);
		multiplicacion *= abs(x[i]);
	}
	return suma + multiplicacion;
}
__device__ double* F7lims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}
//Schwefel 1.2; D>2; limits={-10,10}; f*=0; Unimodal; No separable.
__device__ double F8(double* x) {
	double suma = 0.0, suma2 = 0.0;
	for (int i = 0; i < D; i++) {
		suma2 = 0.0;
		for (int j = 0; j < i; j++) {
			suma2 += x[j];
		}
		suma += pow(suma2, 2.0);
	}
	return suma;
}
__device__ double* F8lims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}
//Rosenbrock; D>2; limites={-30,30}; f*=0; Unimodal; No separable.
__device__ double F9(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D - 1; i++)
		sum += 100.0 * pow(x[i + 1] - pow(x[i], 2.0), 2.0) + pow(x[i] - 1.0, 2.0);
	return sum;
}
__device__ double* F9lims() {
	double limites[] = { -5.0, 10.0 };
	return limites;
}
//Dixon-Price; D>2; limites={-10,10}; f*=0; Unimodal; No separable.
__device__ double F10(double* x) {
	double sum = 0.0;
	for (int i = 1; i < D; i++)
		sum += (double(i) + 1.0) * pow(2 * pow(x[i], 2.0) - x[i - 1], 2.0);
	return sum + pow(x[0] - 1.0, 2.0);
}
__device__ double* F10lims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}

//Rastrigin; D>2; limites={-5.12,5.12}; f*=0; Multimodal; Separable.
__device__ double F11(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D; i++)
		sum += pow(x[i], 2.0) - 10.0 * cos(2.0 * PI * x[i]);
	return sum + 10.0 * D;
}
__device__ double* F11lims() {
	double limites[] = { -5.12, 5.12 };
	return limites;
}
//Schwefel; D>2; limits={-500,500}; f*=-418.9829*D; Multimodal; Separable.
__device__ double F12(double* x) {
	double f = 0.0;
	for (int i = 0; i < D; i++)
		f -= x[i] * sin(sqrt(fabs(x[i])));
	f += 418.9829 * D;
	return f;
}
__device__ double* F12lims() {
	double limites[] = { -500.0, 500.0 };
	return limites;
}
//Griewank; D>2; limits={-600,600}; f*=0; Multimodal; No separable.
__device__ double F13(double* x) {
	double sum = 0.0, prod = 1.0;
	for (int i = 0; i < D; i++) {
		sum += pow(x[i], 2.0) / 4000.0;
		prod = prod * cos(x[i] / sqrt(i + 1.0));
	}
	return sum - prod + 1.0;
}
__device__ double* F13lims() {
	double limites[] = { -600.0, 600.0 };
	return limites;
}
//Ackley; D>2; limits={-32.768, 32.768}; f*=0; Multimodal; No separable.
__device__ double F14(double* x) {
	double sum1 = 0.0, sum2 = 0.0, a = 20.0, b = 0.2, c = 2 * PI;
	for (int i = 0; i < D; i++) {
		sum1 += pow(x[i], 2.0);
		sum2 += cos(c * x[i]);
	}
	return -a * exp(-b * sqrt(sum1 / D)) - exp(sum2 / D) + a + exp(1.0);
}
__device__ double* F14lims() {
	double limites[] = { -32.768, 32.768 };
	return limites;
}
//Penalized1; D>2; limits={-50,50}; f*=0; Multimodal; No separable.
__device__ double F15(double* x) {
	double sum1 = 0.0, sum2 = 0.0;
	for (int i = 0; i < D - 1; i++) {
		sum1 += pow(x[i] - 1.0, 2.0) * (1.0 + 10.0 * pow(sin(PI * x[i + 1]), 2.0));
	}
	sum1 += 10.0 * pow(sin(PI * x[0]), 2.0) + pow(x[-1] - 1.0, 2.0);
	sum1 = (sum1 * PI) / D;
	for (int i = 0; i < D; i++) {
		if (x[i] > 10.0)
			sum2 += 100.0 * pow(x[i] - 10.0, 4.0);
		else if (x[i] < -10.0)
			sum2 += 100.0 * pow(-x[i] - 10.0, 4.0);
	}
	return sum1 + sum2;
}
__device__ double* F15lims() {
	double limites[] = { -50.0, 50.0 };
	return limites;
}
//Penalized2; D>2; limits={-50,50}; f*=0; Multimodal; No separable.
__device__ double F16(double* x) {
	double sum1 = 0.0, sum2 = 0.0;
	for (int i = 0; i < D - 1; i++) {
		sum1 += pow(x[i] - 1.0, 2.0) * (1.0 + 10.0 * pow(sin(3.0 * PI * x[i + 1]), 2.0));
	}
	sum1 += pow(sin(PI * x[0]), 2.0) + pow(x[-1] - 1.0, 2.0) * (1.0 + pow(sin(2.0 * PI * x[-1]), 2.0));

	for (int i = 0; i < D; i++) {
		if (x[i] > 5.0)
			sum2 += 100.0 * pow(x[i] - 10.0, 4.0);
		else if (x[i] < -5.0)
			sum2 += 100.0 * pow(-x[i] - 10.0, 4.0);
	}
	return sum1 + sum2;
}
__device__ double* F16lims() {
	double limites[] = { -50.0, 50.0 };
	return limites;
}
//Levy; D>2; limits={-10,10}; f*=0.
__device__ double F17(double* x) {
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
__device__ double* F17lims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}
//Perm; D>2; limits={-D,D}; f*=0.
__device__ double F18(double* x) {
	double outer = 0.0, inner;
	for (int i = 0; i < D; i++) {
		inner = 0.0;
		for (int j = 0; j < D; j++)
			inner += (double(j) + 1.0 + 10.0) * (pow(x[j], double(i) + 1.0) - pow(1.0 / (double(j) + 1.0), double(i) + 1.0));
		outer += pow(inner, 2.0);
	}
	return outer;
}
__device__ double* F18lims() {
	double limites[] = { -double(D), double(D) };
	return limites;
}
//Hyperellipsoid; D>2; limits={-65.53,65.53}; f*=0.
__device__ double F19(double* x) {
	double outer = 0.0, inner;
	for (int i = 0; i < D; i++) {
		inner = 0.0;
		for (int j = 0; j < D; j++)
			inner += pow(x[j], 2.0);
		outer += inner;
	}
	return outer;
}
__device__ double* F19lims() {
	double limites[] = { -65.536, 65.536 };
	return limites;
}
//Sum Power; D>2; limits={-1,1}, f*=0.
__device__ double F20(double* x) {
	double sum = 0.0;
	for (int i = 0; i < D; i++)
		sum += pow(fabs(x[i]), double(i) + 2.0);
	return sum;
}
__device__ double* F20lims() {
	double limites[] = { -1.0, 1.0 };
	return limites;
}

//Michalewicz2; D=2; limits={0,PI}; f*=-1.8013; Multimodal; Separable.
__device__ double F21(double* x) {
	double suma = 0.0, m = 10.0;
	suma = sin(x[0]) * pow(sin(pow(x[0], 2.0) / PI), 2.0 * m) + sin(x[1]) * pow(sin(pow(x[1], 2.0) / PI), 2.0 * m);
	return suma;
}
__device__ double* F21lims() {
	double limites[] = { 0.0, PI };
	return limites;
}
//Schaffer; D=2; limits={-100,100}; f*=0; Multimodal; No separable.
__device__ double F22(double* x) {
	double fact1 = 0.0, fact2 = 0.0;
	fact1 = pow(sin(pow(x[0], 2.0) - pow(x[1], 2.0)), 2.0) - 0.5;
	fact2 = pow(1.0 + 0.001 * (pow(x[0], 2.0) + pow(x[1], 2.0)), 2.0);
	return 0.5 + fact1 / fact2;
}
__device__ double* F22lims() {
	double limites[] = { -100.0, 100.0 };
	return limites;
}
//Six Hump Camel Back; D=2; limits={-5,5}; f*=-1.0316; Multimodal; No separable.
__device__ double F23(double* x) {
	double suma = 0.0;
	suma = 4.0 * pow(x[0], 2.0) - 2.1 * pow(x[0], 4.0) + (1.0 / 3.0) * pow(x[0], 6.0) + x[0] * x[1] - 4.0 * pow(x[1], 2.0) + 4.0 * pow(x[1], 4.0);
	return suma;
}
__device__ double* F23lims() {
	double limites[] = { -5.0, 5.0 };
	return limites;
}
//Bohachevsky2; D=2; limits={-100,100}; f*=0; Multimodal; No separable.
__device__ double F24(double* x) {
	double suma = 0.0;
	suma = pow(x[0], 2.0) + 2.0 * pow(x[1], 2.0) - 0.3 * cos(3.0 * PI * x[0]) * (4.0 * PI * x[1]) + 0.3;
	return suma;
}
__device__ double* F24lims() {
	double limites[] = { -100.0, 100.0 };
	return limites;
}
//Bohachevsky3; D=2; limits={-100,100}; f*=0; Multimodal; No separable.
__device__ double F25(double* x) {
	double suma = 0.0;
	suma = pow(x[0], 2.0) + 2.0 * pow(x[1], 2.0) - 0.3 * cos(3.0 * PI * x[0] + 4.0 * PI * x[1]) + 0.3;
	return suma;
}
__device__ double* F25lims() {
	double limites[] = { -100.0, 100.0 };
	return limites;
}
//Shubert; D=2; limits={-10,10}; f*=-186.7309; Multimodal; No separable.
__device__ double F26(double* x) {
	double fact1 = 0.0, fact2 = 0.0;
	for (int i = 0; i < 5; i++) {
		fact1 += (i + 1.0) * cos((i + 2.0) * x[0] + (i + 1.0));
		fact2 += (i + 1.0) * cos((i + 2.0) * x[1] + (i + 1.0));
	}
	return fact1 * fact2;
}
__device__ double* F26lims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}
//Goldstein-Price; D=2; limits={-2,2}; f*=3.0; Multimodal; No separable.
__device__ double F27(double* x) {
	double fact1a = 0.0, fact1b = 0.0, fact1c = 0.0, fact2a = 0.0, fact2b = 0.0, fact2c = 0.0;
	fact1a = pow(x[0] + x[1] + 1.0, 2.0);
	fact1b = 19.0 - 14.0 * x[0] + 3.0 * pow(x[0], 2.0) - 14.0 * x[1] + 6.0 * x[0] * x[1] + 3.0 * pow(x[1], 2.0);
	fact1c = 1.0 + fact1a * fact1b;
	fact2a = pow(2.0 * x[0] - 3.0 * x[1], 2.0);
	fact2b = 18.0 - 32.0 * x[0] + 12.0 * pow(x[0], 2.0) + 48.0 * x[1] - 36.0 * x[0] * x[1] + 27.0 * pow(x[1], 2.0);
	fact2c = 30.0 + fact2a * fact2b;
	return fact1c * fact2c;
}
__device__ double* F27lims() {
	double limites[] = { -2.0, 2.0 };
	return limites;
}
//Langermann; D=2; limits={0,10}; f*=-1.4; Multimodal; No separable.
__device__ double F28(double* x) {
	double suma1 = 0.0, fact1 = 0.0, fact2 = 0.0, suma2 = 0.0;
	double a1[] = { 3.0, 5.0, 2.0, 1.0, 7.0 };
	double a2[] = { 5.0, 2.0, 1.0, 4.0, 9.0 };
	double c[] = { 1.0, 2.0, 5.0, 2.0, 3.0 };
	for (int i = 0; i < 5; i++) {
		suma1 = pow(x[0] - a1[i], 2.0) + pow(x[1] - a2[i], 2.0);
		fact1 = c[i] * exp(-(1.0 / PI) * suma1);
		fact2 = cos(PI * suma1);
		suma2 += fact1 * fact2;
	}
	return suma2;
}
__device__ double* F28lims() {
	double limites[] = { 0.0, 10.0 };
	return limites;
}

//Beale; D=2; limits={-4.5,4.5}; f*=0; Unimodal; No separable.
__device__ double F29(double* x) {
	double suma = pow((1.5 - x[0] * (1.0 - x[1])), 2.0) + pow((2.25 - x[0] * (1.0 - pow(x[1], 2.0))), 2.0) + pow((2.625 - x[0] * (1.0 - pow(x[1], 3.0))), 2.0);
	return suma;
}
__device__ double* F29lims() {
	double limites[] = { -4.5, 4.5 };
	return limites;
}
//Easom; D=2; limits={-100,100}; f*=-1.0; Unimodal; No separable. 
__device__ double F30(double* x) {
	double suma = -cos(x[0]) * cos(x[1]) * exp(-pow(x[0] - PI, 2.0) - pow(x[1] - PI, 2.0));
	return suma;
}
__device__ double* F30lims() {
	double limites[] = { -100.0, 100.0 };
	return limites;
}
//Matyas; D=2; limits={-10,10}; f*=0; Unimodal; No separable.
__device__ double F31(double* x) {
	double suma = 0.26 * (pow(x[0], 2.0) + pow(x[1], 2.0)) - 0.48 * x[0] * x[1];
	return suma;
}
__device__ double* F31lims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}
//Foxholes; D=2; limits = {-65.536, 65.536}; f*=0.9980038; Multimodal; Separable.
__device__ double F32(double* x) {
	double suma = 0.0, suma2 = 0.0;
	double a1[] = { -32.0, -16.0, 0.0, 16.0, 32.0, -32.0, -16.0, 0.0, 16.0, 32.0, -32.0, -16.0, 0.0, 16.0, 32.0, -32.0, -16.0, 0.0, 16.0, 32.0, -32.0, -16.0, 0.0, 16.0, 32.0 };
	double a2[] = { -32.0, -32.0, -32.0, -32.0, -32.0, -16.0, -16.0, -16.0, -16.0, -16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 16.0, 16.0, 16.0, 16.0, 32.0, 32.0, 32.0, 32.0, 32.0 };
	for (int j = 0; j < 25; j++) {
		suma2 = 1.0 / (j + 1 + pow(x[0] - a1[j], 6.0) + pow(x[1] - a2[j], 6.0));
		suma += suma2;
	}
	suma2 = (1.0 / 500.0) + suma;
	suma = pow(suma2, -1.0);
	return suma;
}
__device__ double* F32lims() {
	double limites[] = { -65.536, 65.536 };
	return limites;
}
//Bohachevsky1; D=2; limits={-100,100}; f*=0; Multimodal; Separable.
__device__ double F33(double* x) {
	double suma = 0.0;
	suma = pow(x[0], 2.0) + 2.0 * pow(x[1], 2.0) - 0.3 * cos(3.0 * PI * x[0]) - 0.4 * cos(4.0 * PI * x[1]) + 0.7;
	return suma;
}
__device__ double* F33lims() {
	double limites[] = { -100.0, 100.0 };
	return limites;
}
//Booth; D=2; limits={-10,10}; f*=0; Multimodal; Separable.
__device__ double F34(double* x) {
	double suma = 0.0;
	suma = pow(x[0] + 2.0 * x[1] - 7.0, 2.0) + pow(2.0 * x[0] + x[1] - 5, 2.0);
	return suma;
}
__device__ double* F34lims() {
	double limites[] = { -10.0, 10.0 };
	return limites;
}
//Holder; D=2; limits={-10,10}; f*=-19.2085; x*=(8.05502,9.66459); Multimodal; No separable.
__device__ double F35(double* x) {
	double suma = 0.0;
	suma = abs(sin(x[0]) * cos(x[1]) * exp(abs(1.0 - (sqrt(pow(x[0], 2.0) + pow(x[1], 2.0)) / PI))));
	return suma;
}
__device__ double* F35lims() {
	double limites[] = { -10.0, 10.0 };
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
__device__ llamadaaFunciones funciones[] = { F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F31, F32, F33, F34, F35 };
typedef double* (*llamadaaFunciones2)();
__device__ llamadaaFunciones2 limites[] = { F1lims, F2lims, F3lims, F4lims, F5lims, F6lims, F7lims, F8lims, F9lims, F10lims, F11lims, F12lims, F13lims, F14lims, F15lims, F16lims, F17lims, F18lims, F19lims, F20lims, F21lims, F22lims, F23lims, F24lims, F25lims, F26lims, F27lims, F28lims, F29lims, F30lims, F31lims, F32lims, F33lims, F34lims, F35lims };

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
