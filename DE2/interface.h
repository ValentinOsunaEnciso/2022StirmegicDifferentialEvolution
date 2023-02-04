/*Differential Evolution, version DE/best/1/bin; Valentin Osuna-Enciso, DIC/2020, UDG*/
//Interface entre principal.cpp y deBest.cu
#ifndef _INTERFACE_H
#define _INTERFACE_H
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

extern "C" void iniciaPoblacionyLimites(double* d_X, double* d_aleat, double* d_liminf, double* d_limsup, int funcion);
extern "C" void evaluaFitnessPoblacion(double* d_X, double* d_F, int funcion);
extern "C" void obtieneMinimoGlobal(double* d_F, double* d_fbest, double* d_xbest, double* d_X, int* d_ind);
extern "C" void generaEvaluaMutantes(double* d_aleat3, double* d_aleat, double* d_X, double* d_U, double* d_FU, double* d_xbest, float F, float Cr, int funcion, double* d_liminf, double* d_limsup);
extern "C" void comparaFitnessUyX(double* d_X, double* d_U, double* d_F, double* d_FU);
#endif

