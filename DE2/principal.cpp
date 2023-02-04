/*Differential Evolution, version DE/best/1/bin; Valentin Osuna-Enciso, DIC/2020, UDG*/
//Archivos: principal.cpp; interface.h; deBest.cu
// CANONICAL DE, EXPERIMENTO 2:
// D=100; Np=512; 5000 iteraciones; 30 corridas; 12 funciones. 
// Goal: Execution time.
#include "interface.h"

int main(void) {
	double* d_aleat, * d_X, * d_F, * d_FU, * d_limsup, * d_liminf, * d_xbest, * d_fbest, * d_U, * d_aleat3;//Apuntador a datos en device
	dim3 blocksXgridX(Np, 1), hilosXblockX(D, 1);  		//Bloques en grid;Hilos por bloque				
	size_t tamanoDoble = sizeof(double), tamanoInt = sizeof(int);
	int iteracion, corrida, funcion = 0, * h_ind, * d_ind, i, j, cont;
	curandGenerator_t gen;								//Crea variable generador 
	double* h_Data, * h_F, * h_fbest, * h_xbest, * h_aleat3;//Apuntador a datos en host
	FILE* fPtr;											//Apuntador a archivo de texto
	fopen_s(&fPtr, "canonic5.txt", "a");				//Crear y abrir archivo en modo append

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

	float tiempo;
	//CREAR MATRIZ DE NUMEROS ALEATORIOS: 	
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);	//Crea generador	
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));	//Pone semilla en generador		
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);

	for (funcion = 0; funcion < 1; funcion++) {
		for (corrida = 0; corrida < 30; corrida++) {			
			cudaEventRecord(start, 0);
			curandGenerateUniformDouble(gen, d_aleat, Np * D);		//Genera aleatorios en device
			//INICIALIZAR LIMITE INFERIOR, LIMITE SUPERIOR, POBLACION:
			iniciaPoblacionyLimites(d_X, d_aleat, d_liminf, d_limsup, funcion);

			//EVALUAR POBLACIÓN, OBTENER fbest Y xbest EN ITERACION 0:
			evaluaFitnessPoblacion(d_X, d_F, funcion);
			obtieneMinimoGlobal(d_F, d_fbest, d_xbest, d_X, d_ind);

			for (iteracion = 0; iteracion < 5000; iteracion++) {

				//GENERA Y EVALUA MATRIZ DE VECTORES MUTADOS:
				curandGenerateUniformDouble(gen, d_aleat3, Np * 3);	//Genera aleatorios en device
				curandGenerateUniformDouble(gen, d_aleat, Np * D);	//Genera aleatorios en device
				generaEvaluaMutantes(d_aleat3, d_aleat, d_X, d_U, d_FU, d_xbest, 0.8, 0.3, funcion, d_liminf, d_limsup);

				//COMPARA FU Y F; OBTIENE fbest Y xbest:
				comparaFitnessUyX(d_X, d_U, d_F, d_FU);
				obtieneMinimoGlobal(d_F, d_fbest, d_xbest, d_X, d_ind);

				//MUESTRA MEJOR FITNESS EN CADA ITERACION:
				//cudaMemcpy(h_fbest, d_fbest, tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host	
				//fprintf(fPtr, "%f\n", h_fbest[0]);
				//printf("Iteracion %d: Mejor fitness: %.4e \n \n", iteracion, h_fbest[0]);
			}
			//fclose(fPtr);
			//cudaMemcpy(h_xbest, d_xbest, D * tamanoDoble, cudaMemcpyDeviceToHost); //Copia info a host	
			//for (i = 0; i < D; i++) printf("%f   ", h_xbest[i]);
			//printf("ya...");
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaDeviceSynchronize();
			cudaEventElapsedTime(&tiempo, start, stop);
			fprintf(fPtr, "%f, ", tiempo / 1000);
			printf("Funcion: %d; Corrida: %d; Tiempo: %f S\n", funcion, corrida, tiempo / 1000);
		}
		fprintf(fPtr, "\n");
	}
	fclose(fPtr);
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