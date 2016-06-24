#include <stdio.h>

/* Genera Ai
 * A 	matrice da riempire
 * dim 	numero di righe (e di colonne)
 * c 	coefficiente
 */ 
void genera_Ai (double *A, int dim, double c){
int ii;
double a0 =  4.0 * c;
double a1 = -1.0 * c;	

	A[0] = a0; A[1] = a1;
	for (ii = 1; ii<dim-1; ii++){
		A[ii*dim + ii - 1 ] = a1;
		A[ii*dim + ii     ] = a0;
		A[ii*dim + ii + 1 ] = a1;
	}
	A[(dim-1)*dim + dim - 2 ] = a1; A[(dim-1)*dim + dim - 1 ] = a0;

}

/* Genera Bi
 * B 	matrice da riempire
 * dim 	numero di righe (e di colonne)
 * c 	coefficiente
 */ 
void genera_Bi (double *A, int dim, double c){
int ii;
double a0 =  6.0 * c;
double a1 = -1.0 * c;	

	A[0] = a0; A[1] = a1; A[2] = a1;
	A[dim] = a1; A[dim+1] = a0; A[dim+2] = a1; A[dim+3] = a1;
	for (ii = 2; ii<dim-2; ii++){
		A[ii*dim + ii - 2 ] = a1;
		A[ii*dim + ii - 1 ] = a1;
		A[ii*dim + ii     ] = a0;
		A[ii*dim + ii + 1 ] = a1;
		A[ii*dim + ii + 2 ] = a1;
	}
	A[(dim-2)*dim + dim - 4 ] = a1;
	A[(dim-2)*dim + dim - 3 ] = a1;
	A[(dim-2)*dim + dim - 2 ] = a0;
	A[(dim-2)*dim + dim - 1 ] = a1;

	A[(dim-1)*dim + dim - 3 ] = a1;
	A[(dim-1)*dim + dim - 2 ] = a1;
	A[(dim-1)*dim + dim - 1 ] = a0;

}



/* Stampa una matrice di r righe per c colonne
 */
void print_mat(double *A, int r, int c){
int ii = 0, jj = 0;

	for (ii = 0; ii < r; ii++){
		
		for (jj = 0; jj < c; jj++)
			printf("% 10.4f ", A[ii*c+jj]);

		printf("\n");
	}
}

/* 
 * Risolve s sistemi:
 * U*X = B
 *
 * U di dim m x m 
 * m
 * B di dim m x s
 * s
 * X di dim m x s
 * U triangolare superiore
 */
void sost_indietro(double *U, int m, double * B, int s, double *X){
int ii,jj,kk;	

	// last variables
	for(kk = 0; kk < s; kk++)
		X[(m-1)*s + kk] = B[(m-1)*s +kk] / U[m*m-1];


	for (ii = m-2; ii > -1; ii--){

		for(kk = 0; kk < s; kk++)
			X[ii*s+kk] = 0.0;

		for (jj = ii+1; jj < m; jj++)
			for (kk = 0; kk < s; kk++)
				X[ii*s+kk] += X[jj*s+kk] * U[ii*m+jj];
		
		for(kk = 0; kk < s; kk++)
			X[ii*s+kk] = (B[ii*s+kk] - X[ii*s+kk]) / U[ii*m+ii];	
	}

}

/* 
 * Risolve s sistemi:
 * L*X = B
 *
 * L di dim m x m 
 * m
 * B di dim m x s
 * s
 * X di dim m x s
 * 
 * L triangolare inferiore, si suppone 1 sulla diagonale
 */
void elim_avanti(double *L, int m, double *B, int s, double *X){
int ii,jj,kk;	

	// first variables
	for(kk = 0; kk < s; kk++)
		X[kk] = B[kk];


	for (ii = 1; ii < m; ii++){

		for(kk = 0; kk < s; kk++)
			X[ii*s+kk] = 0.0;

		for (jj = 0; jj < ii; jj++)
			for (kk = 0; kk < s; kk++)
				X[ii*s+kk] += X[jj*s+kk] * L[ii*m+jj];
		
		for(kk = 0; kk < s; kk++)
			X[ii*s+kk] = B[ii*s+kk] - X[ii*s+kk];	
	}

}
/* Implementazione della decomposizione LU senza pivoting
 * A e' la matrice da decomporre, di dimensione n x n
 * In A si ottiene U, tc:
 * A = L*U;
*/
void LU(double *A,int n){
int ii = 0, jj = 0, kk = 0;
double m = 0.0;

	// starting from row 0,
	// take the diagonal entry as pivot
	for ( ii = 0; ii < n; ii++ ){
		// update the rows below the pivot
		for ( jj = ii+1; jj < n; jj++ ){
			m = A[jj*n + ii]/A[ii*n + ii];
			// row(j) = row(j) - m*row(i)
			for (kk = ii+1; kk < n; kk++)
				A[jj*n + kk] -= m*A[ii*n + kk];
			A[jj*n + ii] = m;
		}	
	}
}

/**
 * General Matrix Multiplication
 *
 * Calcola C = C + alpha * A * B
 * m 	righe di A e C	
 * n 	colonne di B e C
 * k    colonne di A e righe di B
 * A	
 * lda	leading dimension di A
 * alpha 
 * B   
 * ldb	leading dimension di B
 * C
 * ldc	leading dimension di C
 */ 
void my_gemm( int m, int n, int k,
	      double *A, int lda, 
	      double alpha,
	      double *B, int ldb, 
	      double *C, int ldc ){
int ii,jj,kk;
double sum;
	for (ii = 0; ii < m; ii++){
		for (jj = 0; jj < n; jj++){
			sum = 0.0;
			for(kk = 0; kk < k; kk++){
				sum += A[ii*lda+kk] * B[kk*ldb+jj];
			}
			C[ii*ldc + jj] += alpha * sum;
		}
	}
}

