#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "res.h"

#define BS 256
#define N 128
#define MDIM BS * BS
#define VDIM BS

int main ( int argc, char * argv[] ){

	int ii, jj, kk;
	clock_t start;
	double * A = ( double * )calloc( N * MDIM , sizeof( double ));
	double * B = ( double * )calloc( N * MDIM , sizeof( double ));
	double * C = ( double * )calloc( N * MDIM , sizeof( double ));
	double * D = ( double * )calloc( N * MDIM , sizeof( double ));
	double * X = ( double * )calloc( N * MDIM , sizeof( double ));
	double * Y = ( double * )calloc( N * MDIM , sizeof( double ));
	double * A_cap = ( double * )calloc( MDIM , sizeof( double ));
	double * sum_Di = ( double * )calloc( MDIM , sizeof( double ));
	double * As = ( double * )calloc( MDIM , sizeof( double ));
	double * x = ( double * )calloc( N * VDIM , sizeof( double ));
	double * xs = ( double * )calloc( VDIM , sizeof(double));
	double * b = ( double * )calloc( N * VDIM , sizeof( double ));
	double * bs = ( double * )calloc( VDIM , sizeof(double));
	double * xd = ( double * )calloc( N * VDIM , sizeof( double ));
	double * yd = ( double * )calloc( N * VDIM , sizeof( double ));
	double * y = ( double * )calloc( N * VDIM , sizeof( double ));
	double * d = ( double * )calloc( N * VDIM , sizeof( double ));
	double * sum_di = ( double * )calloc( VDIM , sizeof( double ));
	double * bx = ( double * )calloc( N * VDIM , sizeof( double ));
	double * b_cap = ( double * )calloc( VDIM , sizeof( double ));
	double * y_cap = ( double * )calloc( VDIM , sizeof( double ));
	double * result = ( double * )malloc( (N + 1) * VDIM * sizeof( double ));	

	start = clock();

	/*	PUNTO 1
		Genero Ai, As, Bi, Ci, bi, bs
		Eseguo la decomposizione LU delle matrici Ai (nella medesima area di memoria).
		Si suppone la diagonale principale (diag(Ai)) == 1
	*/
	
	for( ii = 0 ; ii < N ; ++ii ){
		
		genera_Ai( &A[ii * MDIM], BS, ii+1 );
		LU( &A[ii * MDIM], BS );
		genera_Bi( &B[ii * MDIM], BS, ii+1 );
		genera_Bi( &C[ii * MDIM], BS, ii+1 );

		for( jj=0 ; jj < BS ; ++jj )
		b[ii * VDIM + jj] = 1.0;
	}
	
	genera_Ai( As, BS, N+1 );

	for( ii = 0; ii < BS; ++ii )
	bs[ii] = 1.0;
	
	
	/*	PUNTO 2
		Eseguo il calcolo xd = Ai^-1 * bi ( sfruttando la fattorizzazione LU )
		Calcolo i termini di = Ci * xd
	*/
	
	for( ii = 0 ; ii < N ; ++ii ){
		
		elim_avanti( &A[ii * MDIM], BS, &b[ii * VDIM], 1, &yd[ii * VDIM] );
		sost_indietro( &A[ii * MDIM], BS, &yd[ii * VDIM], 1, &xd[ii * VDIM]);
		my_gemm(BS, 1, BS, &C[ii * MDIM], BS, 1, &xd[ii * VDIM], 1, &d[ii * VDIM], 1);
	}

	
	/*	PUNTO 3
		Eseguo sum_di = somma 0:N-1 (di)
		Calcolo b_cap = bs - sum_di
	*/

	for( jj = 0 ; jj < BS ; ++jj ){
		
		for( ii = 0; ii < N; ++ii )
		sum_di[jj] += d[ii * VDIM + jj];
		
		b_cap[jj] = bs[jj] - sum_di[jj];
	}


	/*	PUNTO 4
		Calcolo Xi = Ai^-1 * Bi (n sistemi lineari, X contiene N soluzioni xj)
		Calcolo il termine Di = Ci * Xi
	*/

	for( ii = 0; ii < N; ++ii ){
		
		elim_avanti( &A[ii * MDIM], BS, &B[ii * MDIM], BS, &Y[ii * MDIM] );
		sost_indietro( &A[ii * MDIM], BS, &Y[ii * MDIM], BS, &X[ii * MDIM]);
		
		my_gemm(BS, BS, BS, &C[ii * MDIM], BS, 1, &X[ii * MDIM], BS, &D[ii * MDIM], BS);
	} 	 

	/*	PUNTO 5
		Eseguo sum_Di = somma 0:N-1 (Di)
		Calcolo A_cap = As - sum_Di
	*/

	for( jj = 0; jj < BS; ++jj ){
		for ( kk = 0; kk < BS; ++kk ){
			for( ii = 0; ii < N ; ++ii)
			sum_Di[jj * BS + kk] += D[ii * MDIM + jj * BS + kk];

			A_cap[jj * BS + kk] = As[jj * BS + kk] - sum_Di[jj * BS + kk];
		}
	}


	/*	PUNTO 6
		Calcolo la fattorizzazione LU di A_cap
		Risolvo il sistema lineare A_cap * xs = b_cap per determinare xs
	*/ 

	LU( A_cap, BS );

	elim_avanti( A_cap, BS, b_cap, 1, y_cap );
	sost_indietro( A_cap, BS, y_cap, 1, xs );


	/*	PUNTO 8
		Eseguo bxi = Bi * xs
		Trovo bxi = bi - bxi
		Risolvo N sistemi lineari Ai * xi = bxi
	*/

	for( ii = 0; ii < N ; ++ii ){
		
		my_gemm(BS, 1, BS, &B[ii * MDIM], BS, 1, xs, 1, &bx[ii * VDIM], 1);
		
		for( jj = 0 ; jj < BS ; ++jj )	
		bx[ii * VDIM + jj] = b[ii * VDIM + jj] - bx[ii * VDIM + jj];
		
		elim_avanti( &A[ii * MDIM], BS, &bx[ii * VDIM], 1, &y[ii * VDIM] );
		sost_indietro( &A[ii * MDIM], BS, &y[ii * VDIM], 1, &x[ii * VDIM]);
	}


	/*	PUNTO 9
		Compongo la soluzione finale in -> result
	*/

	memcpy(result, x, N * VDIM * sizeof( double ));
	memcpy(&result[N * VDIM], xs, VDIM * sizeof( double ));

#ifdef PRINT_DATA

	
	for( ii = 0; ii < N; ++ii){
		printf("\n Ecco x[%d]\n", ii);
		print_mat(&result[ii * VDIM], VDIM, 1 );
	}
	printf("\n Ecco xs\n");
	print_mat(&result[ii * VDIM], VDIM, 1 );
	

#endif

	printf( "SERIAL_TIME:   %7.3lf [s]\n", ( clock() - start ) / ( double ) CLOCKS_PER_SEC );
	

	free(A);
	free(B);
	free(C);
	free(D);
	free(X);
	free(Y);
	free(A_cap);
	free(sum_Di);
	free(As);
	free(x);
	free(xs);
	free(b);
	free(bs);
	free(xd);
	free(yd);
	free(y);
	free(d);
	free(sum_di);
	free(bx);
	free(b_cap); 
	free(y_cap);

	return 0;
}
