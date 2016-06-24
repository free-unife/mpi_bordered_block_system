#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "res.h"

#define BS 256
#define N 128
#define MDIM BS * BS
#define VDIM BS

int main ( int argc, char * argv[ ] ){
	
	int myid;
	int numProc;
	double start;
	int blockPerProc;
	int ii, jj, kk;
	double * A = NULL;
	double * B = NULL;
	double * C = NULL;
	double * D = NULL;
	double * X = NULL;
	double * Y = NULL;
	double * A_cap = NULL;
	double * sum_Di = NULL;
	double * As = NULL;
	double * x = NULL;
	double * xs = NULL;
	double * b = NULL;
	double * bs = NULL;
	double * xd = NULL;
	double * yd = NULL;
	double * y = NULL;
	double * d = NULL;
	double * sum_di = NULL;
	double * bx = NULL;
	double * b_cap = NULL;
	double * y_cap = NULL;
	double * result = NULL;
	double * btot = NULL;
	double * di = NULL;
	double * Di = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_size( MPI_COMM_WORLD, &numProc );
	MPI_Comm_rank( MPI_COMM_WORLD, &myid );
	blockPerProc = N / numProc;

	start = MPI_Wtime();

	/*	PUNTO 1
		Ogni processo alloca la memoria ad esso destinato
		Genero Ai, Bi, Ci
		Eseguo la decomposizione LU delle matrici Ai (nella medesima area di memoria).
		Si suppone la diagonale principale (diag(Ai)) == 1
		Il processore alloca memoria per le variabili, genera il vettore dei termini noti, As
		Attraverso la funzione Scatter ogni processo riceve una porzione dei termini noti
	*/

	A = ( double * )calloc( blockPerProc * MDIM , sizeof( double ));
	B = ( double * )calloc( blockPerProc * MDIM , sizeof( double ));
	C = ( double * )calloc( blockPerProc * MDIM , sizeof( double ));
	D = ( double * )calloc( blockPerProc * MDIM , sizeof( double ));
	X = ( double * )calloc( blockPerProc * MDIM , sizeof( double ));
	Y = ( double * )calloc( blockPerProc * MDIM , sizeof( double ));
	x = ( double * )calloc( blockPerProc * VDIM , sizeof( double ));
	y = ( double * )calloc( blockPerProc * VDIM , sizeof( double ));
	d = ( double * )calloc( blockPerProc * VDIM , sizeof( double ));
	xd = ( double * )calloc( blockPerProc * VDIM , sizeof( double ));
	yd = ( double * )calloc( blockPerProc * VDIM , sizeof( double ));
	bx = ( double * )calloc( blockPerProc * VDIM , sizeof( double ));
	xs = ( double * )calloc( VDIM , sizeof(double));
	b = ( double * )malloc( blockPerProc * VDIM * sizeof( double ));
	di = ( double * )malloc( VDIM *  sizeof( double ));
	Di = ( double * )malloc( MDIM * sizeof( double ));
	
	ii = blockPerProc * myid;
	jj = 0;
	
	while(( ii < blockPerProc * myid + blockPerProc ) && ( jj < blockPerProc )){
		
		genera_Ai( &A[jj * MDIM], BS, ii+1 );
		LU( &A[jj * MDIM], BS );
		genera_Bi( &B[jj * MDIM], BS, ii+1 );
		genera_Bi( &C[jj * MDIM], BS, ii+1 );
		++ii;
		++jj;
	} 

	if ( ! myid ){
		
		btot = ( double * )malloc( N * VDIM * sizeof( double ));	
		As = ( double * )calloc( MDIM , sizeof( double ));
		bs = ( double * )calloc( VDIM , sizeof( double ));
		A_cap = ( double * )calloc( MDIM , sizeof( double ));
		b_cap = ( double * )calloc( VDIM , sizeof( double ));
		y_cap = ( double * )calloc( MDIM , sizeof( double ));
		sum_Di = ( double * )malloc( MDIM * sizeof( double ));
		sum_di = ( double * )malloc( VDIM * sizeof( double ));
		result = ( double * )malloc( (N + 1) * VDIM * sizeof( double ));	

		for( ii = 0; ii < N ; ++ii ){
			for( jj = 0 ; jj < BS ; ++jj )
			btot[ii * VDIM + jj] = 1.0;
		}

		for( ii = 0; ii < BS; ++ii )
		bs[ii] = 1.0;

		genera_Ai( As, BS, N+1 );
	}

	MPI_Scatter( btot, blockPerProc * VDIM, MPI_DOUBLE, b, blockPerProc * VDIM, MPI_DOUBLE, 0, MPI_COMM_WORLD );

	
	/*	PUNTO 2
		Ogni blocco esegue il calcolo xd = Ai^-1 * bi ( sfruttando la fattorizzazione LU) 
		Calcolo i termini di = Ci * xd
	*/
 		
	for( ii = 0; ii < blockPerProc; ++ii ){

		elim_avanti( &A[ii * MDIM], BS, &b[ii * VDIM], 1, &yd[ii * VDIM] );
		sost_indietro( &A[ii * MDIM], BS, &yd[ii * VDIM], 1, &xd[ii * VDIM] );
		my_gemm( BS, 1, BS, &C[ii * MDIM], BS, 1, &xd[ii * VDIM], 1, &d[ii * VDIM], 1 );
	}


	/*	PUNTO 3
		Ogni processo esegue una sommatoria dei propri blocchi d ("reduce locale")
		Attraverso un operazione Reduce ottengo la somma di ogni blocco di all'interno di sum_di ( memorizzato in root )
		Calcolo b_cap = bs - sum_di
	*/

	for( jj = 0 ; jj < BS ; ++jj ){
			for( ii = 0; ii < blockPerProc; ++ii ){
			di[jj] += d[ii * VDIM + jj];
		}
	}

	MPI_Reduce(di, sum_di, VDIM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(!myid){
		
		for( jj = 0 ; jj < BS ; ++jj )
			b_cap[jj] = bs[jj] - sum_di[jj];
	}

	/*	PUNTO 4
		Calcolo Xi = Ai^-1 * Bi (n sistemi lineari, X contiene N soluzioni xj)
		Calcolo il termine Di = Ci * Xi
	*/

	for( ii = 0; ii < blockPerProc; ++ii ){
		
		elim_avanti( &A[ii * MDIM], BS, &B[ii * MDIM], BS, &Y[ii * MDIM] );
		sost_indietro( &A[ii * MDIM], BS, &Y[ii * MDIM], BS, &X[ii * MDIM]);
		
		my_gemm(BS, BS, BS, &C[ii * MDIM], BS, 1, &X[ii * MDIM], BS, &D[ii * MDIM], BS);
	} 	 


	/*	PUNTO 5
		Ogni processo esegue una sommatoria dei propri blocchi D ("reduce locale")
		Attraverso un operazione Reduce ottengo la somma di ogni blocco di all'interno di sum_Di ( memorizzato in root )
		Calcolo A_cap = As - sum_Di
	*/

	for( jj = 0; jj < BS; ++jj ){
		for ( kk = 0; kk < BS; ++kk ){
			for( ii = 0; ii < blockPerProc; ++ii )
			Di[jj * BS + kk] += D[ii * MDIM + jj * BS + kk];
		}
	}
	
	MPI_Reduce(Di, sum_Di, MDIM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(!myid){
		
		for( jj = 0; jj < BS; ++jj ){
			for ( kk = 0; kk < BS; ++kk )
			A_cap[jj * BS + kk] = As[jj * BS + kk] - sum_Di[jj * BS + kk];
		}


	/*	PUNTO 6
		Calcolo la fattorizzazione LU di A_cap
		Risolvo il sistema lineare A_cap * xs = b_cap per determinare xs
	*/ 

		LU( A_cap, BS );

		elim_avanti( A_cap, BS, b_cap, 1, y_cap );
		sost_indietro( A_cap, BS, y_cap, 1, xs );
	}


	/*	PUNTO 7
		La soluzione xs viene viene comunicato a tutti i processori tramite una Broadcast;
	*/

	MPI_Bcast( xs, VDIM, MPI_DOUBLE, 0, MPI_COMM_WORLD );


	/*	PUNTO 8
		Eseguo bxi = Bi * xs
		Trovo bxi = bi - bxi
		Risolvo N sistemi lineari Ai * xi = bxi
	*/

	for( ii = 0; ii < blockPerProc; ++ii ){
		
		my_gemm(BS, 1, BS, &B[ii * MDIM], BS, 1, xs, 1, &bx[ii * VDIM], 1);
		
		for( jj = 0 ; jj < BS ; ++jj )	
		bx[ii * VDIM + jj] = b[ii * VDIM + jj] - bx[ii * VDIM + jj];
		
		elim_avanti( &A[ii * MDIM], BS, &bx[ii * VDIM], 1, &y[ii * VDIM] );
		sost_indietro( &A[ii * MDIM], BS, &y[ii * VDIM], 1, &x[ii * VDIM]);
	}


	/*	PUNTO 9
		Le componenti xi vengono riunite in un unico vettore (result) tramite una comunicazione di tipo
		Gather. Unite assieme a xs ( in result(N) ), formano la soluzione al problema iniziale.
		Libero memoria.
	*/
	
	MPI_Gather(x, blockPerProc * VDIM, MPI_DOUBLE, result, blockPerProc * VDIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if( ! myid )
	memcpy(&result[N * VDIM], xs, VDIM * sizeof( double ));


#ifdef PRINT_DATA

	if( ! myid ){	
		for( ii = 0; ii < N; ++ii){
			printf("\n Ecco x[%d]\n", ii);
			print_mat(&result[ii * VDIM], VDIM, 1 );
		}
		printf("\n Ecco xs\n");
		print_mat(&result[ii * VDIM], VDIM, 1 );
	}

#endif

	if( ! myid ){

		printf( "PARALLEL_TIME: %7.3lf [s]\n", MPI_Wtime() - start );

		free( btot );
		free( As );
		free( bs );
		free( A_cap );
		free( b_cap );
		free( y_cap );
		free( sum_Di );
		free( sum_di );
		free( result );
	}

	free( A );
	free( B );
	free( C );
	free( D );
	free( X );
	free( Y );
	free( x );
	free( y );
	free( d );
	free( xd );
	free( yd );
	free( bx );
	free( xs );
	free( b );
	free( di );
	free( Di );
	
	MPI_Finalize();
	exit( EXIT_SUCCESS );
}
