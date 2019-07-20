#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <SFMT.h>
#include <mpi.h>

#define N 4000
#define N_EXC 3200
#define N_INH ( ( N ) - ( N_EXC ) )

#define T 1000.
#define DT 1.

#define TAU_M 20.
#define TAU_GE 5.
#define TAU_GI 10.

#define V_LEAK -49.
#define V_INIT -60.
#define V_RESET -60.
#define THETA -50.

#define G_EXC 1.62
#define G_INH -9.
#define P 0.02

extern void sfmt_init_gen_rand ( sfmt_t * sfmt, uint32_t seed );
extern double sfmt_genrand_real2 ( sfmt_t * sfmt );
extern void timer_start ( void );
extern double timer_elapsed ( void );

static double v [ N ], ge [ N ], gi [ N ];
static int *w_exc, *w_inh, *spike;

static FILE *file_spike;

void initialize ( const int mpi_size, const int mpi_rank )
{
  // PRNG
  sfmt_t rng;
  sfmt_init_gen_rand ( &rng, 23 );

  // Output file
  if ( mpi_rank == 0 ) { file_spike = fopen ( "spike.dat", "w" ); }

  const int n_each = ( N + mpi_size - 1 ) / mpi_size;
  spike = ( int * ) malloc ( n_each * mpi_size * sizeof ( int ) );
  
  // Cell parameters
  for ( int i = 0; i < N; i++ ) {
    v [ i ] = V_INIT + 10. * sfmt_genrand_real2 ( &rng );
    ge [ i ] = 0.;
    gi [ i ] = 0.;
    spike [ i ] = 0;
  }

  // Synaptic weights
  w_exc = (int *) malloc ( N * N * sizeof ( int ) );
  w_inh = (int *) malloc ( N * N * sizeof ( int ) );
  for ( int i = 0; i < N; i++ ) {
    // From excitatory neurons to other neurons
    for ( int j = 0; j < N_EXC; j++ ) {
      w_exc [ j + N * i ] = ( sfmt_genrand_real2 ( &rng ) < P ) ? 1 : 0;
    }
    // From inhibitory neurons to other neurons
    for ( int j = N_EXC; j < N_EXC + N_INH; j++ ) {
      w_inh [ j + N * i ] = ( sfmt_genrand_real2 ( &rng ) < P ) ? 1 : 0;
    }
  }
}

void finalize ( const int mpi_rank )
{
  if ( mpi_rank == 0 ) { fclose ( file_spike ); }
  free ( spike );
  free ( w_exc );
  free ( w_inh );
}

void calculateSynapse ( const int n, const int offset )
{
  int i = n + offset;

  if ( i < N ) {
    double r = 0.;
    for ( int j = 0; j < N_EXC; j++ ){
      r += w_exc [ j + N * i ] * spike [ j ];
    }
    ge [ i ] += DT * ( G_EXC * r - ge [ i ] ) / TAU_GE;

    r = 0.;
    for ( int j = N_EXC; j < N_EXC + N_INH; j++ ){
      r += w_inh [ j + N * i ] * spike [ j ];
    }
    gi [ i ] += DT * ( G_INH * r - gi [ i ] ) / TAU_GI;
  }
}

void updateMembranePotential ( const int n, const int offset, int spike_local [] )
{
  int i = n + offset; 

  if ( i < N ) {
    double dv = DT * ( - ( v [ i ] - V_LEAK ) + ge [ i ] + gi [ i ] ) / TAU_M;
    spike_local [ n ] = ( v [ i ] > THETA ) ? 1 : 0;
    v [ i ] = ( v [ i ] > THETA ) ? V_RESET : v [ i ] + dv;
  }
}

void outputSpike ( const double t )
{
  for ( int i = 0; i < N; i++ ) {
    if ( spike [ i ] ) { fprintf ( file_spike, "%f %d\n", t, i ); }
  }
}

void loop ( const int mpi_size, const int mpi_rank )
{
  const int n_each = ( N + mpi_size - 1 ) / mpi_size;
  //const int n_each = N / mpi_size;
  int spike_local [ n_each ];
  
  double t = 0.;
  timer_start ();
  while ( t < T ) {
    for ( int n = 0; n < n_each; n++ ) {
      calculateSynapse ( n, n_each * mpi_rank );
      updateMembranePotential ( n, n_each * mpi_rank, spike_local );
    }

    MPI_Allgather ( spike_local, n_each, MPI_INT, spike, n_each, MPI_INT, MPI_COMM_WORLD );

    if ( mpi_rank == 0 ) { outputSpike ( t ); }
    t = t + DT;
  }
  double elapsedTime = timer_elapsed ();
  if ( mpi_rank == 0 ) { printf ( "Elapsed time = %f sec.\n", elapsedTime); }
}

int main ( int argc, char *argv[] )
{
  MPI_Init ( &argc, &argv );

  int mpi_size, mpi_rank;
  MPI_Comm_size ( MPI_COMM_WORLD, &mpi_size );
  MPI_Comm_rank ( MPI_COMM_WORLD, &mpi_rank );
  
  initialize ( mpi_size, mpi_rank );
  loop ( mpi_size, mpi_rank );
  finalize ( mpi_rank );

  MPI_Finalize ();
  
  return 0;
}
