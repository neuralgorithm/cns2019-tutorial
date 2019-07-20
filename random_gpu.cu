#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <SFMT.h>

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

#define BLOCK_SIZE 32

extern "C" { void sfmt_init_gen_rand ( sfmt_t * sfmt, uint32_t seed ); }
extern "C" { double sfmt_genrand_real2 ( sfmt_t * sfmt ); }
extern "C" { void timer_start ( void ); }
extern "C" { double timer_elapsed ( void ); }

static double v [ N ], ge [ N ], gi [ N ];
static int *w_exc, *w_inh, spike [ N ];
static double *d_v, *d_ge, *d_gi;
static int *d_w_exc, *d_w_inh, *d_spike;;

static FILE *file_spike;

void initialize ( void )
{
  // PRNG
  sfmt_t rng;
  sfmt_init_gen_rand ( &rng, 23 );

  // Output file
  file_spike = fopen ( "spike.dat", "w" );

  // Cell parameters
  for ( int i = 0; i < N; i++ ) {
    v [ i ] = V_INIT + 10. * sfmt_genrand_real2 ( &rng );
    ge [ i ] = 0.;
    gi [ i ] = 0.;
    spike [ i ] = 0;
  }

  cudaMalloc ( &d_v, N * sizeof ( double ) );
  cudaMalloc ( &d_ge, N * sizeof ( double ) );
  cudaMalloc ( &d_gi, N * sizeof ( double ) );
  cudaMalloc ( &d_spike, N * sizeof ( int ) );

  cudaMemcpy ( d_v, v, N * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( d_ge, ge, N * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( d_gi, gi, N * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( d_spike, spike, N * sizeof ( int ), cudaMemcpyHostToDevice );

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
  cudaMalloc ( &d_w_exc, N * N * sizeof ( int ) );
  cudaMalloc ( &d_w_inh, N * N * sizeof ( int ) );
  cudaMemcpy ( d_w_exc, w_exc, N * N * sizeof ( int ), cudaMemcpyHostToDevice );
  cudaMemcpy ( d_w_inh, w_inh, N * N * sizeof ( int ), cudaMemcpyHostToDevice );
}

void finalize ( void )
{
  fclose ( file_spike );
  free ( w_exc );
  free ( w_inh );
  cudaFree ( d_v );
  cudaFree ( d_ge );
  cudaFree ( d_gi );
  cudaFree ( d_spike );
  cudaFree ( d_w_exc );
  cudaFree ( d_w_inh );
}

__device__ void calculateSynapse ( int i, double *ge, double *gi, int *w_exc, int *w_inh, int *spike )
{
  if ( i < N )
  {
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

__device__ void updateMembranePotential ( int i, double *v, double *ge, double *gi, int *spike )
{
  if ( i < N )
  {
    double dv = DT * ( - ( v [ i ] - V_LEAK ) + ge [ i ] + gi [ i ] ) / TAU_M;
    spike [ i ] = ( v [ i ] > THETA ) ? 1 : 0;
    v [ i ] = ( v [ i ] > THETA ) ? V_RESET : v [ i ] + dv;
  }
}

__global__ void kernel ( double *v, double *ge, double *gi, int *spike, int *w_exc, int *w_inh )
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ( i < N ) {
    calculateSynapse ( i, ge, gi, w_exc, w_inh, spike );
    updateMembranePotential ( i, v, ge, gi, spike );
  }
}

void outputSpike ( const double t )
{
  for ( int i = 0; i < N; i++ ) {
    if ( spike [ i ] ) { fprintf ( file_spike, "%f %d\n", t, i ); }
  }
}

void loop ( void )
{
  double t = 0.;
  timer_start ();

  int gridsize = ( N + BLOCK_SIZE - N % BLOCK_SIZE ) / BLOCK_SIZE;
  while ( t < T ) {
    kernel <<< gridsize, BLOCK_SIZE >>> ( d_v, d_ge, d_gi, d_spike, d_w_exc, d_w_inh );
    cudaMemcpy ( spike, d_spike, N * sizeof ( int ), cudaMemcpyDeviceToHost );
    outputSpike ( t );
    t = t + DT;
  }
  double elapsedTime = timer_elapsed ();
  printf ( "Elapsed time = %f sec.\n", elapsedTime);
}

int main ( void )
{
  initialize ();
  loop ();
  finalize ();

  return 0;
}
