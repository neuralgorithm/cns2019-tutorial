#include <stdio.h>

#define TAU 20.0
#define V_LEAK -65.0
#define V_INIT (V_LEAK)
#define V_RESET (V_LEAK)
#define THETA -55.0
#define R 1.0
#define DT 1.0
#define T 1000.0
#define I_EXT 12.0
  
void loop ( void )
{
  double t = 0;
  double v [ 2 ] = { V_INIT, V_INIT - 10.0 };
  int spike [ 2 ] = { 0, 0 };

  while ( t < T ) {
    printf ( "%f %f %f\n", t, ( spike [ 0 ] ? 0.0 : v [ 0 ] ), ( spike [ 1 ] ? 0.0 : v [ 1 ] ) );
    double dv [ 2 ] = { 0.0, 0.0 };
    for ( int i = 0; i < 2; i++ ) {
      dv [ i ] = ( DT / TAU ) * ( - ( v [ i ] - V_LEAK ) + R * I_EXT );
    }
    for ( int i = 0; i < 2; i++ ) {
      spike [ i ] = ( v [ i ] > THETA ) ? 1 : 0;
      v [ i ] = ( v [ i ] > THETA ) ? V_RESET : v [ i ] + dv [ i ];
    }
    t = t + DT;
  }
}

int main ( void )
{
  loop ( );
  return 0;
}
