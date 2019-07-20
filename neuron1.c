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
  double v = V_INIT;
  int spike = 0;
  while ( t < T ) {
    printf ( "%f %f\n", t, (spike ? 0.0 : v ) );
    double dv = ( DT / TAU ) * ( - ( v - V_LEAK ) + R * I_EXT );
    spike = ( v > THETA ) ? 1 : 0;
    v = ( v > THETA ) ? V_RESET : v + dv;
    t = t + DT;
  }
}
 
int main ( void )
{
  loop ( );
  return 0;
}
