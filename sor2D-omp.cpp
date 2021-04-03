// g++ -std=c++11 -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h> 
#include "utils.h"

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}

int main(int argc, char** argv) 
{

    long n = 1028;           // number of mesh points in 1-D ...
    long NIT = 8*n;       // number of sor iterations ...
    double rtol = 1.0e-13; // error tolerance for residual ...

    double* U_new = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double)); 
    double* U_cur = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double)); 
    double* U_old = (double*) aligned_malloc((n+2) * (n+2) * sizeof(double)); 

    // Initialize matrices ...
    // Dirichlet BC for j,k = 0,(n+1) ["ghost points"] ...
    // interior points: 1 <= j,k <= n ...

    // iterative scheme has two time levels ...

    for (long i = 0; i < (n+2)*(n+2); i++) U_cur[i] = 0.0;
    for (long i = 0; i < (n+2)*(n+2); i++) U_old[i] = 0.0;

    double h = 1.0/(n+1);   // mesh size ...

    // Garabedian's "sor" PDE is damped wave eqn: "a u_tt + e u_t = RHS" ...
    // RHS = nabla^2 u + 1, delta t = h works for CFL
    // Note: delta t = h*h for Jacobi, Gauss-Siedel: "u_t = RHS"

    double a = 2.0;  // satisfies CFL ...
    double e = 12.0; // adjust empirically for best decay ...
    
    // resulting parameters for iteration scheme ...

    double a1 = 1.0/(a + e*h);
    double a2 = (2.0*a + e*h)/(a + e*h);
    double a3 = a/(a + e*h);
 
    Timer t;
    t.tic();

    printf("\n n = %ld, number of threads = %d \n\n",n, omp_get_max_threads());

    printf(" it    rmax\n");
    for (long it = 1; it <= NIT; it++) 
    {
       double rmax = 0.0;
#pragma omp parallel for
       for (long k = 1; k <= n; k++) {
           for (long j = 1; j <= n; j++) {
             double u_jk   = U_cur[j     + k    *(n+2)];
             double u_jpk  = U_cur[(j+1) + k    *(n+2)];
             double u_jmk  = U_cur[(j-1) + k    *(n+2)];
             double u_jkp  = U_cur[j     + (k+1)*(n+2)];
             double u_jkm  = U_cur[j     + (k-1)*(n+2)];
             double um_jk  = U_old[j     + k    *(n+2)];
             
             // R_jk = right hand side*h*h ...

             double R_jk = (u_jpk + u_jmk + u_jkp + u_jkm - 4.0*u_jk + h*h) ;
             rmax = fmax(fabs(R_jk),rmax);

             // iteration scheme: un_jk = U_new, u_jk = U_cur, um_jk = U_old  ...

             double un_jk = a1*R_jk + a2*u_jk - a3*um_jk;

             U_new[j+k*(n+2)] = un_jk;
          }
       }

       if (it == 1)          printf("%ld   %13.5e \n",it, rmax);
       if (it%(NIT/30) == 0) printf("%ld   %13.5e \n",it, rmax);

       // update by Georg's pointer switch ...

#pragma omp parallel for
       for (long k = 1; k <= n; k++) {
         for (long j = 1; j <= n; j++) {
             double temp = U_cur[j+k*(n+2)];
             U_cur[j+k*(n+2)] = U_new[j+k*(n+2)];
             U_old[j+k*(n+2)] = temp;
          }
       }

       // double* temp = U_cur;
       // U_cur = U_new;
       // U_old = temp;

       // if (rmax <= rtol) 
       // {
       //   printf("%ld   %13.5e \n",it, rmax);
       //   break;
       // }
    }
    double time = t.toc();
    printf("\n cpu time number of threads\n");
    printf("%f      %d\n",time, omp_get_max_threads());

    aligned_free(U_new);
    aligned_free(U_cur);
    aligned_free(U_old);

  return 0;
}
