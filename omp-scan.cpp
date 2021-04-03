#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define T_MAX 100 // hard-wire maximum number of threads (for dimensioning)

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];              // adjust to match the homework specs
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) 
{
  int n_threads = omp_get_max_threads();
  printf("\n maximum number of threads = %d\n", n_threads);

  if (n_threads == 0) return;
  long inc = n/n_threads;

  if (n == 0) return;

  long i;
  #pragma omp parallel
  {
     #pragma omp for 
     for (long i = 0; i < n; i+=inc) // try to split loop among threads
     {
        prefix_sum[i] = A[i];    // no 'memory' of earlier sums
        // don't access A[k] for k > n-1 -- bombs
        for (long ii = 1; ii < inc && ii + i < n; ii++)
        {
          prefix_sum[i+ii] = prefix_sum[i+ii-1] + A[i+ii];
        }
     }
  }

  // serial code for scalar offsets ...
  
  long suminc[T_MAX];   // needed T_MAX here ...

  long k = 0;
  suminc[0] = 0;
    for (long i = inc; i < n; i+=inc)
  {
    k = k+1;
    suminc[k] = prefix_sum[i-1] + suminc[k-1]; // accumulate offsets to add in
  }

  #pragma omp parallel private(k)   // shouldn't share k -- not public
  {
     #pragma omp for schedule(static)  
     for (long i = 0; i < n; i+=inc)
     {
       k = i/inc; 
       for (long ii = 0; ii < inc && ii+i < n; ii++)
         {
           prefix_sum[i+ii] = prefix_sum[i+ii] + suminc[k];
         }
     }
  }
}

int main() {
  long N = 100000000;
  // long N = 24;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));

  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("\n sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  double time_par = omp_get_wtime() - tt;
  printf("\n parallel-scan   = %fs\n", time_par);

  // for (long i = 0; i < N; i++) 
  // printf("i, A[i], B0[i], B1[i] = %ld, %ld, %ld, %ld \n",i, A[i], B0[i], B1[i]);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("\n error = %ld\n\n", err);

  printf("N_thread Time \n");
  printf("   %ld   %f \n", omp_get_max_threads(), time_par);
  free(A);
  free(B0);
  free(B1);
  return 0;
}
