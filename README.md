# hw3

Ex3.3: omp-scan.cpp

     Ex3.3.png:  plot of cpu time versus number of threads from omp-scan.cpp running on greene
     
Ex3.4: sor2D.cpp

     successive-overrelaxation (SOR) for u_xx + u_yy = -1 (dirichelt on 0 < x,y < 1)
     scales better than jacobi or gauss-seidel: residual converges to machine eps in ~8 N iterations
     Ex3.4a.png: plot of residual versus itersation number for N = 1028 grid points running on greene
     Ex3.4b.png: plot of cpu time versus number of threads for N = 1028 grid points running on greene 
