
#include <omp.h>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include "reader.h"

#ifndef SIZE
#define SIZE 1024
#endif

double dot_product(double* b1, double* b2, int n);

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("usage: %s data_dir\n", argv[0]);
        return 1;
    }
    char filepath[256];
    strcpy(filepath, argv[1]);

    FILE *time; 
    struct timeval tv1,tv2; 
    struct timezone tz;
    char file_time[40];

    int thread=0;   

    printf("Matrix size: %d*%d\n", SIZE, SIZE);

    double alpha, beta, d1, d2, diff;

    double** a = read_matrix(SIZE, filepath);
    double* b = read_vector(SIZE, filepath);
    if (a == NULL || b == NULL)
    {
        return 1;
    }
    double* x = static_cast<double*>(malloc(SIZE * sizeof(double)));
    double* x_n = static_cast<double*>(malloc(SIZE * sizeof(double)));
    double* r = static_cast<double*>(malloc(SIZE * sizeof(double)));
    double* r_n = static_cast<double*>(malloc(SIZE * sizeof(double)));
    double* h = static_cast<double*>(malloc(SIZE * sizeof(double)));
    double* h_n = static_cast<double*>(malloc(SIZE * sizeof(double)));
    double* tmp = static_cast<double*>(malloc(SIZE * sizeof(double)));

    gettimeofday(&tv1, &tz);
//    StartTimer();

    // 1. Set x0
    for (int i = 0; i < SIZE; ++i)
    {
        x[i] = b[i] / a[i][i];
    }
    // паралелим там где два вложенных цикла
    #pragma omp parallel for private(i, j)
    {
        // 2. Calculate residual vector r0 = b - A * x0
        for (int i = 0; i < SIZE; ++i)
        {
            tmp[i] = 0.0;
            for (int j = 0; j < SIZE; ++j)
            {
                tmp[i] += a[i][j] * x[j];
            }
            r[i] = b[i] - tmp[i];
        }
    }
    

    // 3. Calculate direction vector h0 = r0
    memcpy(h, r, SIZE * sizeof(double));

    double norm;
    int it = 0;
    do
    {
        it++;   /

        //1 Calculate alpha
        d1 = dot_product(r, r, SIZE);

        for (int i = 0; i < SIZE; ++i)
        {
            tmp[i] = 0.0;
            for (int j = 0; j < SIZE; ++j)
            {
                tmp[i] += a[i][j] * h[j];
            }
        }

        d2 = dot_product(tmp, h, SIZE);
        alpha = d1 / d2;

        //2 Calculate r_n
        for (int i = 0; i < SIZE; ++i)
        {
            r_n[i] = r[i] - alpha * tmp[i];
        }

        //3 Calculate beta
        d2 = dot_product(r_n, r_n, SIZE);
        beta = d2 / d1;

        //4 Calculate h_n and x_n
        for (int i = 0; i < SIZE; ++i)
        {
            h_n[i] = r_n[i] + beta * h[i];
            x_n[i] = x[i] + alpha * h[i];
        }

        norm = fabs(x_n[0] - x[0]);

        // параллелим из-за количества функций
        #pragma omp parallel for reduction(max:norm)
        {
            for (int i = 0; i < SIZE; ++i)
            {
                diff = fabs(x_n[i] - x[i]);
                if (diff > norm)
                {
                    norm = diff;
                }
                x[i] = x_n[i];
                h[i] = h_n[i];
                r[i] = r_n[i];
            }
        }
        

    } while (norm > 1e-8 && it < 10000);

    gettimeofday(&tv2, &tz);
    const double elapsed = (tv2.tv_sec-tv1.tv_sec) + (tv2.tv_usec-tv1.tv_usec)*1e-6;
    
    sprintf(file_time,"./out/time_%d.dat",  SIZE);
    if ((time=fopen(file_time, "a"))==NULL) {
		printf("Error at opening of the file.\n");
		exit(1);
    }
    fprintf(time, "%d\t%.12le\t%d\t%.12le\n",thread, elapsed, it, norm);
    fclose(time);


    printf("Number of iterations(K): %d\n", it);
    printf("Total time: %10.8f s\n", elapsed);

    // Проверка: 
    double err = 0.0;
    for (int i = 0; i < SIZE; ++i)
    {
        if (fabs(x[i] - 1.0) > err)
        {
            err = fabs(x[i] - 1.0);
        }
    }
    printf("max|x - 1|: %.3e\n", err);
    if (SIZE <= 12)
    {
        printf("x =");
        for (int i = 0; i < SIZE; ++i)
        {
            printf(" %.10f", x[i]);
        }
        printf("\n");
    }

    for (int i = 0; i < SIZE; ++i)
    {
        free(a[i]);
    }
    free(a);
    free(b);
    free(x);
    free(x_n);
    free(r);
    free(r_n);
    free(h);
    free(h_n);
    free(tmp);

    return 0;
}
// тут функция часта испольуется
#pragma omp parallel
{
    double dot_product(double* b1, double* b2, int n)
    {
        double res = 0.0;
        for (int i = 0; i < n; ++i)
        {
            res += b1[i] * b2[i];
        }
        return res;
    }

}

