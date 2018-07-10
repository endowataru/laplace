/*************************************************
 * Laplace OpenMP C Version
 *
 * Temperature is initially 0.0
 * Boundaries are as follows:
 *
 *      0         T         0
 *   0  +-------------------+  0
 *      |                   |
 *      |                   |
 *      |                   |
 *   T  |                   |  T
 *      |                   |
 *      |                   |
 *      |                   |
 *   0  +-------------------+ 100
 *      0         T        100
 *
 *  John Urbanic, PSC 2014
 *
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// size of plate
#define COLUMNS       672
#define ROWS          672        // this is a "global" row count

// Use 10752 (16 times bigger) for large challenge problem
// All chosen to be easily divisible by Bridges' 28 cores per node

// largest permitted change in temp (This value takes 3264 steps)
#define MAX_TEMP_ERROR 0.01

double Temperature[ROWS+2][COLUMNS+2];      // temperature grid
double Temperature_last[ROWS+2][COLUMNS+2]; // temperature grid from last iteration

//   helper routines
void initialize();
void track_progress(int iter, double dt);


int main(int argc, char *argv[]) {

    int i, j;                                            // grid indexes
    int max_iterations;                                  // number of iterations
    int iteration=1;                                     // current iteration
    double dt=100;                                       // largest change in t
    double start_time, stop_time, elapsed_time;          // timers

#pragma omp parallel
#pragma omp master
    printf("Running on %d OpenMP threads\n\n", omp_get_num_threads());

    //    printf("Maximum iterations [100-4000]?\n");
    //    scanf("%d", &max_iterations);
    max_iterations = 4000;
    printf("Maximum iterations = %d\n", max_iterations);

    start_time = omp_get_wtime();

    initialize();                   // initialize Temp_last including boundary conditions

    // do until error is minimal or until max steps
    while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {

        // main calculation: average my four neighbors
        #pragma omp parallel for private(i,j)
        for(i = 1; i <= ROWS; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }
        
        dt = 0.0; // reset largest temperature change

        // copy grid to old grid for next iteration and find latest dt
        #pragma omp parallel for reduction(max:dt) private(i,j)
        for(i = 1; i <= ROWS; i++){
            for(j = 1; j <= COLUMNS; j++){
	      dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
	      Temperature_last[i][j] = Temperature[i][j];
            }
        }

        // periodically print test values
        if((iteration % 100) == 0) {
	  track_progress(iteration, dt);
        }

	iteration++;
    }

    stop_time = omp_get_wtime();
    elapsed_time = stop_time - start_time;
    
    printf("\nMax error at iteration %d was %20.15g\n", iteration-1, dt);
    printf("Total time was %f seconds.\n", elapsed_time);
}


// initialize plate and boundary conditions
// Temp_last is used to to start first iteration
void initialize(){

    int i,j;

    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

    // set left side to 0 and right to a linear increase
    for(i = 0; i <= ROWS+1; i++) {
        Temperature_last[i][0] = 0.0;
        Temperature_last[i][COLUMNS+1] = (100.0/ROWS)*i;
    }
    
    // set top to 0 and bottom to linear increase
    for(j = 0; j <= COLUMNS+1; j++) {
        Temperature_last[0][j] = 0.0;
        Temperature_last[ROWS+1][j] = (100.0/COLUMNS)*j;
    }
}


// print diagonal in bottom right corner where most action is
void track_progress(int iteration, double dt) {

    int i;

    printf("---- Iteration %d, dt = %f ----\n", iteration, dt);
    for(i = ROWS-5; i <= ROWS-3; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, Temperature[i][i]);
    }
    printf("\n");
}
