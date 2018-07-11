/****************************************************************
 * Laplace MPI C Version                                         
 *                                                               
 * T is initially 0.0                                            
 * Boundaries are as follows                                     
 *                                                               
 *                T                      4 sub-grids            
 *   0  +-------------------+  0    +-------------------+       
 *      |                   |       |                   |           
 *      |                   |       |-------------------|         
 *      |                   |       |                   |      
 *   T  |                   |  T    |-------------------|             
 *      |                   |       |                   |     
 *      |                   |       |-------------------|            
 *      |                   |       |                   |   
 *   0  +-------------------+ 100   +-------------------+         
 *      0         T       100                                    
 *                                                                 
 * Each PE only has a local subgrid.
 * Each PE works on a sub grid and then sends         
 * its boundaries to neighbors.
 *                                                                 
 *  John Urbanic, PSC 2014
 *
 *******************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define COLUMNS       672
#define ROWS_GLOBAL   672        // this is a "global" row count

// Use 10752 (16 times bigger) for large challenge problem
// All chosen to be easily divisible by Bridges' 28 cores per node

#define NPES            4        // number of processors
#define ROWS (ROWS_GLOBAL/NPES)  // number of real local rows

// communication tags
#define DOWN     100
#define UP       101   

// largest permitted change in temp (This value takes 3264 steps)
#define MAX_TEMP_ERROR 0.01
#define MAX_ITERATIONS 4000
#define N_BUF_TRACK    4*MAX_ITERATIONS/100

double Temperature[ROWS+2][COLUMNS+2];
double Temperature_last[ROWS+2][COLUMNS+2];

void initialize(int npes, int my_PE_num);
void track_progress(int const iter, double* buf_track);
void output(int my_pe, int iteration);


int main(int argc, char *argv[]) {

    int i, j;
    int iteration=1;
    double dt;
    double start_time, stop_time, elapsed_time;

    int        npes;                // number of PEs
    int        my_PE_num;           // my PE number
    double     dt_global=100;       // delta t across all PEs
    MPI_Status status;              // status returned by MPI calls

    double buf_track[N_BUF_TRACK];
    int i_buf;

    // the usual MPI startup routines
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);


    if (my_PE_num == 0)
      {
	printf("Running on %d MPI processes\n\n", npes);
      }

    // verify only NPES PEs are being used
    if(npes != NPES) {
      if(my_PE_num==0) {
        printf("This code must be run with %d PEs\n", NPES);
      }
      MPI_Finalize();
      exit(1);
    }

    // PE 0 asks for input
    if(my_PE_num==0) {
      //      printf("Maximum iterations [100-4000]?\n");
      //      fflush(stdout); // Not always necessary, but can be helpful
      //      scanf("%d", &max_iterations);
      printf("Maximum iterations = %d\n", MAX_ITERATIONS);
    }

    if (my_PE_num==0) start_time = MPI_Wtime();

    initialize(npes, my_PE_num);

    while ( dt_global > MAX_TEMP_ERROR && iteration <= MAX_ITERATIONS ) {

        // main calculation: average my four neighbors
        for(i = 1; i <= ROWS; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }

        // COMMUNICATION PHASE: send ghost rows for next iteration

        // send bottom real row down
        if(my_PE_num != npes-1){             //unless we are bottom PE
            MPI_Send(&Temperature[ROWS][1], COLUMNS, MPI_DOUBLE, my_PE_num+1, DOWN, MPI_COMM_WORLD);
        }

        // receive the bottom row from above into our top ghost row
        if(my_PE_num != 0){                  //unless we are top PE
            MPI_Recv(&Temperature_last[0][1], COLUMNS, MPI_DOUBLE, my_PE_num-1, DOWN, MPI_COMM_WORLD, &status);
        }

        // send top real row up
        if(my_PE_num != 0){                    //unless we are top PE
            MPI_Send(&Temperature[1][1], COLUMNS, MPI_DOUBLE, my_PE_num-1, UP, MPI_COMM_WORLD);
        }

        // receive the top row from below into our bottom ghost row
        if(my_PE_num != npes-1){             //unless we are bottom PE
            MPI_Recv(&Temperature_last[ROWS+1][1], COLUMNS, MPI_DOUBLE, my_PE_num+1, UP, MPI_COMM_WORLD, &status);
        }

        dt = 0.0;

        for(i = 1; i <= ROWS; i++){
            for(j = 1; j <= COLUMNS; j++){
	        dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
	        Temperature_last[i][j] = Temperature[i][j];
            }
        }

        // find global dt                                                        
        MPI_Reduce(&dt, &dt_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt_global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // periodically print test values - only for PE in lower corner
        if((iteration % 100) == 0) {
	  if (my_PE_num == npes-1) {
	    i_buf = 0.04 * (iteration-1);   // i_buf = (iteration/100) * 4
	    buf_track[i_buf]   = dt_global;
	    buf_track[i_buf+1] = Temperature[ROWS-5][COLUMNS-5];
	    buf_track[i_buf+2] = Temperature[ROWS-4][COLUMNS-4];
	    buf_track[i_buf+3] = Temperature[ROWS-3][COLUMNS-3];
	    // track_progress_org(iteration, dt_global);
	  }
        }

	iteration++;
    }

    track_progress(i_buf+3, buf_track);

    // Slightly more accurate timing and cleaner output 
    MPI_Barrier(MPI_COMM_WORLD);

    // PE 0 finish timing and output values
    if (my_PE_num==0){
        stop_time = MPI_Wtime();
	elapsed_time = stop_time - start_time;

	printf("\nMax error at iteration %d was %20.15g\n", iteration-1, dt_global);
	printf("Total time was %f seconds.\n", elapsed_time);
    }

    MPI_Finalize();
}



void initialize(int npes, int my_PE_num){

    double tMin, tMax;  //Local boundary limits
    int i,j;

    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // Local boundry condition endpoints
    tMin = (my_PE_num)*100.0/npes;
    tMax = (my_PE_num+1)*100.0/npes;

    // Left and right boundaries
    for (i = 0; i <= ROWS+1; i++) {
      Temperature_last[i][0] = 0.0;
      Temperature_last[i][COLUMNS+1] = tMin + ((tMax-tMin)/ROWS)*i;
    }

    // Top boundary (PE 0 only)
    if (my_PE_num == 0)
      for (j = 0; j <= COLUMNS+1; j++)
	Temperature_last[0][j] = 0.0;

    // Bottom boundary (Last PE only)
    if (my_PE_num == npes-1)
      for (j=0; j<=COLUMNS+1; j++)
	Temperature_last[ROWS+1][j] = (100.0/COLUMNS) * j;

    // Allocate buffer for track_progress
    
}


// only called by last PE
// print diagonal in bottom right corner where most action is
void track_progress(int const iter, double* buf_track) {

    int i;
    for (i = 0; i < iter; i+=4) {
      printf("---- Iteration %d, dt = %f ----\n", (i+1)*100, buf_track[i]);
      printf("[%d,%d]: %5.2f  [%d,%d]: %5.2f  [%d,%d]: %5.2f  ",
	     ROWS_GLOBAL-i, COLUMNS-i, buf_track[i+1],
	     ROWS_GLOBAL-i, COLUMNS-i, buf_track[i+2],
	     ROWS_GLOBAL-i, COLUMNS-i, buf_track[i+3]);
    }
    printf("\n");
}


void output(int my_pe, int iteration) {
  FILE* fp;
  char filename[50];
  sprintf(filename, "output%d.txt", iteration);

  for (int pe = 0; pe<4; pe++) {
    if (my_pe == pe) {
      fp = fopen(filename, "a");

      for(int y = 1; y <= ROWS; y++) {
	for (int x = 1; x < COLUMNS; x++) {
	  fprintf(fp, "%5.3f ", Temperature[y][x]);
	}
	fprintf(fp, "\n");
      }
      fflush(fp);
      fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
