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
#include <string.h>
#include <math.h>
#include <mpi.h>

//#define ENABLE_LARGE

#ifdef ENABLE_LARGE
    #define COLUMNS       10752
    #define ROWS_GLOBAL   10752        // this is a "global" row count
#else
    #define COLUMNS       672
    #define ROWS_GLOBAL   672        // this is a "global" row count
#endif

// Use 10752 (16 times bigger) for large challenge problem
// All chosen to be easily divisible by Bridges' 28 cores per node

#define NPES            4        // number of processors
#define ROWS (ROWS_GLOBAL/NPES)  // number of real local rows

#define ROWS_GPU        100
#define ROW_GPU_FIRST   ((ROWS-ROWS_GPU)/2)
#define ROW_GPU_LAST    (ROW_GPU_FIRST+ROWS_GPU)

// communication tags
#define DOWN     100
#define UP       101

// Original Parmeters
//#define DEPTH	1 //It must be lower than ROWS
#define DEPTH	10 //It must be lower than ROWS
// largest permitted change in temp (This value takes 3264 steps)
#define MAX_TEMP_ERROR 0.01

#ifdef ENABLE_LARGE
    #define CHECK_ROW   8064
    #define CHECK_COL   10702
#else
    #define CHECK_ROW   504
    #define CHECK_COL   622
#endif

double Temperature[ROWS+2*DEPTH][COLUMNS+2];
double Temperature_last[ROWS+2*DEPTH][COLUMNS+2];

void initialize();
void track_progress(int iter, double dt);

int        npes;                // number of PEs
int        my_PE_num;           // my PE number

int main(int argc, char *argv[]) {

    int i;
    int max_iterations;
    int iteration=1;
    double dt;
    double start_time, stop_time, elapsed_time;

    double     dt_global=100;       // delta t across all PEs
    MPI_Status status;              // status returned by MPI calls

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
        max_iterations = 4000;
        printf("Maximum iterations = %d\n", max_iterations);

    }

    // bcast max iterations to other PEs
    MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_PE_num==0) start_time = MPI_Wtime();

    initialize(npes, my_PE_num);
    
    #pragma acc data copy(Temperature_last), create(Temperature)
    while ( dt_global > MAX_TEMP_ERROR && iteration <= max_iterations ) {
        // main calculation: average my four neighbors
        #define INNER_LOOP { \
            int j; \
            for(j = 1; j <= COLUMNS; j++) { \
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] + \
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]); \
            } \
        }
        
        #pragma acc parallel
        {
            for (int d = 0; d < DEPTH; d++) {
                int start_i = ROW_GPU_FIRST-(DEPTH-1)+d;
                int end_i   = ROW_GPU_LAST +(DEPTH-1)-d;
                #pragma acc loop
                for(i = start_i; i < end_i; i++) { INNER_LOOP }
            }
        }
        #pragma omp parallel
        {
            for (int d = 0; d < DEPTH; d++) {
                int upper_start_i = 1+d;
                int upper_end_i   = ROW_GPU_FIRST+(DEPTH-1)-d;
                int lower_start_i = ROW_GPU_LAST +(DEPTH-1)-d;
                int lower_end_i   = (ROWS+DEPTH*2)-1-d;
                #pragma omp for nowait
                for(i = upper_start_i; i < upper_end_i; i++) { INNER_LOOP }
                #pragma omp for nowait
                for(i = lower_start_i; i < lower_end_i; i++) { INNER_LOOP }
            }
        }
        
        #undef INNER_LOOP
        
        //#pragma acc kernels wait(1)

        // COMMUNICATION PHASE: send ghost rows for next iteration

        // send bottom real row down
        if(my_PE_num != npes-1){             //unless we are bottom PE
            MPI_Send(&Temperature[ROWS+DEPTH-1][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, DOWN, MPI_COMM_WORLD);
        }

        // receive the bottom row from above into our top ghost row
        if(my_PE_num != 0){                  //unless we are top PE
            MPI_Recv(&Temperature_last[0][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, DOWN, MPI_COMM_WORLD, &status);
        }

        // send top real row up
        if(my_PE_num != 0){                    //unless we are top PE
            MPI_Send(&Temperature[1+DEPTH-1][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, UP, MPI_COMM_WORLD);
        }

        // receive the top row from below into our bottom ghost row
        if(my_PE_num != npes-1){             //unless we are bottom PE
            MPI_Recv(&Temperature_last[ROWS+1][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, UP, MPI_COMM_WORLD, &status);
        }

        dt = 0.0;

        //#pragma omp parallel for collapse(2)
        {
        double dt_omp_1 = 0.0, dt_omp_2 = 0.0, dt_acc = 0.0;
        
        #define INNER_LOOP(dt_red) { \
            int j; \
            for(j = 1; j <= COLUMNS; j++){ \
                dt_red = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt_red); \
                Temperature_last[i][j] = Temperature[i][j]; \
            } \
        }
        
        #pragma acc kernels
        for (i = ROW_GPU_FIRST; i < ROW_GPU_LAST; i++) { INNER_LOOP(dt_acc) }
        #pragma omp parallel
        {
            int upper_start_i = DEPTH;
            int upper_end_i   = ROW_GPU_FIRST;
            int lower_start_i = ROW_GPU_LAST ;
            int lower_end_i   = (ROWS+DEPTH*2)-DEPTH;
            #pragma omp for nowait reduction(max:dt_omp_1)
            for (i = upper_start_i; i < upper_end_i; i++) { INNER_LOOP(dt_omp_1) }
            #pragma omp for nowait reduction(max:dt_omp_2)
            for (i = lower_start_i; i < lower_end_i; i++) { INNER_LOOP(dt_omp_2) }
        }
        
        dt = fmax(fmax(dt_omp_1, dt_omp_2), dt_acc);
        
        #undef INNER_LOOP
        }

        // find global dt
        MPI_Reduce(&dt, &dt_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast(&dt_global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // periodically print test values - only for PE in lower corner
        if((iteration % 100) == 0) {
            if (my_PE_num == npes-1){
                //#pragma acc update host(Temperature)
                track_progress(iteration, dt_global);
            }
        }

        iteration+=DEPTH;
        
        #pragma acc update device(Temperature_last[ROW_GPU_FIRST-DEPTH:DEPTH][0:COLUMNS]), \
                           host(Temperature_last[ROW_GPU_FIRST:DEPTH][0:COLUMNS]), \
                           host(Temperature_last[ROW_GPU_LAST-DEPTH:DEPTH][0:COLUMNS]), \
                           device(Temperature_last[ROW_GPU_LAST:DEPTH][0:COLUMNS])
    }

    // Slightly more accurate timing and cleaner output
    MPI_Barrier(MPI_COMM_WORLD);

    // PE 0 finish timing and output values
    if (my_PE_num==0){
        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        printf("\nMax error at iteration %d was %20.15g\n", iteration-1, dt_global);
        printf("Total time was %f seconds.\n", elapsed_time);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    if (CHECK_ROW/ROWS == my_PE_num) {
        printf("PE %d: T(%d,%d) = %lf (%lf)\n",
                my_PE_num, CHECK_ROW, CHECK_COL,
                Temperature_last[CHECK_ROW % ROWS][CHECK_COL],
                Temperature[CHECK_ROW % ROWS][CHECK_COL]
                // TODO: Temperature becomes 0.0
              );
    }

    MPI_Finalize();
}



void initialize(){

    double tMin, tMax;  //Local boundary limits
    int i,j;

    memset(Temperature_last, 0, sizeof(Temperature_last));

    // Local boundry condition endpoints
    tMin = (my_PE_num)*100.0/npes;
    tMax = (my_PE_num+1)*100.0/npes;

    // Left and right boundaries
    for (i = 0; i <= ROWS+1; i++) {
        Temperature_last[i+DEPTH-1][0] = 0.0;
        Temperature_last[i+DEPTH-1][COLUMNS+1] = tMin + ((tMax-tMin)/ROWS)*i;
    }

    // Top boundary (PE 0 only)
    if (my_PE_num == 0)
        for (j = 0; j <= COLUMNS+1; j++)
            Temperature_last[DEPTH-1][j] = 0.0;

    // Bottom boundary (Last PE only)
    if (my_PE_num == npes-1)
        for (j=0; j<=COLUMNS+1; j++)
            Temperature_last[DEPTH+ROWS][j] = (100.0/COLUMNS) * j;

}


// only called by last PE
// print diagonal in bottom right corner where most action is
void track_progress(int iteration, double dt) {

    int i;

    printf("---- Iteration %d, dt = %f ----\n", iteration, dt);
    // output global coordinates so user doesn't have to understand decompositi
    for(i = 5; i >= 3; i--) {
        printf("[%d,%d]: %5.2f  ", ROWS_GLOBAL-i, COLUMNS-i, Temperature[ROWS-i][COLUMNS-i]);
    }
    printf("\n");
}
