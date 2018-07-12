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
#define ENABLE_DOUBLE_BUFFERING

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

#define ROWS_GPU        (ROWS/4*3)
    // TODO
#define ROW_GPU_FIRST   ((ROWS-ROWS_GPU)/2)
#define ROW_GPU_LAST    (ROW_GPU_FIRST+ROWS_GPU)

// communication tags
#define DOWN     100
#define UP       101

// Original Parmeters
#define DEPTH	4 //It must be lower than ROWS
// largest permitted change in temp (This value takes 3264 steps)
#define MAX_TEMP_ERROR 0.01

#ifdef ENABLE_LARGE
    #define CHECK_ROW   8064
    #define CHECK_COL   10702
#else
    #define CHECK_ROW   504
    #define CHECK_COL   622
#endif

#ifdef ENABLE_DOUBLE_BUFFERING
double Temperature_0[ROWS+2*DEPTH][COLUMNS+2];
double Temperature_1[ROWS+2*DEPTH][COLUMNS+2];

#define Temperature_last    ((temp_iter & 1 == 1) ? Temperature_0 : Temperature_1)
#define Temperature         ((temp_iter & 1 == 1) ? Temperature_1 : Temperature_0)
#else
double Temperature[ROWS+2*DEPTH][COLUMNS+2];
double Temperature_last[ROWS+2*DEPTH][COLUMNS+2];
#endif

void initialize();
void track_progress(int iter, double dt);
void debug_dumpallarry();

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

    // MPI_Request ireq;

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

    //initialize(npes, my_PE_num);
    initialize();
    
    int temp_iter = 0;
    
    #ifdef ENABLE_DOUBLE_BUFFERING
    #pragma acc data copyin(Temperature_0), create(Temperature_1)
    #else
    #pragma acc data copyin(Temperature_last), create(Temperature)
    #endif
    while ( dt_global > MAX_TEMP_ERROR && iteration <= max_iterations ) {
        // main calculation: average my four neighbors
        #define INNER_LOOP_CALC \
            for (int j = 1; j <= COLUMNS; j++) { \
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] + \
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]); \
            }
        
        #define INNER_LOOP_MAX(dt_red) \
            for (int j = 1; j <= COLUMNS; j++) { \
                dt_red = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt_red); \
            }
        
        #define INNER_LOOP_COPY \
            for (int j = 1; j <= COLUMNS; j++) { \
                Temperature_last[i][j] = Temperature[i][j]; \
            }
        
        double dt_omp_1 = 0.0, dt_omp_2 = 0.0, dt_acc = 0.0;
        
        for(int d=0;d<DEPTH;d++){
            temp_iter = iteration+d;
            // main calculation: average my four neighborsa
            
            #pragma acc parallel async(1)
            {
                #pragma acc loop
                for(i = ROW_GPU_FIRST; i < ROW_GPU_LAST; i++) { INNER_LOOP_CALC }
            }
            
            //This code block is to surpress updates of bounderies.
            int start_i=1+d;
            int end_i=ROWS+DEPTH*2-2-d;
            if(my_PE_num == 0) start_i = fmax(start_i, DEPTH);
            if(my_PE_num == npes-1) end_i = fmin(end_i, ROWS+DEPTH-1);
            // if(my_PE_num==2){ debug_dumpallarry(); printf("%d-%d\n",start_i,end_i);}
            
            #pragma omp parallel
            {
                #pragma omp for nowait
                for(i = start_i; i < ROW_GPU_FIRST; i++) { INNER_LOOP_CALC }
                #pragma omp for
                for(i = ROW_GPU_LAST; i <= end_i; i++) { INNER_LOOP_CALC }
            }
            
            #pragma acc wait(1)
            
            #pragma acc update device(Temperature[ROW_GPU_FIRST-1:1][1:COLUMNS]), \
                               host(Temperature[ROW_GPU_FIRST:1][1:COLUMNS]), \
                               host(Temperature[ROW_GPU_LAST-1:1][1:COLUMNS]), \
                               device(Temperature[ROW_GPU_LAST:1][1:COLUMNS])
            
            #ifndef ENABLE_DOUBLE_BUFFERING
            if(d != DEPTH-1) {
                for(i = start_i; i <= end_i; i++){ INNER_LOOP_COPY }
            }
            #endif
        }

        #pragma acc parallel async(1)
        {
            #pragma acc loop
            for(i = ROW_GPU_FIRST; i < ROW_GPU_LAST; i++) { INNER_LOOP_MAX(dt_acc) INNER_LOOP_COPY }
        }
        #pragma omp parallel
        {
            #pragma omp for nowait reduction(max:dt_omp_1)
            for (i = DEPTH; i < ROW_GPU_FIRST; i++) { INNER_LOOP_MAX(dt_omp_1) INNER_LOOP_COPY }
            #pragma omp for        reduction(max:dt_omp_2)
            for (i = ROW_GPU_LAST; i <= ROWS+DEPTH-1; i++) { INNER_LOOP_MAX(dt_omp_2) INNER_LOOP_COPY }
        }
        #pragma acc wait(1)
        
        dt = fmax(fmax(dt_omp_1, dt_omp_2), dt_acc);

        // COMMUNICATION PHASE: send ghost rows for next iteration

        // send bottom real row down
        if(my_PE_num != npes-1){             //unless we are bottom PE
            MPI_Send(&Temperature[ROWS][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, DOWN, MPI_COMM_WORLD);
            /* MPI_Isend(&Temperature_last[ROWS][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, DOWN, MPI_COMM_WORLD, &ireq); */
        }

        // receive the bottom row from above into our top ghost row
        if(my_PE_num != 0){                  //unless we are top PE
            #ifdef ENABLE_DOUBLE_BUFFERING
            MPI_Recv(&Temperature[0][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, DOWN, MPI_COMM_WORLD, &status);
            #else
            MPI_Recv(&Temperature_last[0][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, DOWN, MPI_COMM_WORLD, &status);
            /* MPI_Irecv(&Temperature_last[0][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, DOWN, MPI_COMM_WORLD, &ireq); */
            #endif
        }

        // send top real row up
        if(my_PE_num != 0){                    //unless we are top PE
            MPI_Send(&Temperature[DEPTH][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, UP, MPI_COMM_WORLD);
            /* MPI_Isend(&Temperature_last[DEPTH][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, UP, MPI_COMM_WORLD, &ireq); */
        }

        // receive the top row from below into our bottom ghost row
        if(my_PE_num != npes-1){             //unless we are bottom PE
            #ifdef ENABLE_DOUBLE_BUFFERING
            MPI_Recv(&Temperature[ROWS+DEPTH][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, UP, MPI_COMM_WORLD, &status);
            #else
            MPI_Recv(&Temperature_last[ROWS+DEPTH][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, UP, MPI_COMM_WORLD, &status);
            /* MPI_Irecv(&Temperature_last[ROWS+DEPTH][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, UP, MPI_COMM_WORLD, &ireq); */
            #endif
        }

        // find global dt
        MPI_Reduce(&dt, &dt_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast(&dt_global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // periodically print test values - only for PE in lower corner
        if((iteration % 100) == 0) {
            if (my_PE_num == npes-1 ){
                track_progress(iteration, dt_global);
            }
        }

        iteration+=DEPTH;
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
                Temperature_last[CHECK_ROW % ROWS + DEPTH][CHECK_COL],
                Temperature[CHECK_ROW % ROWS + DEPTH][CHECK_COL]
                // TODO: Temperature becomes 0.0
              );
    }

    MPI_Finalize();
}



void initialize(){

    double tMin, tMax;  //Local boundary limits
    int i,j;
    #ifdef ENABLE_DOUBLE_BUFFERING
    int temp_iter = 1;
    #endif

    memset(Temperature_last, 0, sizeof(Temperature_last));

    // Local boundry condition endpoints
    tMin = (my_PE_num)*100.0/npes;
    tMax = (my_PE_num+1)*100.0/npes;

    // Left and right boundaries
    for (i = 0; i < ROWS+DEPTH*2; i++) {
        Temperature[i][0] = Temperature_last[i][0] = 0.0;
        Temperature[i][COLUMNS+1] = Temperature_last[i][COLUMNS+1] = tMin + ((tMax-tMin)/ROWS)*(i-DEPTH+1);
    }

    // Top boundary (PE 0 only)
    if (my_PE_num == 0)
        for (j = 0; j <= COLUMNS+1; j++)
            Temperature[DEPTH-1][j] = Temperature_last[DEPTH-1][j] = 0.0;

    // Bottom boundary (Last PE only)
    if (my_PE_num == npes-1)
        for (j=0; j<=COLUMNS+1; j++)
            Temperature[DEPTH+ROWS][j] = Temperature_last[DEPTH+ROWS][j] = (100.0/COLUMNS) * j;

}


// only called by last PE
// print diagonal in bottom right corner where most action is
void track_progress(int iteration, double dt) {

    int i;
    #ifdef ENABLE_DOUBLE_BUFFERING
    int temp_iter = iteration+DEPTH-1;
    #endif

    printf("---- Iteration %d, dt = %f ----\n", iteration, dt);
    // output global coordinates so user doesn't have to understand decompositi
    for(i = 5; i >= 3; i--) {
        printf("[%d,%d]: %5.2f  ", ROWS_GLOBAL-i, COLUMNS-i, Temperature[ROWS+DEPTH-i][COLUMNS-i]);
    }
    printf("\n");
}

#if 0
void debug_dumpallarry(){
    for(int i =0;i<ROWS+2*DEPTH;i++){
        for(int j=0;j<COLUMNS+2;j++){
            printf("%5.2f,",Temperature_last[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
#endif
