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

double (*Temperature_last)[COLUMNS+2] = Temperature_0;
double (*Temperature)[COLUMNS+2] = Temperature_1;
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

    int i, j;
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

    while ( dt_global > MAX_TEMP_ERROR && iteration <= max_iterations ) {

        for(int d=0;d<DEPTH;d++){
            Temperature_last = ((iteration+d) % 2 == 1) ? Temperature_0 : Temperature_1;
            Temperature      = ((iteration+d) % 2 == 1) ? Temperature_1 : Temperature_0;
            
            // main calculation: average my four neighborsa

            //This code block is to surpress updates of bounderies.
            int start_i=1+d;
            int end_i=ROWS+DEPTH*2-2-d;
            if(my_PE_num == 0) start_i = fmax(start_i, DEPTH);
            if(my_PE_num == npes-1) end_i = fmin(end_i, ROWS+DEPTH-1);
            // if(my_PE_num==2){ debug_dumpallarry(); printf("%d-%d\n",start_i,end_i);}

            for(i = start_i; i <= end_i; i++) {
                for(j = 1; j <= COLUMNS; j++) {
                    Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] + Temperature_last[i][j+1] + Temperature_last[i][j-1]);
                }
            }
            #ifndef ENABLE_DOUBLE_BUFFERING
            if(d!=DEPTH-1){
                for(i = start_i; i <= end_i; i++){
                    for(j = 1; j <= COLUMNS; j++){
                        Temperature_last[i][j]=Temperature[i][j];
                    }
                }
            }
            #endif
        }

        dt = 0;

        for(i = DEPTH; i <= ROWS+DEPTH-1; i++){
            for(j = 1; j <= COLUMNS; j++){
                dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
                #ifndef ENABLE_DOUBLE_BUFFERING
                Temperature_last[i][j] = Temperature[i][j];
                #endif
            }
        }

        // COMMUNICATION PHASE: send ghost rows for next iteration

        // send bottom real row down
        if(my_PE_num != npes-1){             //unless we are bottom PE
            MPI_Send(&Temperature[ROWS][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, DOWN, MPI_COMM_WORLD);
        }

        // receive the bottom row from above into our top ghost row
        if(my_PE_num != 0){                  //unless we are top PE
            #ifdef ENABLE_DOUBLE_BUFFERING
            MPI_Recv(&Temperature[0][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, DOWN, MPI_COMM_WORLD, &status);
            #else
            MPI_Recv(&Temperature_last[0][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, DOWN, MPI_COMM_WORLD, &status);
            #endif
        }

        // send top real row up
        if(my_PE_num != 0){                    //unless we are top PE
            MPI_Send(&Temperature[DEPTH][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num-1, UP, MPI_COMM_WORLD);
        }

        // receive the top row from below into our bottom ghost row
        if(my_PE_num != npes-1){             //unless we are bottom PE
            #ifdef ENABLE_DOUBLE_BUFFERING
            MPI_Recv(&Temperature[ROWS+DEPTH][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, UP, MPI_COMM_WORLD, &status);
            #else
            MPI_Recv(&Temperature_last[ROWS+DEPTH][1], (COLUMNS+2)*DEPTH-2, MPI_DOUBLE, my_PE_num+1, UP, MPI_COMM_WORLD, &status);
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

    printf("---- Iteration %d, dt = %f ----\n", iteration, dt);
    // output global coordinates so user doesn't have to understand decompositi
    for(i = 5; i >= 3; i--) {
        printf("[%d,%d]: %5.2f  ", ROWS_GLOBAL-i, COLUMNS-i, Temperature[ROWS+DEPTH-i][COLUMNS-i]);
    }
    printf("\n");
}


void debug_dumpallarry(){
    for(int i =0;i<ROWS+2*DEPTH;i++){
        for(int j=0;j<COLUMNS+2;j++){
            printf("%5.2f,",Temperature_last[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

