!*************************************************
! Laplace MPI Fortran Version
!
! Temperature is initially 0.0
! Boundaries are as follows:
!
!          T=0.   
!       _________                    ____|____|____|____  
!       |       | 0                 |    |    |    |    | 
!       |       |                   |    |    |    |    | 
! T=0.  | T=0.0 | T                 | 0  | 1  | 2  | 3  | 
!       |       |                   |    |    |    |    | 
!       |_______| 100               |    |    |    |    | 
!       0     100                   |____|____|____|____|
!                                        |    |    |
! Each Processor works on a sub grid and then sends its
! boundaries to neighbours
!
!  John Urbanic, PSC 2014
!
!*************************************************

program mpilaplace

      use mpi

      implicit none

      !Size of plate
      integer, parameter             :: columns_global=672
      integer, parameter             :: rows=672

      !Use 10752 (16 times bigger) for large challenge problem
      !All chosen to be easily divisible by Bridges' 28 cores per node

      !these are the new parameters for parallel purposes
      integer, parameter             :: total_pes=4
      integer, parameter             :: columns=columns_global/total_pes
      integer, parameter             :: left=100, right=101

      !usual mpi variables
      integer                        :: mype, npes, ierr
      integer                        :: status(MPI_STATUS_SIZE)

      !largest permitted change in temp (This value takes 3264 steps)
      double precision, parameter    :: max_temp_error=0.01

      integer                        :: i, j, max_iterations, iteration=1
      double precision               :: dt, dt_global=100.0
      double precision               :: start_time, stop_time

      double precision, dimension(0:rows+1,0:columns+1) :: temperature, temperature_last

      !usual mpi startup routines
      call MPI_Init(ierr)
      call MPI_Comm_size(MPI_COMM_WORLD, npes, ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD, mype, ierr)

      if (mype == 0) then
         write(*,*) 'Running on ', npes, ' MPI processes'
         write(*,*)
      end if

      !It is nice to verify that proper number of PEs are running
      if ( npes /= total_pes ) then
         if( mype == 0 ) then
            print *,'This example is hardwired to run only on ', total_pes, ' PEs'
         endif
         call MPI_Finalize(ierr)
         stop
      endif

      !Only one PE should prompt user
      if( mype == 0 ) then
!         print*, 'Maximum iterations [100-4000]?'
!         read*,   max_iterations
         max_iterations = 4000
         print*, 'Maximum iterations = ', max_iterations
      endif
      
      !Other PEs need to recieve this information
      call MPI_Bcast(max_iterations, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

      start_time = MPI_Wtime()

      call initialize(temperature_last, npes, mype)

      !do until global error is minimal or until maximum steps
      do while ( dt_global > max_temp_error .and. iteration <= max_iterations)

         do j=1,columns
            do i=1,rows
               temperature(i,j)=0.25*(temperature_last(i+1,j)+temperature_last(i-1,j)+ &
                                      temperature_last(i,j+1)+temperature_last(i,j-1) )
            enddo
         enddo

         ! COMMUNICATION PHASE
         !send data
         if (mype < npes-1) then
            call MPI_Send(temperature(1,columns), rows, MPI_DOUBLE_PRECISION, &
                          mype+1, RIGHT, MPI_COMM_WORLD, ierr)
         endif

         !receive data
         if (mype /= 0) then
            call MPI_Recv(temperature_last(1,0), rows, MPI_DOUBLE_PRECISION, &
                          mype-1, RIGHT, MPI_COMM_WORLD, status, ierr)
         endif

         if (mype /= 0) then
            call MPI_Send(temperature(1,1), rows, MPI_DOUBLE_PRECISION, &
                          mype-1, LEFT, MPI_COMM_WORLD, ierr)
         endif

         if (mype /= npes-1) then
            call MPI_Recv(temperature_last(1,columns+1), rows, MPI_DOUBLE_PRECISION, &
                          mype+1, LEFT, MPI_COMM_WORLD, status, ierr)
         endif

         dt=0.0

         do j=1,columns
            do i=1,rows
               dt = max( abs(temperature(i,j) - temperature_last(i,j)), dt )
               temperature_last(i,j) = temperature(i,j)
            enddo
         enddo

         !Need to determine and communicate maximum error
         call MPI_Reduce(dt, dt_global, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
         call MPI_Bcast(dt_global, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

         !periodically print test values - only for PE in lower corner
         if( mod(iteration,100).eq.0 ) then
            if( mype == npes-1 ) then
               call track_progress(temperature, iteration, dt_global)
            endif
         endif

         iteration = iteration+1

      enddo

      !Slightly more accurate timing and cleaner output
      call MPI_Barrier(MPI_COMM_WORLD, ierr)

      stop_time = MPI_Wtime()

      if( mype == 0 ) then
         print*, 'Max error at iteration ', iteration-1, ' was ',dt_global
         print*, 'Total time was ',stop_time-start_time, ' seconds.'
      endif

      call MPI_Finalize(ierr)

    contains

      !Parallel version requires more attention to global coordinates

      subroutine initialize(temperature_last, npes, mype )
      implicit none

      integer                        :: i,j
      integer                        :: npes,mype

      double precision, dimension(0:rows+1,0:columns+1) :: temperature_last
      double precision               :: tmin, tmax

      temperature_last = 0

      !Left and Right Boundaries
      if( mype == 0 ) then
         do i=0,rows+1
            temperature_last(i,0) = 0.0
         enddo
      endif
      if( mype == npes-1 ) then
         do i=0,rows+1
            temperature_last(i,columns+1) = (100.0/rows) * i
         enddo
      endif

      !Top and Bottom Boundaries
      tmin =  mype    * 100.0/npes
      tmax = (mype+1) * 100.0/npes
      do j=0,columns+1
         temperature_last(0,j) = 0.0
         temperature_last(rows+1,j) = tmin + ((tmax-tmin)/columns) * j
      enddo

end subroutine initialize

subroutine track_progress(temperature, iteration, dt)
      implicit none

      integer                        :: i,iteration

      double precision, dimension(0:rows+1,0:columns+1) :: temperature
      double precision :: dt

!Parallel version uses global coordinate output so users don't need
!to understand decomposition
      write (*, '("---- Iteration : ", i6, ", dt = ", f10.6, " ----")') &
           iteration, dt
      do i=5,3,-1
         write (*,'("("i5,",",i5,"):",f6.2,"  ")',advance='no') &
                   rows-i,columns_global-i,temperature(rows-i,columns-i)
      enddo
      print *
end subroutine track_progress

end program mpilaplace
