!*************************************************
! Laplace OpenMP Fortran Version
!
! Temperature is initially 0.0
! Boundaries are as follows:
!
!      0         T         0
!   0  +-------------------+  0
!      |                   |
!      |                   |
!      |                   |
!   T  |                   |  T
!      |                   |
!      |                   |
!      |                   |
!   0  +-------------------+ 100
!      0         T        100
!
!  John Urbanic, PSC 2014
!
!*************************************************

program openmp

      use omp_lib

      implicit none

      !Size of plate
      integer, parameter             :: columns=672
      integer, parameter             :: rows=672

      !Use 10752 (16 times bigger) for large challenge problem
      !All chosen to be easily divisible by Bridges' 28 cores per node

      !largest permitted change in temp (This value takes 3264 steps)
      double precision, parameter    :: max_temp_error=0.01

      integer                        :: i, j, max_iterations, iteration=1
      double precision               :: dt=100.0
      double precision               :: start_time, stop_time

      double precision, dimension(0:rows+1,0:columns+1) :: temperature, temperature_last

!      print*, 'Maximum iterations [100-4000]?'
!      read*,   max_iterations

      !$omp parallel
      !$omp master
      write(*,*) 'Running on ', omp_get_num_threads(), ' OpenMP threads'
      write(*,*)
      !$omp end master
      !$omp end parallel

      max_iterations = 4000
      print*, 'Maximum iterations = ', max_iterations

      start_time = omp_get_wtime()

      call initialize(temperature_last)

      !do until error is minimal or until maximum steps
      do while ( dt > max_temp_error .and. iteration <= max_iterations)

         !$omp parallel do
         do j=1,columns
            do i=1,rows
               temperature(i,j)=0.25*(temperature_last(i+1,j)+temperature_last(i-1,j)+ &
                                      temperature_last(i,j+1)+temperature_last(i,j-1) )
            enddo
         enddo

         dt=0.0

         !copy grid to old grid for next iteration and find max change
         !$omp parallel do reduction(max:dt)
         do j=1,columns
            do i=1,rows
               dt = max( abs(temperature(i,j) - temperature_last(i,j)), dt )
               temperature_last(i,j) = temperature(i,j)
            enddo
         enddo

         !periodically print test values
         if( mod(iteration,100).eq.0 ) then
            call track_progress(temperature, iteration, dt)
         endif

         iteration = iteration+1

      enddo

      stop_time = omp_get_wtime()

      print*, 'Max error at iteration ', iteration-1, ' was ',dt
      print*, 'Total time was ',stop_time-start_time, ' seconds.'


    contains

      ! initialize plate and boundery conditions
      ! temp_last is used to to start first iteration
      subroutine initialize( temperature_last )
      implicit none

      integer                        :: i,j

      double precision, dimension(0:rows+1,0:columns+1) :: temperature_last

      temperature_last = 0.0

      !these boundary conditions never change throughout run

      !set left side to 0 and right to linear increase
      do i=0,rows+1
         temperature_last(i,0) = 0.0
         temperature_last(i,columns+1) = (100.0/rows) * i
      enddo

      !set top to 0 and bottom to linear increase
      do j=0,columns+1
         temperature_last(0,j) = 0.0
         temperature_last(rows+1,j) = ((100.0)/columns) * j
      enddo

end subroutine initialize


!print diagonal in bottom corner where most action is
subroutine track_progress(temperature, iteration, dt)
      implicit none

      integer                        :: i,iteration
      double precision, dimension(0:rows+1,0:columns+1) :: temperature
      double precision :: dt

      write (*, '("---- Iteration : ", i6, ", dt = ", f10.6, " ----")') &
           iteration, dt
      do i=5,3,-1
         write (*,'("("i5,",",i5,"):",f6.2,"  ")',advance='no') &
                   rows-i,columns-i,temperature(rows-i,columns-i)
      enddo
      print *
end subroutine track_progress

end program openmp
