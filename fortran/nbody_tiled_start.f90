! ============================================================================
! N-body simulation -- tiled OpenMP target offload exercise, starting point.
!
! This is the advanced/optional task described in the "Advanced Task" section
! of README.md: implement calc_acc_tiled, a tiled version of calc_acc's
! pairwise force loop that uses GPU shared memory for reuse. Read that section
! before starting -- it explains the tiling concept and the OpenMP mechanism
! from scratch. The TODO comments below (Task 5a-5j) mark where each step
! goes and are labelled to match.
!
! This file assumes you've already completed the main exercise (Tasks 1-4):
! calc_acc, advance_pos and run_sim below are already fully offloaded, the
! same as the finished nbody.f90. run_sim currently calls the plain,
! untiled calc_acc -- Task 5i is switching it over to calc_acc_tiled once
! that's implemented and tested.
!
! nbody_tiled.f90 in this same directory is the finished solution to this
! task; try not to look at it until you've had a go.
! ============================================================================
module nbody_simulation
    ! use iso_fortran_env, only: wp => real64
    implicit none

    ! integer, parameter :: wp = kind(1.0d0)
    integer, parameter :: wp = selected_real_kind(6, 37)
    real(wp), parameter :: PI = 3.14159265358979323846_wp
    real(wp), parameter :: N_YEARS = 0.1_wp
    integer :: file_unit, ios

contains

    subroutine random_numbers(num, a, b, res)
        integer, intent(in) :: num
        real(wp), intent(in) :: a, b
        real(wp), intent(out) :: res(num)
        real(wp) :: temp(num)

        call random_number(temp)
        res = temp * (b - a) + a
    end subroutine random_numbers

    subroutine calc_stable_orbit(r, theta, pos, vel)
        real(wp), intent(in) :: r(:)
        real(wp), intent(in) :: theta(:)
        real(wp), intent(out) :: pos(:,:)
        real(wp), intent(out) :: vel(:,:)

        integer :: n
        real(wp), allocatable :: v_mag(:)

        n = size(r)
        allocate(v_mag(n))

        v_mag = 1.0_wp / sqrt(r)

        pos(:,1) = r * sin(theta)
        pos(:,2) = r * cos(theta)

        vel(:,1) = -v_mag * cos(theta)
        vel(:,2) =  v_mag * sin(theta)

        deallocate(v_mag)
    end subroutine calc_stable_orbit

    subroutine generate_random_star_system(num, pos, vel, mass, min_radius, max_radius, min_mass, max_mass)
        integer, intent(in) :: num
        real(wp), intent(out) :: pos(num, 2)
        real(wp), intent(out) :: vel(num, 2)
        real(wp), intent(out) :: mass(num)
        real(wp), intent(in), optional :: min_radius, max_radius, min_mass, max_mass

        real(wp) :: r_min, r_max, m_min, m_max
        real(wp), allocatable :: r(:), theta(:)

        r_min = 0.4_wp; if (present(min_radius)) r_min = min_radius
        r_max = 20.0_wp; if (present(max_radius)) r_max = max_radius
        m_min = 1.0_wp/6000000.0_wp; if (present(min_mass)) m_min = min_mass
        m_max = 1.0_wp/1000.0_wp; if (present(max_mass)) m_max = max_mass

        allocate(r(num), theta(num))

        call random_numbers(num, r_min, r_max, r)
        call random_numbers(num, m_min, m_max, mass)
        call random_numbers(num, 0.0_wp, PI, theta)

        call calc_stable_orbit(r, theta, pos, vel)

        ! Add central star
        pos(1,:) = 0.0_wp
        vel(1,:) = 0.0_wp
        mass(1) = 1.0_wp

        deallocate(r, theta)
    end subroutine generate_random_star_system

    subroutine create_solar_system(pos, vel, mass)
        real(wp), intent(out) :: pos(9, 2)
        real(wp), intent(out) :: vel(9, 2)
        real(wp), intent(out) :: mass(9)

        real(wp) :: r(9)
        real(wp) :: theta(9)

        mass = [1.0_wp, 1.0_wp/6023600.0_wp, 1.0_wp/408524.0_wp, 1.0_wp/332946.038_wp, &
                1.0_wp/3098710.0_wp, 1.0_wp/1047.55_wp, 1.0_wp/3499.0_wp, 1.0_wp/22962.0_wp, 1.0_wp/19352.0_wp]
        r = [0.1_wp, 0.4_wp, 0.7_wp, 1.0_wp, 1.5_wp, 5.2_wp, 9.5_wp, 19.2_wp, 30.1_wp]

        call random_numbers(9, 0.0_wp, PI, theta)
        call calc_stable_orbit(r, theta, pos, vel)

        pos(1,:) = 0.0_wp
        vel(1,:) = 0.0_wp
        mass(1) = 1.0_wp
    end subroutine create_solar_system

    subroutine calc_acc(acc, pos, mass)
        real(wp), intent(inout) :: acc(:,:)
        real(wp), intent(in) :: pos(:,:)
        real(wp), intent(in) :: mass(:)

        integer :: i, j, n
        real(wp) :: epsilon, dx, dy, dist_sq, inv_dist_cube

        n = size(pos, 1)
        epsilon = 1.1_wp * (real(n, wp)**(-0.48_wp))

        !$omp target teams distribute parallel do
        do i = 1, n
          acc(i,1) = 0.0_wp
          acc(i,2) = 0.0_wp
        enddo
        !$omp end target teams distribute parallel do

        !$omp target teams distribute parallel do
        do i = 1, n
            do j = 1, n
                dx = pos(j,1) - pos(i,1)
                dy = pos(j,2) - pos(i,2)
                dist_sq = dx**2 + dy**2 + epsilon**2
                ! inv_dist_cube = 1.0_wp / (dist_sq**1.5_wp)
                inv_dist_cube = 1.0_wp / (dist_sq * sqrt(dist_sq))
                acc(i,1) = acc(i,1) + dx * mass(j) * inv_dist_cube
                acc(i,2) = acc(i,2) + dy * mass(j) * inv_dist_cube
            end do
        end do
        !$omp end target teams distribute parallel do
    end subroutine calc_acc

    ! Tiled version of calc_acc's pairwise loop -- see the "Advanced Task"
    ! section of README.md for the concept (why tiling helps, and how it
    ! maps onto OpenMP's teams/distribute/parallel/barrier constructs when
    ! there's no `__shared__` keyword to reach for) before starting.
    !
    ! Each GPU thread still owns exactly one particle `i` and accumulates
    ! its own acceleration, same as calc_acc -- what changes is *how* it
    ! reads the source particles `j`: in tiles, staged through team-shared
    ! memory, instead of one at a time straight from global memory.
    !
    !   for each tile of source particles:
    !       every thread loads one particle into the shared tile arrays
    !       <barrier: wait for the whole tile to be loaded>
    !       every thread accumulates its own acceleration from the tile
    !       <barrier: wait for everyone to finish reading before it's overwritten>
    !
    ! One CUDA block == one OpenMP team == one iteration of the `distribute`
    ! loop; one CUDA thread == one OpenMP thread of the nested `parallel`
    ! region. TILE doubles as both the tile size and the team (block) size.
    subroutine calc_acc_tiled(acc, pos, mass)
        real(wp), intent(inout) :: acc(:,:)
        real(wp), intent(in) :: pos(:,:)
        real(wp), intent(in) :: mass(:)

        ! TODO (Task 5a): declare TILE as a compile-time constant (try 128)
        ! -- it's both the tile size and the team (thread-block) size, the
        ! same reason CUDA's tiled kernel needs `const int` rather than a
        ! variable for its block size. Then declare the two team-private
        ! tile-staging arrays it sizes: pos_s(TILE, 2) and mass_s(TILE).
        ! You'll also need locals for: n, epsilon (identical role to
        ! calc_acc's), num_teams_needed, team_id, tid, i, t, tile_start, j,
        ! ax, ay, dx, dy, dist_sq, inv_dist_cube -- compare with calc_acc
        ! for which of these play the same role its i/j/dx/dy/... do. You
        ! will also need `use omp_lib, only: omp_get_thread_num`.

        ! TODO (Task 5b): compute n and epsilon exactly as calc_acc does,
        ! then num_teams_needed from n and TILE (round up -- how many teams
        ! of TILE particles each does it take to cover n particles?). Then
        ! open the teams/distribute/parallel skeleton:
        !   !$omp target teams num_teams(num_teams_needed) thread_limit(TILE) &
        !   !$omp&   map(to: pos, mass) map(tofrom: acc)
        !   !$omp distribute private(pos_s, mass_s)
        !   do team_id = 0, num_teams_needed - 1
        !       !$omp parallel private(tid, i, ax, ay, t, tile_start, j, dx, dy, dist_sq, inv_dist_cube)
        ! The `private()` clause on `distribute` is what gives each team its
        ! own instance of pos_s/mass_s, shared by that team's threads --
        ! this is the OpenMP mechanism the "Advanced Task" section explains.

        ! TODO (Task 5c): get tid from omp_get_thread_num(), then this
        ! thread's global particle index i from team_id, TILE and tid
        ! (compare: how does a CUDA kernel compute its global thread index
        ! from blockIdx, blockDim and threadIdx?). Zero this thread's
        ! accumulator, ax and ay -- the same role acc(i,:) = 0 plays at the
        ! start of calc_acc.

        ! TODO (Task 5d): outer loop over tiles, `do t = 0, num_teams_needed - 1`,
        ! tile_start = t * TILE. Cooperative load: each thread copies
        ! exactly one source particle -- global index tile_start + tid + 1
        ! -- into pos_s(tid+1,:) and mass_s(tid+1). Guard against reading
        ! past the end of pos/mass on the last, partly-full tile: pad with
        ! zeros there instead of skipping the write (see the hint on
        ! padding vs. branching in README.md -- skipping it will break the
        ! barrier below for some threads).

        ! TODO (Task 5e): first barrier (`!$omp barrier`). Nobody may read
        ! the tile until every thread in the team has finished loading it.

        ! TODO (Task 5f): accumulate into ax/ay from every slot of this tile
        ! (`do j = 1, TILE`), using pos_s/mass_s in place of the pos/mass
        ! calc_acc's j loop reads -- same force maths, just a different
        ! source array. Only threads with a real particle (i <= n) should
        ! do this; note you do NOT need to skip j == i here, the same as
        ! calc_acc doesn't -- it's naturally visited once (dx = dy = 0) and
        ! contributes exactly zero.

        ! TODO (Task 5g): second barrier. Nobody may start loading the next
        ! tile until every thread has finished reading this one. This ends
        ! the tile loop (`end do`).

        ! TODO (Task 5h): write this thread's final ax/ay into acc(i,:),
        ! guarded by i <= n. Then close the parallel region, the distribute
        ! loop, and the target teams region:
        !   !$omp end parallel
        !   end do
        !   !$omp end distribute
        !   !$omp end target teams
    end subroutine calc_acc_tiled

    subroutine advance_pos(acc, pos, pos_prev, pos_temp, dt)
        real(wp), intent(in) :: acc(:,:)
        real(wp), intent(inout) :: pos(:,:)
        real(wp), intent(inout) :: pos_prev(:,:)
        real(wp), intent(out) :: pos_temp(:,:)
        real(wp), intent(in) :: dt
        integer :: i, n

        n = size(pos, 1)

        !$omp target teams distribute parallel do
        do i=1,n
            pos_temp(i,1) = pos(i,1)
            pos_temp(i,2) = pos(i,2)
            pos(i,1) = 2.0_wp * pos(i,1) - pos_prev(i,1) + acc(i,1) * dt**2
            pos(i,2) = 2.0_wp * pos(i,2) - pos_prev(i,2) + acc(i,2) * dt**2
            pos_prev(i,1) = pos_temp(i,1)
            pos_prev(i,2) = pos_temp(i,2)
        end do
        !$omp end target teams distribute parallel do
    end subroutine advance_pos

    function run_sim(is_solar_system, plot, n_particles) result(completion_time)
        logical, intent(in) :: is_solar_system
        logical, intent(in) :: plot
        integer, intent(in) :: n_particles
        real(wp) :: completion_time

        real(wp) :: dt, total_time, t
        real(wp), allocatable :: pos(:,:), vel(:,:), mass(:)
        real(wp), allocatable :: acc(:,:), pos_temp(:,:), pos_prev(:,:)
        integer :: n, count_rate, count_start, count_end
        integer :: i

        if (is_solar_system) then
            print *, "Running regular solar system"
            n = 9
        else
            print *, "Running with ", n_particles, " particles"
            n = n_particles
        end if

        dt = 0.01_wp
        total_time = 10.0_wp * dt

        allocate(pos(n, 2), vel(n, 2), mass(n))
        allocate(acc(n, 2), pos_temp(n, 2), pos_prev(n, 2))

        if (is_solar_system) then
            call create_solar_system(pos, vel, mass)
        else
            call generate_random_star_system(n, pos, vel, mass)
        end if

        !$omp target data map(to: pos, mass) map(from: acc)
        ! TODO (Task 5i): once calc_acc_tiled is implemented and its test
        ! (Task 5j) passes, change this to calc_acc_tiled.
        call calc_acc(acc, pos, mass)
        !$omp end target data

        pos_prev = pos - vel * dt - 0.5_wp * acc * dt**2

        t = 0.0_wp

        !$omp target data map(tofrom: pos) map(to:mass, pos_prev) map(alloc: acc, pos_temp)
        call system_clock(count_start, count_rate)
        do while (t < total_time)
            ! TODO (Task 5i): change this to calc_acc_tiled too.
            call calc_acc(acc, pos, mass)
            call advance_pos(acc, pos, pos_prev, pos_temp, dt)
            t = t + dt
        end do
        call system_clock(count_end)
        !$omp end target data

        if (ios == 0) then
            do i = 1, n
                write(file_unit, '(F12.6, A, F12.6)') pos(i, 1), ',', pos(i, 2)
            end do
        end if

        completion_time = real(count_end - count_start, wp) / real(count_rate, wp)

        print '(A, F10.4, A)', "Time to complete: ", completion_time, " s"

        if (plot) then
            print *, "Plotting is not implemented in standard Fortran. Please use a library like DISLIN or export data to CSV."
        end if

        deallocate(pos, vel, mass, acc, pos_temp, pos_prev)
    end function run_sim

    subroutine assert_almost_equal(a, b, label)
        real(wp), intent(in) :: a(:,:), b(:,:)
        character(len=*), intent(in) :: label
        real(wp), parameter :: tol = 1e-6
        if (any(abs(a - b) > tol)) then
            print *, "Assertion Failed: ", label
            print *, "Expected:", b
            print *, "Actual:", a
            stop 1
        else
            print *, "Assertion Passed: ", label
        end if
    end subroutine assert_almost_equal

    subroutine test_calc_stable_orbit()
        real(wp) :: r(1), theta(1)
        real(wp) :: pos(1, 2), vel(1, 2)
        real(wp) :: pos2(1, 2), vel2(1, 2)
        real(wp) :: pos3(1, 2), vel3(1, 2)

        r = 1.0_wp
        theta = 0.0_wp
        call calc_stable_orbit(r, theta, pos, vel)
        call assert_almost_equal(pos, reshape([0.0_wp, 1.0_wp], [1_wp, 2_wp]), "test_calc_stable_orbit pos 1")
        call assert_almost_equal(vel, reshape([-1.0_wp, 0.0_wp], [1_wp, 2_wp]), "test_calc_stable_orbit vel 1")

        theta = 2.0_wp * PI
        call calc_stable_orbit(r, theta, pos2, vel2)
        call assert_almost_equal(pos, pos2, "test_calc_stable_orbit pos 2")
        call assert_almost_equal(vel, vel2, "test_calc_stable_orbit vel 2")

        theta = PI
        call calc_stable_orbit(r, theta, pos3, vel3)
        call assert_almost_equal(pos, -1.0 * pos3, "test_calc_stable_orbit pos 3")
        call assert_almost_equal(vel, -1.0 * vel3, "test_calc_stable_orbit vel 3")
    end subroutine test_calc_stable_orbit

    subroutine test_calc_acc()
        real(wp) :: mass(2)
        real(wp) :: pos(2, 2)
        real(wp) :: acc(2, 2)
        real(wp) :: epsilon
        real(wp) :: expected_acc(2, 2)

        mass = [2.0, 0.5]
        pos(1, :) = [0.0, 0.0]
        pos(2, :) = [1.0, 0.0]

        !$omp target data map(to: pos, mass) map(from: acc)
        call calc_acc(acc, pos, mass)
        !$omp end target data
        epsilon = 1.1 * (2.0**(-0.48))

        expected_acc(1, :) = [1.0, 0.0] * mass(2) * (1.0 + epsilon**2)**(-1.5)
        expected_acc(2, :) = -[1.0, 0.0] * mass(1) * (1.0 + epsilon**2)**(-1.5)
        call assert_almost_equal(acc, expected_acc, "test_calc_acc horizontal")

        pos(1, :) = [0.0, 0.0]
        pos(2, :) = [0.0, 1.0]

        call calc_acc(acc, pos, mass)
        expected_acc(1, :) = [0.0, 1.0] * mass(2) * (1.0 + epsilon**2)**(-1.5)
        expected_acc(2, :) = -[0.0, 1.0] * mass(1) * (1.0 + epsilon**2)**(-1.5)
        call assert_almost_equal(acc, expected_acc, "test_calc_acc vertical")
    end subroutine test_calc_acc

    ! TODO (Task 5j): this is currently a copy of test_calc_acc that
    ! happens to call calc_acc -- it isn't testing calc_acc_tiled at all
    ! yet. Change both `call calc_acc(...)` lines below to
    ! `call calc_acc_tiled(...)`, then find the `#ifdef TEST` program block
    ! near the bottom of this file and add `call test_calc_acc_tiled()`
    ! there.
    !
    ! With only 2 particles against a TILE of 128, this test runs entirely
    ! inside one partly-empty tile -- exactly the padding path from
    ! Task 5d -- but it can't catch a bug that only shows up across
    ! multiple tiles. See "Checking you are right" in README.md for how to
    ! test that.
    subroutine test_calc_acc_tiled()
        real(wp) :: mass(2)
        real(wp) :: pos(2, 2)
        real(wp) :: acc(2, 2)
        real(wp) :: epsilon
        real(wp) :: expected_acc(2, 2)

        mass = [2.0, 0.5]
        pos(1, :) = [0.0, 0.0]
        pos(2, :) = [1.0, 0.0]

        !$omp target data map(to: pos, mass) map(from: acc)
        call calc_acc(acc, pos, mass)
        !$omp end target data
        epsilon = 1.1 * (2.0**(-0.48))

        expected_acc(1, :) = [1.0, 0.0] * mass(2) * (1.0 + epsilon**2)**(-1.5)
        expected_acc(2, :) = -[1.0, 0.0] * mass(1) * (1.0 + epsilon**2)**(-1.5)
        call assert_almost_equal(acc, expected_acc, "test_calc_acc_tiled horizontal")

        pos(1, :) = [0.0, 0.0]
        pos(2, :) = [0.0, 1.0]

        !$omp target data map(to: pos, mass) map(from: acc)
        call calc_acc(acc, pos, mass)
        !$omp end target data
        expected_acc(1, :) = [0.0, 1.0] * mass(2) * (1.0 + epsilon**2)**(-1.5)
        expected_acc(2, :) = -[0.0, 1.0] * mass(1) * (1.0 + epsilon**2)**(-1.5)
        call assert_almost_equal(acc, expected_acc, "test_calc_acc_tiled vertical")
    end subroutine test_calc_acc_tiled

    subroutine test_advance_pos()
        real(wp) :: pos(1, 2)
        real(wp) :: pos_prev(1, 2)
        real(wp) :: pos_temp(1, 2)
        real(wp) :: acc(1, 2)
        real(wp) :: dt
        real(wp) :: expected_pos(1, 2)

        dt = 0.5
        pos(1, :) = [1.0, 2.0]
        pos_prev(1, :) = [0.5, 3.0]
        acc(1, :) = [0.5, -1.0]

        call advance_pos(acc, pos, pos_prev, pos_temp, dt)

        expected_pos(1, 1) = 2.0 - 0.5 + 0.5 * 0.5**2
        expected_pos(1, 2) = 4.0 - 3.0 + (-1.0) * 0.5**2

        call assert_almost_equal(pos, expected_pos, "test_advance_pos")
    end subroutine test_advance_pos

end module nbody_simulation

#ifdef MAIN
program main

    use nbody_simulation
    implicit none

    integer :: i
    !integer, dimension(5) :: n_particle_range = [800, 1600, 3200, 6400, 12800]
    !real(wp), dimension(5) :: runtimes
    integer, dimension(1) :: n_particle_range = [50000]
    real(wp), dimension(1) :: runtimes

    open(newunit=file_unit, file='trajectory.csv', status='replace', action='write', iostat=ios)
    if (ios /= 0) then
        print *, "Error opening trajectory.csv"
    end if

    do i = 1, 1
        runtimes(i) = run_sim(.false., .false., n_particle_range(i))
    end do

    if (ios == 0) close(file_unit)

    print *, "Particle counts:"
    print *, n_particle_range
    print *, "Runtimes:"
    print *, runtimes

end program main
#endif

#ifdef TEST
program test

    use nbody_simulation
    implicit none

    call test_calc_stable_orbit()
    call test_calc_acc()
    ! TODO (Task 5j): add `call test_calc_acc_tiled()` here once you've
    ! pointed that test at calc_acc_tiled.
    call test_advance_pos()

end program test
#endif
