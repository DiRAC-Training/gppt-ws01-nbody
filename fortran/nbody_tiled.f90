! ============================================================================
! N-body simulation -- tiled OpenMP target offload reference.
!
! PRESENTER / ADVANCED REFERENCE ONLY. This is not part of the student
! exercise (see README.md and nbody_start.f90) and is not distributed to
! learners, the same way ../cuda/nbody_tiled.cu is CUDA-side reference
! material only.
!
! This is a copy of the nbody.f90 solution with one addition:
! `calc_acc_tiled`, which reimplements calc_acc's O(n^2) pairwise loop using
! an explicit tiling strategy for GPU shared-memory reuse -- the OpenMP
! analogue of ../cuda/nbody_tiled.cu's calc_acc_tiled. run_sim calls
! calc_acc_tiled instead of calc_acc; calc_acc itself is left in place,
! unused by run_sim, purely so its unit test still exercises it directly.
!
! Why this is not `!$omp target teams distribute parallel do` like the rest
! of the exercise: OpenMP has no `__shared__` keyword. The nearest thing is
! asking for a team-private array -- one instance per GPU thread block,
! visible to every thread in that block -- which requires separating
! `teams`/`distribute` (one iteration per team = one CUDA block) from a
! nested `parallel` region (threads within that block), so that a
! `private()` clause on `distribute` can give each team its own copy of the
! tile-staging arrays. `nvfortran` places such team-private arrays in real
! CUDA shared memory (confirmed here with `-Minfo=mp`, which reports
! `Team private (..., pos_s, mass_s) located in CUDA shared memory`).
!
! The standards-correct spelling of this -- `!$omp allocate(...)
! allocator(omp_pteam_mem_alloc)` on an array declared inside a Fortran
! `BLOCK` construct nested in the `distribute` loop -- is NOT supported by
! nvfortran 26.5: it fails to compile with "Unrecognized OpenMP directive -
! allocate" and "Unimplemented feature: BLOCK construct in the scope of a
! parallel directive". The `private()`-clause approach below is the tested
! fallback; if you're on a newer nvfortran, it's worth re-trying the
! allocator spelling, which is more portable to other OpenMP-offload
! compilers.
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

    ! Tiled version of calc_acc's pairwise loop. Each GPU thread still owns
    ! exactly one particle `i` and accumulates its own acceleration, same as
    ! calc_acc -- what changes is *how* it reads the source particles `j`.
    !
    ! Instead of every thread independently re-reading all n entries of pos
    ! and mass from global memory, threads within a team cooperatively stage
    ! one tile (TILE particles) into team-shared arrays, all threads in the
    ! team consume that tile from shared memory, then the team moves on to
    ! the next tile. This makes the reuse explicit instead of hoping the
    ! cache catches it.
    !
    !   for each tile of source particles:
    !       every thread loads one particle into the shared tile arrays
    !       <barrier: wait for the whole tile to be loaded>
    !       every thread accumulates its own acceleration from the tile
    !       <barrier: wait for everyone to finish reading before it's overwritten>
    !
    ! One CUDA block == one OpenMP team == one iteration of the `distribute`
    ! loop below; one CUDA thread == one OpenMP thread of the nested
    ! `parallel` region. TILE doubles as both the tile size and the team
    ! (block) size, same as ../cuda/nbody_tiled.cu.
    subroutine calc_acc_tiled(acc, pos, mass)
        use omp_lib, only: omp_get_thread_num
        real(wp), intent(inout) :: acc(:,:)
        real(wp), intent(in) :: pos(:,:)
        real(wp), intent(in) :: mass(:)

        integer, parameter :: TILE = 128
        integer :: n, num_teams_needed
        integer :: team_id, tid, i, t, tile_start, j
        real(wp) :: epsilon, dx, dy, dist_sq, inv_dist_cube, ax, ay
        ! Team-private tile-staging arrays. Privatized (one instance per
        ! team, shared by that team's threads) by the `private()` clause on
        ! `distribute` below, not by an explicit allocator -- see the file
        ! header comment for why.
        real(wp) :: pos_s(TILE, 2)
        real(wp) :: mass_s(TILE)

        n = size(pos, 1)
        epsilon = 1.1_wp * (real(n, wp)**(-0.48_wp))
        num_teams_needed = (n + TILE - 1) / TILE

        !$omp target teams num_teams(num_teams_needed) thread_limit(TILE) &
        !$omp&   map(to: pos, mass) map(tofrom: acc)
        !$omp distribute private(pos_s, mass_s)
        do team_id = 0, num_teams_needed - 1
            !$omp parallel private(tid, i, ax, ay, t, tile_start, j, dx, dy, dist_sq, inv_dist_cube)
            tid = omp_get_thread_num()
            i = team_id * TILE + tid + 1
            ax = 0.0_wp
            ay = 0.0_wp

            do t = 0, num_teams_needed - 1
                tile_start = t * TILE

                ! Cooperative load: each thread stages exactly one source
                ! particle. Threads past the end of a partly-full final tile
                ! pad with mass = 0, which contributes exactly nothing to
                ! the sum below -- that's what lets every thread take the
                ! same path through both barriers with no branch in the
                ! inner loop. (See ../cuda/nbody_tiled.cu, Hint 4, for the
                ! same reasoning in CUDA.)
                if (tile_start + tid + 1 <= n) then
                    pos_s(tid+1, 1) = pos(tile_start + tid + 1, 1)
                    pos_s(tid+1, 2) = pos(tile_start + tid + 1, 2)
                    mass_s(tid+1)   = mass(tile_start + tid + 1)
                else
                    pos_s(tid+1, 1) = 0.0_wp
                    pos_s(tid+1, 2) = 0.0_wp
                    mass_s(tid+1)   = 0.0_wp
                end if
                !$omp barrier

                ! The self-interaction term (source particle == i) is not
                ! special-cased: it's still visited here with dx = dy = 0,
                ! contributing exactly zero, same as in calc_acc.
                if (i <= n) then
                    do j = 1, TILE
                        dx = pos_s(j,1) - pos(i,1)
                        dy = pos_s(j,2) - pos(i,2)
                        dist_sq = dx**2 + dy**2 + epsilon**2
                        inv_dist_cube = 1.0_wp / (dist_sq * sqrt(dist_sq))
                        ax = ax + dx * mass_s(j) * inv_dist_cube
                        ay = ay + dy * mass_s(j) * inv_dist_cube
                    end do
                end if
                !$omp barrier
            end do

            if (i <= n) then
                acc(i,1) = ax
                acc(i,2) = ay
            end if
            !$omp end parallel
        end do
        !$omp end distribute
        !$omp end target teams
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
        call calc_acc_tiled(acc, pos, mass)
        !$omp end target data

        pos_prev = pos - vel * dt - 0.5_wp * acc * dt**2

        t = 0.0_wp

        !$omp target data map(tofrom: pos) map(to:mass, pos_prev) map(alloc: acc, pos_temp)
        call system_clock(count_start, count_rate)
        do while (t < total_time)
            call calc_acc_tiled(acc, pos, mass)
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

    ! Same as test_calc_acc, but exercises calc_acc_tiled instead. Only 2
    ! particles against TILE = 128 means every call here runs a single,
    ! mostly-empty tile -- this is exactly the partial-tile padding path
    ! described in calc_acc_tiled's comments, so a broken pad value or a
    ! missing barrier would show up here as a wrong answer, not a hang or
    ! crash. It does NOT exercise the multi-tile path (see README verification
    ! notes: that's checked separately by diffing a large-N trajectory
    ! against calc_acc's).
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
        call calc_acc_tiled(acc, pos, mass)
        !$omp end target data
        epsilon = 1.1 * (2.0**(-0.48))

        expected_acc(1, :) = [1.0, 0.0] * mass(2) * (1.0 + epsilon**2)**(-1.5)
        expected_acc(2, :) = -[1.0, 0.0] * mass(1) * (1.0 + epsilon**2)**(-1.5)
        call assert_almost_equal(acc, expected_acc, "test_calc_acc_tiled horizontal")

        pos(1, :) = [0.0, 0.0]
        pos(2, :) = [0.0, 1.0]

        !$omp target data map(to: pos, mass) map(from: acc)
        call calc_acc_tiled(acc, pos, mass)
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
    call test_calc_acc_tiled()
    call test_advance_pos()

end program test
#endif
