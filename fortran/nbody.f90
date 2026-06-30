module nbody_simulation
    use iso_fortran_env, only: wp => real64
    implicit none

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
        acc = 0.0_wp

        do i = 1, n
            do j = 1, n
                dx = pos(j,1) - pos(i,1)
                dy = pos(j,2) - pos(i,2)
                dist_sq = dx**2 + dy**2 + epsilon**2
                inv_dist_cube = 1.0_wp / (dist_sq**1.5_wp)
                acc(i,1) = acc(i,1) + dx * mass(j) * inv_dist_cube
                acc(i,2) = acc(i,2) + dy * mass(j) * inv_dist_cube
            end do
        end do
    end subroutine calc_acc

    subroutine advance_pos(acc, pos, pos_prev, pos_temp, dt)
        real(wp), intent(in) :: acc(:,:)
        real(wp), intent(inout) :: pos(:,:)
        real(wp), intent(inout) :: pos_prev(:,:)
        real(wp), intent(out) :: pos_temp(:,:)
        real(wp), intent(in) :: dt

        pos_temp = pos
        pos = 2.0_wp * pos - pos_prev + acc * dt**2
        pos_prev = pos_temp
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
        total_time = 10000.0_wp * dt

        allocate(pos(n, 2), vel(n, 2), mass(n))
        allocate(acc(n, 2), pos_temp(n, 2), pos_prev(n, 2))

        if (is_solar_system) then
            call create_solar_system(pos, vel, mass)
        else
            call generate_random_star_system(n, pos, vel, mass)
        end if

        call calc_acc(acc, pos, mass)

        pos_prev = pos - vel * dt - 0.5_wp * acc * dt**2

        call system_clock(count_start, count_rate)

        t = 0.0_wp
        do while (t < total_time)
            call calc_acc(acc, pos, mass)
            call advance_pos(acc, pos, pos_prev, pos_temp, dt)
            t = t + dt
        end do

        if (ios == 0) then
            do i = 1, n
                write(file_unit, '(F12.6, A, F12.6)') pos(i, 1), ',', pos(i, 2)
            end do
        end if

        call system_clock(count_end)
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
        
        call calc_acc(acc, pos, mass)
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
    integer, dimension(5) :: n_particle_range = [8, 16, 32, 64, 128]
    real(wp), dimension(5) :: runtimes

    open(newunit=file_unit, file='trajectory.csv', status='replace', action='write', iostat=ios)
    if (ios /= 0) then
        print *, "Error opening trajectory.csv"
    end if

    do i = 1, 5
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
    call test_advance_pos()

end program test
#endif
