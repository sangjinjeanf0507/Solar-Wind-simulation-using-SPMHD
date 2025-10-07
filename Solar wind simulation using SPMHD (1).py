import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)


mu_0 = 4 * np.pi * 1e-7
sigma_kernel = 10.0 / (7.0 * np.pi)
gamma = 5.0 / 3.0

dimension = 2
num_particles_max = 100000
dt = 7e-5

hex_spacing = 0.02


magnetic_diffusivity = 5e-4 


domain_min = -10.0
domain_max = 10.0
domain_center = ti.Vector([0.0, 0.0])


gravitational_constant = 5e-3
center_mass = 0


magnet_center = ti.Vector([0.0, 0.0])
initial_magnet_moment_strength = 2000.0
magnet_moment_strength = initial_magnet_moment_strength
magnet_moment_direction = ti.Vector([-1.0, 0.0])


magnetic_field_scale = 0.5
num_grid_points = 50


base_B_magnitude = 0.05


B_magnitude_threshold = 5e-8


max_pressure_cap = 100000000000000000000000.0


max_density_cap = 5.0 * 1e10
density_cap_activation_distance = 0.2


max_velocity_cap = 1000000000000000000.0



num_ions = 5000
num_electron_core = 5000
num_electron_halo = 5000
num_special_particles = num_ions + num_electron_core + num_electron_halo


special_particle_spawn_x_min = domain_min + 0.1 - 2.0
special_particle_spawn_x_max = domain_min + 0.5 - 2.0
special_particle_spawn_y_min = domain_min
special_particle_spawn_y_max = domain_max



ion_initial_velocity = ti.Vector([0.004, 0.0])
electron_core_initial_velocity = ti.Vector([0.004, 0.0])
electron_halo_initial_velocity_low = ti.Vector([0.002, 0.0])
electron_halo_initial_velocity_high = ti.Vector([0.01, 0.0])


ion_initial_B_magnitude = 5e-9


PARTICLE_TYPE_NORMAL = 0
PARTICLE_TYPE_ION = 1
PARTICLE_TYPE_ELECTRON_CORE = 2
PARTICLE_TYPE_ELECTRON_HALO = 3

is_special_particle_type = ti.field(dtype=ti.i32, shape=num_particles_max)


magnetic_free_zone_radius = 1.0


initial_placement_min_radius = 1.1

reposition_radius_min = magnetic_free_zone_radius + 0.1
reposition_radius_max = magnetic_free_zone_radius + 0.5


repopulation_check_radius = 2.0
num_desired_particles_in_center = 2000
repopulation_search_attempts = 100


special_particle_repop_check_center = ti.Vector([special_particle_spawn_x_min + (special_particle_spawn_x_max - special_particle_spawn_x_min) / 2.0, domain_center.y])
special_particle_repop_check_radius = 2.0
num_desired_total_special_particles = num_ions + num_electron_core + num_electron_halo
special_repopulation_search_attempts = 100


min_particle_distance = 0.03
min_particle_distance_sq = min_particle_distance * min_particle_distance


grid_pos = ti.Vector.field(dimension, dtype=ti.f32, shape=num_grid_points * num_grid_points)
grid_B_interpolated = ti.Vector.field(dimension, dtype=ti.f32, shape=num_grid_points * num_grid_points)


grid_B_dipole_only = ti.Vector.field(dimension, dtype=ti.f32, shape=num_grid_points * num_grid_points)


pos = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max)
vel = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max)
mass = ti.field(dtype=ti.f32, shape=num_particles_max)
u = ti.field(dtype=ti.f32, shape=num_particles_max)
rho = ti.field(dtype=ti.f32, shape=num_particles_max)
P_pressure = ti.field(dtype=ti.f32, shape=num_particles_max)
B = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max)
acc = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max)


etha_a_dt_field = ti.field(dtype=ti.f32, shape=num_particles_max)
ae_k_field = ti.field(dtype=ti.f32, shape=num_particles_max)
etha_a_field = ti.field(dtype=ti.f32, shape=num_particles_max)
B_unit_field = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max)


S_a_field = ti.Matrix.field(dimension, dimension, dtype=ti.f32, shape=num_particles_max)
S_b_field = ti.Matrix.field(dimension, dimension, dtype=ti.f32, shape=num_particles_max)


h_smooth = ti.field(dtype=ti.f32, shape=num_particles_max)


dB_dt = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max)


num_actual_particles = ti.field(dtype=ti.i32, shape=())


alpha_visc_p = ti.field(dtype=ti.f32, shape=num_particles_max)
beta_visc_p = ti.field(dtype=ti.f32, shape=num_particles_max)


electron_charge = -1.602176634e-19




effective_electron_charge_per_particle = electron_charge * 1e25



biot_savart_scale_factor = 6.4e6


@ti.func
def W(r, h):
    """SPH Cubic Spline 커널 함수."""
    q = r / h
    alpha = sigma_kernel / (h**dimension)
    result = 0.0
    if 0 <= q < 1:
        result = alpha * (1.0 - 1.5 * q**2 + 0.75 * q**3)
    elif 1 <= q < 2:
        result = alpha * (0.25 * (2.0 - q)**3)
    return result

@ti.func
def grad_W(r_vec, r, h):
    """SPH Cubic Spline 커널 함수의 기울기."""
    q = r / h
    alpha = sigma_kernel / (h**dimension)
    gradient_result = ti.Vector([0.0, 0.0])
    if r < 1e-9:
        pass
    else:
        dw_dq = 0.0
        if 0 <= q < 1:
            dw_dq = alpha * (-3.0 * q + 2.25 * q**2)
        elif 1 <= q < 2:
            dw_dq = alpha * (-0.75 * (2.0 - q)**2)
        gradient_result = dw_dq * r_vec / (r * h)
    return gradient_result

@ti.func
def get_dipole_B_field(p_pos, center, moment_dir, moment_strength_val):
    """
    2D 자기 쌍극자 자기장(B-field)을 계산합니다.
    """
    r_vec = p_pos - center
    r_norm = r_vec.norm()
    result_B_field = ti.Vector([0.0, 0.0])
    if moment_strength_val >= 1e-9 and r_norm >= 1e-5:
        dx = p_pos.x - center.x
        dy = p_pos.y - center.y
        r_sq = r_norm * r_norm
        r_pow_5 = r_sq * r_sq * r_norm
        base_Bx = (3.0 * dx * dy) / r_pow_5
        base_By = (3.0 * dy * dy - r_sq) / r_pow_5
        angle = ti.atan2(moment_dir.y, moment_dir.x)
        cos_a = ti.cos(angle)
        sin_a = ti.sin(angle)
        rotated_B_x = base_Bx * cos_a - base_By * sin_a
        rotated_B_y = base_Bx * sin_a + base_By * cos_a
        result_B_field = moment_strength_val * ti.Vector([rotated_B_x, rotated_B_y])
    return result_B_field

@ti.func
def is_position_valid(candidate_pos, current_idx, current_num_particles, min_dist_sq_val, free_zone_radius_val, min_placement_radius_val):
    """
    주어진 위치가 다음 조건을 만족하는지 확인합니다:
    1. 자기장 없는 구역 (free_zone_radius_val) 밖에 있을 것.
    2. 최소 배치 반경 (min_placement_radius_val)보다 멀리 있을 것.
    3. 다른 기존 입자들과 최소 거리 이상 떨어져 있을 것.
    """
    is_valid_flag = True

    dist_from_magnet_center = (candidate_pos - magnet_center).norm()


    if dist_from_magnet_center < free_zone_radius_val:
        is_valid_flag = False


    if is_valid_flag and dist_from_magnet_center < min_placement_radius_val:
        is_valid_flag = False


    for j in range(current_num_particles):
        if is_valid_flag:
            if j == current_idx:
                pass
            else:
                dist_sq = (candidate_pos - pos[j]).dot(candidate_pos - pos[j])
                if dist_sq < min_dist_sq_val:
                    is_valid_flag = False

    return is_valid_flag

@ti.kernel
def compute_electron_initial_B_field(num_el_core: ti.i32, num_el_halo: ti.i32, initial_idx_offset: ti.i32, charge_per_particle: ti.f32, bs_scale_factor: ti.f32):
    """
    전자 코어 및 전자 헤일로 입자들의 초기 자기장을 비오-사바르 법칙을 이용하여 계산합니다.
    (각 전자 입자의 위치에서 다른 모든 전자 입자들의 전류 요소에 의한 자기장 합산)
    """
    for i in range(num_el_core + num_el_halo):
        current_electron_idx = initial_idx_offset + i

        target_B = ti.Vector([0.0, 0.0])


        for j in range(num_el_core + num_el_halo):
            if i == j:
                continue

            other_electron_idx = initial_idx_offset + j

            r_vec = pos[current_electron_idx] - pos[other_electron_idx]
            r = r_vec.norm()


            if r > 1e-9:
                effective_current_x = charge_per_particle * vel[other_electron_idx].x
                effective_current_y = charge_per_particle * vel[other_electron_idx].y


                dx = r_vec.x
                dy = r_vec.y
                r_sq = r * r

                if r_sq > 1e-9:

                    dBx_from_cx = - effective_current_x * dy / r_sq

                    dBx_from_cy = effective_current_y * dx / r_sq


                    dBy_from_cx = effective_current_x * dx / r_sq

                    dBy_from_cy = - effective_current_y * dy / r_sq


                    target_B.x += (dBx_from_cx + dBx_from_cy) * bs_scale_factor
                    target_B.y += (dBy_from_cx + dBy_from_cy) * bs_scale_factor



        if is_special_particle_type[current_electron_idx] == PARTICLE_TYPE_ELECTRON_CORE or \
           is_special_particle_type[current_electron_idx] == PARTICLE_TYPE_ELECTRON_HALO:
            B[current_electron_idx] = target_B

            if B[current_electron_idx].norm() > 1e-7:
                B[current_electron_idx] = B[current_electron_idx].normalized() * 1e-7


@ti.kernel
def init_particles_kernel(initial_mag_strength: ti.f32, start_row: ti.i32, end_row: ti.i32, initial_min_radius: ti.f32,
                          num_ions_param: ti.i32, num_electron_core_param: ti.i32, num_electron_halo_param: ti.i32,
                          ion_vel: ti.template(), electron_core_vel: ti.template(),
                          electron_halo_vel_low: ti.template(), electron_halo_vel_high: ti.template(),
                          ion_b_mag: ti.f32,
                          spawn_x_min: ti.f32, spawn_x_max: ti.f32, spawn_y_min: ti.f32, spawn_y_max: ti.f32):
    """
    모든 입자들을 초기화합니다 (육각형 격자 배치 및 자기장 강도에 따른 확률적 배치).
    start_row와 end_row는 상대적인 줄 번호로 사용됩니다.
    initial_min_radius는 중심으로부터 입자가 배치될 최소 거리입니다.
    """
    num_actual_particles[None] = 0
    max_placement_radius = ti.max(domain_max, domain_max) * 0.9



    for i in range(num_ions_param):
        if num_actual_particles[None] < num_particles_max:
            idx = ti.atomic_add(num_actual_particles[None], 1)
            if idx < num_particles_max:
                is_special_particle_type[idx] = PARTICLE_TYPE_ION
                alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                vel[idx] = ion_vel

                pos[idx] = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                      spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                mass[idx] = 1.0 / num_particles_max
                u[idx] = 1.0

                rho[idx] = 5.0 * 1e6
                acc[idx] = ti.Vector([0.0, 0.0])
                P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                h_smooth[idx] = 0.04
                etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                B_unit_field[idx] = ti.Vector([0.0, 0.0])
                S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                dB_dt[idx] = ti.Vector([0.0, 0.0])

                current_B_at_pos_dir = get_dipole_B_field(pos[idx], magnet_center, magnet_moment_direction, initial_mag_strength).normalized()
                if current_B_at_pos_dir.norm() < 1e-9:
                    current_B_at_pos_dir = ti.Vector([1.0, 0.0])
                B[idx] = current_B_at_pos_dir * ion_b_mag
            else:
                num_actual_particles[None] -= 1


    electron_core_start_idx = num_actual_particles[None]
    for i in range(num_electron_core_param):
        if num_actual_particles[None] < num_particles_max:
            idx = ti.atomic_add(num_actual_particles[None], 1)
            if idx < num_particles_max:
                is_special_particle_type[idx] = PARTICLE_TYPE_ELECTRON_CORE
                alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                vel[idx] = electron_core_vel

                pos[idx] = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                      spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                mass[idx] = 1.0 / num_particles_max
                u[idx] = 1.0

                rho[idx] = 5.0 * 1e6
                acc[idx] = ti.Vector([0.0, 0.0])
                P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                h_smooth[idx] = 0.04
                etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                B_unit_field[idx] = ti.Vector([0.0, 0.0])
                S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                dB_dt[idx] = ti.Vector([0.0, 0.0])
                B[idx] = ti.Vector([0.0, 0.0])
            else:
                num_actual_particles[None] -= 1


    electron_halo_start_idx = num_actual_particles[None]
    for i in range(num_electron_halo_param):
        if num_actual_particles[None] < num_particles_max:
            idx = ti.atomic_add(num_actual_particles[None], 1)
            if idx < num_particles_max:
                is_special_particle_type[idx] = PARTICLE_TYPE_ELECTRON_HALO
                alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1

                if ti.random(ti.f32) < 0.5:
                    vel[idx] = electron_halo_vel_low
                else:
                    vel[idx] = electron_halo_vel_high


                pos[idx] = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                      spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                mass[idx] = 1.0 / num_particles_max
                u[idx] = 1.0

                rho[idx] = 5.0 * 1e6
                acc[idx] = ti.Vector([0.0, 0.0])
                P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                h_smooth[idx] = 0.04
                etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                B_unit_field[idx] = ti.Vector([0.0, 0.0])
                S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                dB_dt[idx] = ti.Vector([0.0, 0.0])
                B[idx] = ti.Vector([0.0, 0.0])
            else:
                num_actual_particles[None] -= 1


    hex_rows_max = 1000
    row_height = hex_spacing * ti.sqrt(4.0)

    y_base_for_hex_grid = magnet_center.y

    test_pos_for_max_B = magnet_center + ti.Vector([magnetic_free_zone_radius + 0.1, 0.0])
    max_B_norm_estimate = get_dipole_B_field(test_pos_for_max_B, magnet_center, magnet_moment_direction, initial_mag_strength).norm()
    if max_B_norm_estimate < 1e-9:
        max_B_norm_estimate = 1.0

    for row_idx in range(start_row, end_row + 1):
        for col_idx in range(-hex_rows_max, hex_rows_max + 1):
            if num_actual_particles[None] < num_particles_max:
                x_offset = 0.0
                if row_idx % 2 != 0:
                    x_offset = hex_spacing * 0.5

                p_pos_candidate = ti.Vector([
                    col_idx * hex_spacing + x_offset + magnet_center.x,
                    y_base_for_hex_grid + row_idx * row_height
                ])

                if not (domain_min <= p_pos_candidate.x <= domain_max and \
                                     domain_min <= p_pos_candidate.y <= domain_max):
                    continue

                if not is_position_valid(p_pos_candidate, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, initial_min_radius):
                    continue

                if (p_pos_candidate - magnet_center).norm() > max_placement_radius:
                    continue

                probability_of_acceptance = 0.0

                current_B_at_pos = get_dipole_B_field(p_pos_candidate, magnet_center, magnet_moment_direction, initial_mag_strength)
                current_B_norm = current_B_at_pos.norm()

                if current_B_norm < B_magnitude_threshold:
                    probability_of_acceptance = 0.0
                else:
                    probability_of_acceptance = min(1.0, current_B_norm / max_B_norm_estimate * 0.7)

                if ti.random(ti.f32) < probability_of_acceptance:
                    idx = ti.atomic_add(num_actual_particles[None], 1)

                    if idx < num_particles_max:
                        B_initial_direction = ti.Vector([0.0, 0.0])
                        if current_B_norm >= B_magnitude_threshold:
                            B_initial_direction = current_B_at_pos / current_B_norm

                        is_special_particle_type[idx] = PARTICLE_TYPE_NORMAL
                        alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                        vel[idx] = B_initial_direction * 0.001

                        pos[idx] = p_pos_candidate
                        mass[idx] = 1.0 / num_particles_max
                        u[idx] = 1.0
                        rho[idx] = 0.001
                        acc[idx] = ti.Vector([0.0, 0.0])
                        P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                        h_smooth[idx] = 0.04
                        etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                        B_unit_field[idx] = ti.Vector([0.0, 0.0])
                        S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                        dB_dt[idx] = ti.Vector([0.0, 0.0])
                        current_B = get_dipole_B_field(p_pos_candidate, magnet_center, magnet_moment_direction, initial_mag_strength)
                        B[idx] = current_B
                        if current_B.norm() > 1e-9:
                            B_unit_field[idx] = current_B.normalized()
                        else:
                            B_unit_field[idx] = ti.Vector([0.0, 0.0])
                    else:
                        num_actual_particles[None] -= 1
                


@ti.kernel
def compute_density_only():
    """루프 1: 모든 입자의 밀도만 계산합니다."""
    for i in range(num_actual_particles[None]):
        rho[i] = 0.0

    for i in range(num_actual_particles[None]):

        for j in range(num_actual_particles[None]):
            if i != j:
                r_vec = pos[i] - pos[j]
                r = r_vec.norm()
                h_ij = (h_smooth[i] + h_smooth[j]) / 2.0
                if r < 2.0 * h_ij:
                    density_contribution = mass[j] * W(r, h_ij)

                    target_rho = 1.0
                    if is_special_particle_type[i] == PARTICLE_TYPE_ION or \
                       is_special_particle_type[i] == PARTICLE_TYPE_ELECTRON_CORE or \
                       is_special_particle_type[i] == PARTICLE_TYPE_ELECTRON_HALO:

                        target_rho = 5.0 * 1e6

                    if r < density_cap_activation_distance:
                        if rho[i] + density_contribution > max_density_cap:
                            rho[i] = max_density_cap
                        elif rho[i] + density_contribution > target_rho * 1.5:
                            rho[i] = target_rho * 1.5
                        else:
                            rho[i] += density_contribution
                    else:
                        rho[i] += density_contribution

@ti.kernel
def compute_forces_and_magnetic_effects():
    """루프 2: 압력, 점성, 자기항, dBdt 등 힘 관련 계산과 특수입자 효과를 통합 처리합니다."""
    for i in range(num_actual_particles[None]):
        acc[i] = ti.Vector([0.0, 0.0])
        etha_a_dt_field[i] = 0.0
        dB_dt[i] = ti.Vector([0.0, 0.0])

    for i in range(num_actual_particles[None]):

        if rho[i] > 1e-9:
            P_pressure[i] = (gamma - 1.0) * rho[i] * u[i]
        else:
            P_pressure[i] = 0.0

        P_pressure[i] = min(P_pressure[i], max_pressure_cap)


        B_i_norm_sq = B[i].dot(B[i])
        S_a_field[i] = P_pressure[i] * ti.Matrix.identity(ti.f32, dimension) + \
                                    (B[i].outer_product(B[i]) - 0.5 * B_i_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / mu_0

        acc_pressure_i, acc_visc_i, acc_magnetic_i = ti.Vector([0.0, 0.0]), ti.Vector([0.0, 0.0]), ti.Vector([0.0, 0.0])


        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL:
            total_B = ti.Vector([0.0, 0.0])
            for j in range(num_actual_particles[None]):
                if is_special_particle_type[j] != PARTICLE_TYPE_NORMAL:
                    r_vec = pos[i] - pos[j]
                    r = r_vec.norm()
                    if r > 1e-6:
                        current_j = effective_electron_charge_per_particle * vel[j]

                        dB = biot_savart_scale_factor * ti.Vector([-current_j.y, current_j.x]) / (r * r)
                        total_B += dB
            B[i] += total_B


        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL:
            total_B = ti.Vector([0.0, 0.0])
            for j in range(num_actual_particles[None]):
                if is_special_particle_type[j] != PARTICLE_TYPE_NORMAL:
                    r_vec = pos[i] - pos[j]
                    r = r_vec.norm()
                    if r > 1e-6:
                        current_j = effective_electron_charge_per_particle * vel[j]
                        dB = biot_savart_scale_factor * ti.Vector([-current_j.y, current_j.x]) / (r * r)
                        total_B += dB
            B[i] += total_B


        for j in range(num_actual_particles[None]):
            if i == j:
                continue

            r_vec = pos[i] - pos[j]
            r = r_vec.norm()
            h_ij = (h_smooth[i] + h_smooth[j]) / 2.0

            epsilon_stability = 1e-6 * h_ij
            r_effective_for_stability = r + epsilon_stability

            if r < 1e-9 or r > 2.0 * h_ij:
                continue

            grad_Wij = grad_W(r_vec, r, h_ij)

            if rho[i] > 1e-9 and rho[j] > 1e-9:

                acc_pressure_i += -mass[j] * (P_pressure[i] / rho[i]**2 + P_pressure[j] / rho[j]**2) * grad_Wij


                v_dot_r = (vel[i] - vel[j]).dot(r_vec)

                c_avg = 0.0
                Pi_ij = 0.0

                if v_dot_r < 0:
                    h_avg = (h_smooth[i] + h_smooth[j]) / 2.0
                    rho_avg = (rho[i] + rho[j]) / 2.0
                    c_avg = ti.sqrt(gamma * P_pressure[i] / rho[i]) if rho[i] > 1e-9 else 0.0
                    mu_ij = h_avg * v_dot_r / (r**2 + h_avg**2 * 0.01)
                    Pi_ij = (-alpha_visc_p[i] * c_avg * mu_ij + beta_visc_p[i] * mu_ij**2) / rho_avg

                acc_visc_i += -mass[j] * Pi_ij * grad_Wij


                if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL or \
                   is_special_particle_type[i] == PARTICLE_TYPE_ION:

                    B_j_norm_sq = B[j].dot(B[j])
                    S_b_temp = P_pressure[j] * ti.Matrix.identity(ti.f32, dimension) + \
                                             (B[j].outer_product(B[j]) - 0.5 * B_j_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / mu_0
                    acc_magnetic_i += -mass[j] * ((S_a_field[i] / rho[i]**2 + S_b_temp / rho[j]**2) @ grad_Wij)


                etha_a_dt_field[i] += mass[j] * (P_pressure[i]/rho[i]**2 + P_pressure[j]/rho[j]**2 - Pi_ij) * (vel[i] - vel[j]).dot(grad_Wij)



                if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL or \
                   is_special_particle_type[i] == PARTICLE_TYPE_ION:

                    F_z_j = vel[j].x * B[j].y - vel[j].y * B[j].x
                    dB_dt_x_advection_j_contrib = mass[j] * F_z_j * grad_Wij.y / rho[j]
                    dB_dt_y_advection_j_contrib = -mass[j] * F_z_j * grad_Wij.x / rho[j]
                    dB_dt[i].x += dB_dt_x_advection_j_contrib
                    dB_dt[i].y += dB_dt_y_advection_j_contrib


                    diffusion_term_contrib = ti.Vector([0.0, 0.0])
                    if r_effective_for_stability > 1e-9:
                        diffusion_term_contrib = mass[j] * (B[j] - B[i]) * (r_vec.dot(grad_Wij)) / (rho[i] * r_effective_for_stability**2)
                    dB_dt[i] += magnetic_diffusivity * diffusion_term_contrib


        acc[i] = acc_pressure_i + acc_visc_i + acc_magnetic_i

@ti.kernel
def update_particles(dt: ti.f32, current_initial_placement_min_radius: ti.f32):
    """
    입자 속성 (내부 에너지, 자기장, 속도, 위치)를 업데이트하고 경계 조건을 처리합니다.
    current_initial_placement_min_radius는 현재 적용될 최소 배치 반경입니다.
    """
    for i in range(num_actual_particles[None]):
        if rho[i] > 1e-9:
            u[i] += (etha_a_dt_field[i] / rho[i]) * dt



        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL or \
           is_special_particle_type[i] == PARTICLE_TYPE_ION:
            B[i] += dB_dt[i] * dt

            B_norm_current = B[i].norm()
            if B_norm_current < B_magnitude_threshold:
               B[i] = ti.Vector([0.0, 0.0])

        ae_k_field[i] = 0.5 * mass[i] * vel[i].dot(vel[i])

        B_norm_i = B[i].norm()
        if B_norm_i > 1e-9:
            B_unit_field[i] = B[i] / B_norm_i
        else:
            B_unit_field[i] = ti.Vector([0.0, 0.0])

        vel[i] += acc[i] * dt


        velocity_magnitude = vel[i].norm()
        if velocity_magnitude > max_velocity_cap:
            vel[i] = vel[i].normalized() * max_velocity_cap

        pos[i] += vel[i] * dt


        dist_from_center = (pos[i] - magnet_center).norm()
        effective_inner_boundary_for_reflection = ti.max(magnetic_free_zone_radius, current_initial_placement_min_radius)

        if dist_from_center < effective_inner_boundary_for_reflection:
            new_pos_found = False
            attempts = 0
            max_reflection_attempts = 50

            original_pos = pos[i]
            original_vel = vel[i]

            while attempts < max_reflection_attempts and not new_pos_found:
                direction_from_center = original_pos - magnet_center
                if direction_from_center.norm() < 1e-9:
                    direction_from_center = ti.Vector([ti.random(ti.f32)*2-1, ti.random(ti.f32)*2-1]).normalized()

                candidate_pos = magnet_center + direction_from_center.normalized() * \
                                ti.random(ti.f32) * (reposition_radius_max - effective_inner_boundary_for_reflection) + effective_inner_boundary_for_reflection

                if is_position_valid(candidate_pos, i, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    pos[i] = candidate_pos
                    new_pos_found = True

                attempts += 1

            if not new_pos_found:
                pos[i] = magnet_center + (original_pos - magnet_center).normalized() * effective_inner_boundary_for_reflection * 1.05
                vel[i] = -vel[i] * 0.8
                u[i] *= 0.9
            else:
                reflect_direction = (pos[i] - magnet_center).normalized()
                vel_dot_reflect = original_vel.dot(reflect_direction)
                vel[i] = original_vel - 2 * vel_dot_reflect * reflect_direction
                vel[i] *= 0.8
                u[i] *= 0.9



        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL and \
           not (domain_min <= pos[i].x <= domain_max and domain_min <= pos[i].y <= domain_max):

            new_pos_found = False
            attempts = 0
            max_reposition_attempts = 100

            while attempts < max_reposition_attempts and not new_pos_found:
                angle = ti.random(ti.f32) * 2 * math.pi
                effective_reposition_min_radius = ti.max(reposition_radius_min, current_initial_placement_min_radius)

                radius = ti.random(ti.f32) * (reposition_radius_max - effective_reposition_min_radius) + effective_reposition_min_radius

                candidate_pos = magnet_center + ti.Vector([radius * ti.cos(angle), radius * ti.sin(angle)])

                if is_position_valid(candidate_pos, i, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    pos[i] = candidate_pos
                    vel[i] = ti.Vector([ti.random(ti.f32) * 2 - 1, ti.random(ti.f32) * 2 - 1]) * 0.005
                    u[i] = 1.0
                    rho[i] = 1.0
                    B[i] = get_dipole_B_field(pos[i], magnet_center, magnet_moment_direction, magnet_moment_strength)
                    new_pos_found = True

                attempts += 1

            if not new_pos_found:
                pos[i].x = ti.max(domain_min, ti.min(domain_max, pos[i].x))
                pos[i].y = ti.max(domain_min, ti.min(domain_max, pos[i].y))
                vel[i] *= -0.5

@ti.kernel
def count_normal_particles_in_radius(check_radius: ti.f32) -> ti.i32:
    """
    중앙 영역(magnet_center) 내의 일반 입자 수를 세어 반환합니다.
    """
    count = 0
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL:
            dist_from_center = (pos[i] - magnet_center).norm()
            if dist_from_center < check_radius:
                count += 1
    return count

@ti.kernel
def repopulate_particles_kernel(initial_mag_strength: ti.f32, current_initial_placement_min_radius: ti.f32,
                                desired_count: ti.i32, max_attempts_per_particle: ti.i32,
                                max_placement_rad_rep: ti.f32):
    """
    중앙 영역에 일반 입자가 부족할 경우, 새로 입자를 생성하여 배치합니다.
    """
    current_normal_particle_count = 0
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL and \
           (pos[i] - magnet_center).norm() < repopulation_check_radius:
            current_normal_particle_count += 1

    num_to_add = desired_count - current_normal_particle_count

    if num_to_add > 0:
        test_pos_for_max_B = magnet_center + ti.Vector([magnetic_free_zone_radius + 0.1, 0.0])
        max_B_norm_estimate = get_dipole_B_field(test_pos_for_max_B, magnet_center, magnet_moment_direction, initial_mag_strength).norm()
        if max_B_norm_estimate < 1e-9:
            max_B_norm_estimate = 1.0

        for _ in range(num_to_add):
            if num_actual_particles[None] < num_particles_max:
                found_position = False
                attempts = 0
                while attempts < max_attempts_per_particle and not found_position:

                    angle = ti.random(ti.f32) * 2 * math.pi
                    radius = ti.random(ti.f32) * (max_placement_rad_rep - current_initial_placement_min_radius) + current_initial_placement_min_radius
                    candidate_pos = magnet_center + ti.Vector([radius * ti.cos(angle), radius * ti.sin(angle)])

                    if not (domain_min <= candidate_pos.x <= domain_max and \
                                         domain_min <= candidate_pos.y <= domain_max):
                        attempts += 1
                        continue

                    if is_position_valid(candidate_pos, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                        idx = ti.atomic_add(num_actual_particles[None], 1)

                        if idx < num_particles_max:
                            B_initial_direction = ti.Vector([0.0, 0.0])
                            current_B_at_pos = get_dipole_B_field(candidate_pos, magnet_center, magnet_moment_direction, initial_mag_strength)
                            current_B_norm = current_B_at_pos.norm()

                            if current_B_norm >= B_magnitude_threshold:
                                B_initial_direction = current_B_at_pos / current_B_norm

                            is_special_particle_type[idx] = PARTICLE_TYPE_NORMAL
                            alpha_visc_p[idx], beta_visc_p[idx] = 0.5, 0.5
                            vel[idx] = B_initial_direction * 0.001
                            pos[idx] = candidate_pos
                            mass[idx] = 1.0 / num_particles_max
                            u[idx] = 1.0
                            rho[idx] = 1.0
                            acc[idx] = ti.Vector([0.0, 0.0])
                            P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                            h_smooth[idx] = 0.04
                            etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                            B_unit_field[idx] = ti.Vector([0.0, 0.0])
                            S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                            dB_dt[idx] = ti.Vector([0.0, 0.0])
                            B[idx] = get_dipole_B_field(candidate_pos, magnet_center, magnet_moment_direction, initial_mag_strength)

                            found_position = True
                        else:
                            num_actual_particles[None] -= 1
                    attempts += 1

@ti.kernel
def count_current_special_particles_in_zone() -> ti.i32:
    """
    특수 입자 재배치 구역 내의 특수 입자 수를 세어 반환합니다.
    """
    count = 0
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] != PARTICLE_TYPE_NORMAL:
            dist = (pos[i] - special_particle_repop_check_center).norm()
            if dist < special_particle_repop_check_radius:
                count += 1
    return count

@ti.kernel
def repopulate_special_particles_kernel(
    initial_mag_strength: ti.f32,
    current_initial_placement_min_radius: ti.f32,
    desired_ions: ti.i32,
    desired_electron_core: ti.i32,
    desired_electron_halo: ti.i32,
    max_attempts_per_particle: ti.i32,
    spawn_x_min: ti.f32, spawn_x_max: ti.f32, spawn_y_min: ti.f32, spawn_y_max: ti.f32,
    ion_vel: ti.template(),
    electron_core_vel: ti.template(),
    electron_halo_vel_low: ti.template(),
    electron_halo_vel_high: ti.template(),
    ion_b_mag: ti.f32
):
    """
    특수 입자 스폰 영역 내의 특수 입자 수가 부족할 경우, 새로 입자를 생성하여 배치합니다.
    """
    current_ion_count = 0
    current_electron_core_count = 0
    current_electron_halo_count = 0

    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_ION:
            current_ion_count += 1
        elif is_special_particle_type[i] == PARTICLE_TYPE_ELECTRON_CORE:
            current_electron_core_count += 1
        elif is_special_particle_type[i] == PARTICLE_TYPE_ELECTRON_HALO:
            current_electron_halo_count += 1


    num_ions_to_add = desired_ions - current_ion_count
    for _ in range(num_ions_to_add):
        if num_actual_particles[None] < num_particles_max:
            found_position = False
            attempts = 0
            while attempts < max_attempts_per_particle and not found_position:
                candidate_pos = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                           spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])

                if is_position_valid(candidate_pos, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    idx = ti.atomic_add(num_actual_particles[None], 1)
                    if idx < num_particles_max:
                        is_special_particle_type[idx] = PARTICLE_TYPE_ION
                        alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                        vel[idx] = ion_vel
                        pos[idx] = candidate_pos
                        mass[idx] = 1.0 / num_particles_max
                        u[idx] = 1.0
                        rho[idx] = 5.0 * 1e6
                        acc[idx] = ti.Vector([0.0, 0.0])
                        P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                        h_smooth[idx] = 0.04
                        etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                        B_unit_field[idx] = ti.Vector([0.0, 0.0])
                        S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                        dB_dt[idx] = ti.Vector([0.0, 0.0])
                        current_B_at_pos_dir = get_dipole_B_field(pos[idx], magnet_center, magnet_moment_direction, initial_mag_strength).normalized()
                        if current_B_at_pos_dir.norm() < 1e-9:
                            current_B_at_pos_dir = ti.Vector([1.0, 0.0])
                        B[idx] = current_B_at_pos_dir * ion_b_mag
                        found_position = True
                    else:
                        num_actual_particles[None] -= 1
                attempts += 1


    num_electron_core_to_add = desired_electron_core - current_electron_core_count
    for _ in range(num_electron_core_to_add):
        if num_actual_particles[None] < num_particles_max:
            found_position = False
            attempts = 0
            while attempts < max_attempts_per_particle and not found_position:
                candidate_pos = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                           spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                if is_position_valid(candidate_pos, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    idx = ti.atomic_add(num_actual_particles[None], 1)
                    if idx < num_particles_max:
                        is_special_particle_type[idx] = PARTICLE_TYPE_ELECTRON_CORE
                        alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                        vel[idx] = electron_core_vel
                        pos[idx] = candidate_pos
                        mass[idx] = 1.0 / num_particles_max
                        u[idx] = 1.0
                        rho[idx] = 5.0 * 1e6
                        acc[idx] = ti.Vector([0.0, 0.0])
                        P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                        h_smooth[idx] = 0.04
                        etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                        B_unit_field[idx] = ti.Vector([0.0, 0.0])
                        S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                        dB_dt[idx] = ti.Vector([0.0, 0.0])
                        B[idx] = ti.Vector([0.0, 0.0])
                        found_position = True
                    else:
                        num_actual_particles[None] -= 1
                attempts += 1


    num_electron_halo_to_add = desired_electron_halo - current_electron_halo_count
    for _ in range(num_electron_halo_to_add):
        if num_actual_particles[None] < num_particles_max:
            found_position = False
            attempts = 0
            while attempts < max_attempts_per_particle and not found_position:
                candidate_pos = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                           spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                if is_position_valid(candidate_pos, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    idx = ti.atomic_add(num_actual_particles[None], 1)
                    if idx < num_particles_max:
                        is_special_particle_type[idx] = PARTICLE_TYPE_ELECTRON_HALO
                        alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                        if ti.random(ti.f32) < 0.5:
                            vel[idx] = electron_halo_vel_low
                        else:
                            vel[idx] = electron_halo_vel_high
                        pos[idx] = candidate_pos
                        mass[idx] = 1.0 / num_particles_max
                        u[idx] = 1.0
                        rho[idx] = 5.0 * 1e6
                        acc[idx] = ti.Vector([0.0, 0.0])
                        P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                        h_smooth[idx] = 0.04
                        etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                        B_unit_field[idx] = ti.Vector([0.0, 0.0])
                        S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                        dB_dt[idx] = ti.Vector([0.0, 0.0])
                        B[idx] = ti.Vector([0.0, 0.0])
                        found_position = True
                    else:
                        num_actual_particles[None] -= 1
                attempts += 1


@ti.kernel
def visualize_magnetic_field_grid_kernel(current_mag_strength: ti.f32):
    """전반적인 자기장 흐름 시각화를 위해 자기장을 균일한 격자로 보간합니다."""
    grid_spacing = (domain_max - domain_min) / num_grid_points
    for i_grid, j_grid in ti.ndrange(num_grid_points, num_grid_points):
        idx = i_grid * num_grid_points + j_grid
        p_grid_pos = ti.Vector([i_grid * grid_spacing + grid_spacing / 2.0 + domain_min,
                                     j_grid * grid_spacing + grid_spacing / 2.0 + domain_min])
        grid_pos[idx] = p_grid_pos
        interpolated_B, sum_weights = ti.Vector([0.0, 0.0]), 0.0


        if (p_grid_pos - magnet_center).norm() < magnetic_free_zone_radius:
            grid_B_interpolated[idx] = ti.Vector([0.0, 0.0])
            continue

        for i_particle in range(num_actual_particles[None]):
            r_vec = p_grid_pos - pos[i_particle]
            r = r_vec.norm()
            h_particle = h_smooth[i_particle]
            if r < 2.0 * h_particle:

                if is_special_particle_type[i_particle] == PARTICLE_TYPE_NORMAL or \
                   is_special_particle_type[i_particle] == PARTICLE_TYPE_ION or \
                   is_special_particle_type[i_particle] == PARTICLE_TYPE_ELECTRON_CORE or \
                   is_special_particle_type[i_particle] == PARTICLE_TYPE_ELECTRON_HALO:
                    weight = W(r, h_particle)
                    interpolated_B += B[i_particle] * weight
                    sum_weights += weight
        if sum_weights > 1e-9:
            grid_B_interpolated[idx] = interpolated_B / sum_weights
        else:
            grid_B_interpolated[idx] = ti.Vector([0.0, 0.0])

@ti.kernel
def compute_static_dipole_field_kernel(current_mag_strength: ti.f32):
    """순수한 영구 자석(자기 쌍극자)의 자기장을 격자점에 계산합니다."""
    grid_spacing = (domain_max - domain_min) / num_grid_points
    for i_grid, j_grid in ti.ndrange(num_grid_points, num_grid_points):
        idx = i_grid * num_grid_points + j_grid
        p_grid_pos = ti.Vector([i_grid * grid_spacing + grid_spacing / 2.0 + domain_min,
                                     j_grid * grid_spacing + grid_spacing / 2.0 + domain_min])


        if (p_grid_pos - magnet_center).norm() < magnetic_free_zone_radius:
            grid_B_dipole_only[idx] = ti.Vector([0.0, 0.0])
            continue

        grid_B_dipole_only[idx] = get_dipole_B_field(p_grid_pos, magnet_center, magnet_moment_direction, current_mag_strength)


def main_simulation_loop():
    global magnet_moment_strength
    global initial_placement_min_radius

    initial_placement_min_radius = 1.5


    init_particles_kernel(initial_magnet_moment_strength, -205, 205, initial_placement_min_radius,
                          num_ions, num_electron_core, num_electron_halo,
                          ion_initial_velocity, electron_core_initial_velocity,
                          electron_halo_initial_velocity_low, electron_halo_initial_velocity_high,
                          ion_initial_B_magnitude,
                          special_particle_spawn_x_min, special_particle_spawn_x_max,
                          special_particle_spawn_y_min, special_particle_spawn_y_max)

    print(f"Number of actual particles initialized: {num_actual_particles[None]}")



    electron_core_start_idx = num_ions
    compute_electron_initial_B_field(num_electron_core, num_electron_halo, electron_core_start_idx,
                                     effective_electron_charge_per_particle, biot_savart_scale_factor)


    window_width = 800
    window_height = 800
    window = ti.ui.Window("Taichi SPH MHD Simulation with Multi-Type Special Particles", (window_width, window_height), vsync=True)
    canvas = window.get_canvas()

    display_view_min = domain_min * 1.1
    display_view_max = domain_max * 1.1
    display_range = display_view_max - display_view_min

    def rotate_vector(vector, angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        vec_np = np.asarray(vector)
        x = vec_np[0] * c - vec_np[1] * s
        y = vec_np[0] * s + vec_np[1] * c
        return np.array([x, y], dtype=np.float32)


    earth_radius_sim_units = magnetic_free_zone_radius
    earth_color = (0.2, 0.4, 0.8)


    num_segments = 50
    earth_vertices = ti.Vector.field(dimension, dtype=ti.f32, shape=num_segments * 3)

    @ti.kernel
    def generate_earth_vertices_kernel(center_x: ti.f32, center_y: ti.f32, radius: ti.f32, num_seg: ti.i32):
        for i in range(num_seg):
            angle1 = 2 * math.pi * i / num_seg
            angle2 = 2 * math.pi * (i + 1) / num_seg


            earth_vertices[i * 3] = ti.Vector([center_x, center_y])

            earth_vertices[i * 3 + 1] = ti.Vector([center_x + radius * ti.cos(angle1), center_y + radius * ti.sin(angle1)])

            earth_vertices[i * 3 + 2] = ti.Vector([center_x + radius * ti.cos(angle2), center_y + radius * ti.sin(angle2)])

    frame_count = 0
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
        compute_density_only()
        compute_forces_and_magnetic_effects()
        update_particles(dt, initial_placement_min_radius)
        visualize_magnetic_field_grid_kernel(magnet_moment_strength)
        compute_static_dipole_field_kernel(magnet_moment_strength)


        current_normal_particles_in_center = count_normal_particles_in_radius(repopulation_check_radius)
        if current_normal_particles_in_center < num_desired_particles_in_center:

            repopulate_particles_kernel(initial_magnet_moment_strength, initial_placement_min_radius,
                                        num_desired_particles_in_center, repopulation_search_attempts,
                                        reposition_radius_max)



        current_total_special_particles_in_zone = count_current_special_particles_in_zone()
        if current_total_special_particles_in_zone < num_desired_total_special_particles:
            repopulate_special_particles_kernel(initial_magnet_moment_strength, initial_placement_min_radius,
                                                num_ions, num_electron_core, num_electron_halo,
                                                special_repopulation_search_attempts,
                                                special_particle_spawn_x_min, special_particle_spawn_x_max,
                                                special_particle_spawn_y_min, special_particle_spawn_y_max,
                                                ion_initial_velocity, electron_core_initial_velocity,
                                                electron_halo_initial_velocity_low, electron_halo_initial_velocity_high,
                                                ion_initial_B_magnitude)



        canvas.set_background_color((0.0, 0.0, 0.0))


        earth_center_disp_numpy = (magnet_center.to_numpy() - display_view_min) / display_range
        earth_radius_disp_numpy = earth_radius_sim_units / display_range


        generate_earth_vertices_kernel(earth_center_disp_numpy[0], earth_center_disp_numpy[1], earth_radius_disp_numpy, num_segments)


        canvas.triangles(earth_vertices, color=earth_color)


        pos_np = pos.to_numpy()[:num_actual_particles[None]]
        is_particle_type_np = is_special_particle_type.to_numpy()[:num_actual_particles[None]]

        display_pos_np = (pos_np - display_view_min) / display_range

        normal_particles_pos = []
        ion_particles_pos = []
        electron_core_particles_pos = []
        electron_halo_particles_pos = []

        for i in range(num_actual_particles[None]):
            if is_particle_type_np[i] == PARTICLE_TYPE_NORMAL:
                normal_particles_pos.append(display_pos_np[i])
            elif is_particle_type_np[i] == PARTICLE_TYPE_ION:
                ion_particles_pos.append(display_pos_np[i])
            elif is_particle_type_np[i] == PARTICLE_TYPE_ELECTRON_CORE:
                electron_core_particles_pos.append(display_pos_np[i])
            elif is_particle_type_np[i] == PARTICLE_TYPE_ELECTRON_HALO:
                electron_halo_particles_pos.append(display_pos_np[i])

        if len(normal_particles_pos) > 0:
            temp_normal_circles_field = ti.Vector.field(dimension, dtype=ti.f32, shape=len(normal_particles_pos))
            temp_normal_circles_field.from_numpy(np.array(normal_particles_pos, dtype=np.float32))
            canvas.circles(temp_normal_circles_field, radius=0.03 / display_range, color=(0.9, 0.9, 0.9))

        if len(ion_particles_pos) > 0:
            temp_ion_circles_field = ti.Vector.field(dimension, dtype=ti.f32, shape=len(ion_particles_pos))
            temp_ion_circles_field.from_numpy(np.array(ion_particles_pos, dtype=np.float32))
            canvas.circles(temp_ion_circles_field, radius=0.05 / display_range, color=(1.0, 0.0, 0.0))

        if len(electron_core_particles_pos) > 0:
            temp_electron_core_circles_field = ti.Vector.field(dimension, dtype=ti.f32, shape=len(electron_core_particles_pos))
            temp_electron_core_circles_field.from_numpy(np.array(electron_core_particles_pos, dtype=np.float32))
            canvas.circles(temp_electron_core_circles_field, radius=0.05 / display_range, color=(1.0, 0.0, 0.0))

        if len(electron_halo_particles_pos) > 0:
            temp_electron_halo_circles_field = ti.Vector.field(dimension, dtype=ti.f32, shape=len(electron_halo_particles_pos))
            temp_electron_halo_circles_field.from_numpy(np.array(electron_halo_particles_pos, dtype=np.float32))
            canvas.circles(temp_electron_halo_circles_field, radius=0.05 / display_range, color=(1.0, 0.0, 0.0))


        grid_pos_np = grid_pos.to_numpy()
        grid_pos_np_disp = (grid_pos_np - display_view_min) / display_range

        arrowhead_angle_rad = np.pi / 6


        B_unit_field_np = B_unit_field.to_numpy()[:num_actual_particles[None]]
        particle_magnetic_arrow_vertices = np.zeros((num_actual_particles[None] * 3 * 2, dimension), dtype=np.float32)
        particle_arrow_segment_count = 0
        fixed_arrowhead_len_particle = 0.004 / display_range

        for i in range(num_actual_particles[None]):


            if is_particle_type_np[i] in [PARTICLE_TYPE_ION, PARTICLE_TYPE_NORMAL, PARTICLE_TYPE_ELECTRON_CORE, PARTICLE_TYPE_ELECTRON_HALO]:
                if np.linalg.norm(B_unit_field_np[i]) > B_magnitude_threshold:
                    p_start_disp = display_pos_np[i]

                    scale_factor = magnetic_field_scale * 0.5 / display_range
                    if is_particle_type_np[i] == PARTICLE_TYPE_ION:
                        scale_factor = magnetic_field_scale * 1.0 / display_range
                    elif is_particle_type_np[i] == PARTICLE_TYPE_ELECTRON_CORE or \
                         is_particle_type_np[i] == PARTICLE_TYPE_ELECTRON_HALO:
                        scale_factor = magnetic_field_scale * 0.8 / display_range

                    direction_vec_disp = B_unit_field_np[i] * scale_factor
                    p_end_disp = p_start_disp + direction_vec_disp

                    line_norm = np.linalg.norm(direction_vec_disp)
                    if line_norm > 1e-9:

                        particle_magnetic_arrow_vertices[particle_arrow_segment_count * 2] = p_start_disp
                        particle_magnetic_arrow_vertices[particle_arrow_segment_count * 2 + 1] = p_end_disp
                        particle_arrow_segment_count += 1


                        current_arrowhead_len = min(fixed_arrowhead_len_particle, line_norm * 0.4)
                        v_back = -direction_vec_disp / line_norm
                        arrow_v1_dir = rotate_vector(v_back, arrowhead_angle_rad)
                        arrow_p1 = p_end_disp + arrow_v1_dir * current_arrowhead_len
                        particle_magnetic_arrow_vertices[particle_arrow_segment_count * 2] = arrow_p1
                        particle_magnetic_arrow_vertices[particle_arrow_segment_count * 2 + 1] = p_end_disp
                        particle_arrow_segment_count += 1

                        arrow_v2_dir = rotate_vector(v_back, -arrowhead_angle_rad)
                        arrow_p2 = p_end_disp + arrow_v2_dir * current_arrowhead_len
                        particle_magnetic_arrow_vertices[particle_arrow_segment_count * 2] = arrow_p2
                        particle_magnetic_arrow_vertices[particle_arrow_segment_count * 2 + 1] = p_end_disp
                        particle_arrow_segment_count += 1
        
        if particle_arrow_segment_count > 0:
            temp_particle_magnetic_arrows_field = ti.Vector.field(dimension, dtype=ti.f32, shape=particle_arrow_segment_count * 2)
            temp_particle_magnetic_arrows_field.from_numpy(particle_magnetic_arrow_vertices[:particle_arrow_segment_count * 2])

            for type_id, color in [
                (PARTICLE_TYPE_ION, (1.0, 0.2, 0.2)),
                (PARTICLE_TYPE_ELECTRON_CORE, (1.0, 0.2, 0.2)),
                (PARTICLE_TYPE_ELECTRON_HALO, (1.0, 0.2, 0.2)),
                (PARTICLE_TYPE_NORMAL, (0.6, 0.6, 1.0)),
            ]:
                particle_arrow_vertices = []
                for i in range(num_actual_particles[None]):
                    if is_particle_type_np[i] == type_id and np.linalg.norm(B_unit_field_np[i]) > B_magnitude_threshold:
                        p_start_disp = display_pos_np[i]
                        scale_factor = magnetic_field_scale * 0.5 / display_range
                        direction_vec_disp = B_unit_field_np[i] * scale_factor
                        p_end_disp = p_start_disp + direction_vec_disp
                        line_norm = np.linalg.norm(direction_vec_disp)
                        if line_norm > 1e-9:
                            particle_arrow_vertices.append(p_start_disp)
                            particle_arrow_vertices.append(p_end_disp)

                            v_back = -direction_vec_disp / line_norm
                            current_arrowhead_len = 0.004 / display_range
                            for angle_rad in [np.pi / 6, -np.pi / 6]:
                                arrow_dir = rotate_vector(v_back, angle_rad)
                                arrow_p = p_end_disp + arrow_dir * current_arrowhead_len
                                particle_arrow_vertices.append(arrow_p)
                                particle_arrow_vertices.append(p_end_disp)

                if len(particle_arrow_vertices) > 0:
                    arr = np.array(particle_arrow_vertices, dtype=np.float32)
                    field = ti.Vector.field(dimension, dtype=ti.f32, shape=arr.shape[0])
                    field.from_numpy(arr)
                    canvas.lines(field, color=color, width=0.001)



        grid_B_interpolated_np = grid_B_interpolated.to_numpy()
        grid_arrow_vertices = np.zeros((num_grid_points * num_grid_points * 3 * 2, dimension), dtype=np.float32)
        grid_arrow_segment_count = 0
        fixed_arrowhead_len_grid = 0.003 / display_range

        for i_grid in range(num_grid_points * num_grid_points):
            p_start_disp = grid_pos_np_disp[i_grid]
            current_B_norm = np.linalg.norm(grid_B_interpolated_np[i_grid])

            if current_B_norm > B_magnitude_threshold:

                B_normalized_disp = (grid_B_interpolated_np[i_grid] / current_B_norm) * magnetic_field_scale * 0.2 / display_range

                p_end_disp = p_start_disp + B_normalized_disp
                line_norm = np.linalg.norm(B_normalized_disp)

                if line_norm > 1e-9:

                    grid_arrow_vertices[grid_arrow_segment_count * 2] = p_start_disp
                    grid_arrow_vertices[grid_arrow_segment_count * 2 + 1] = p_end_disp
                    grid_arrow_segment_count += 1


                    current_arrowhead_len = min(fixed_arrowhead_len_grid, line_norm * 0.4)
                    v_back = -B_normalized_disp / line_norm
                    arrow_v1_dir = rotate_vector(v_back, arrowhead_angle_rad)
                    arrow_p1 = p_end_disp + arrow_v1_dir * current_arrowhead_len
                    grid_arrow_vertices[grid_arrow_segment_count * 2] = arrow_p1
                    grid_arrow_vertices[grid_arrow_segment_count * 2 + 1] = p_end_disp
                    grid_arrow_segment_count += 1
                    arrow_v2_dir = rotate_vector(v_back, -arrowhead_angle_rad)
                    arrow_p2 = p_end_disp + arrow_v2_dir * current_arrowhead_len
                    grid_arrow_vertices[grid_arrow_segment_count * 2] = arrow_p2
                    grid_arrow_vertices[grid_arrow_segment_count * 2 + 1] = p_end_disp
                    grid_arrow_segment_count += 1

        if grid_arrow_segment_count > 0:
            temp_grid_arrows_field = ti.Vector.field(dimension, dtype=ti.f32, shape=grid_arrow_segment_count * 2)
            temp_grid_arrows_field.from_numpy(grid_arrow_vertices[:grid_arrow_segment_count * 2])
            canvas.lines(temp_grid_arrows_field, color=(0.8, 0.8, 0.8), width=0.001)


        grid_B_dipole_only_np = grid_B_dipole_only.to_numpy()
        grid_dipole_arrow_vertices = np.zeros((num_grid_points * num_grid_points * 3 * 2, dimension), dtype=np.float32)
        grid_dipole_arrow_segment_count = 0
        fixed_arrowhead_len_dipole = 0.003 / display_range

        for i_grid in range(num_grid_points * num_grid_points):
            p_start_disp = grid_pos_np_disp[i_grid]
            

            if np.linalg.norm(grid_pos_np[i_grid] - magnet_center.to_numpy()) < magnetic_free_zone_radius:
                continue

            current_B_norm_dipole = np.linalg.norm(grid_B_dipole_only_np[i_grid])

            if current_B_norm_dipole > B_magnitude_threshold:
                B_normalized_dipole_disp = (grid_B_dipole_only_np[i_grid] / current_B_norm_dipole) * magnetic_field_scale * 0.2 / display_range

                p_end_disp = p_start_disp + B_normalized_dipole_disp
                line_norm = np.linalg.norm(B_normalized_dipole_disp)

                if line_norm > 1e-9:
                    grid_dipole_arrow_vertices[grid_dipole_arrow_segment_count * 2] = p_start_disp
                    grid_dipole_arrow_vertices[grid_dipole_arrow_segment_count * 2 + 1] = p_end_disp
                    grid_dipole_arrow_segment_count += 1

                    current_arrowhead_len = min(fixed_arrowhead_len_dipole, line_norm * 0.4)
                    v_back = -B_normalized_dipole_disp / line_norm
                    arrow_v1_dir = rotate_vector(v_back, arrowhead_angle_rad)
                    arrow_p1 = p_end_disp + arrow_v1_dir * current_arrowhead_len
                    grid_dipole_arrow_vertices[grid_dipole_arrow_segment_count * 2] = arrow_p1
                    grid_dipole_arrow_vertices[grid_dipole_arrow_segment_count * 2 + 1] = p_end_disp
                    grid_dipole_arrow_segment_count += 1
                    arrow_v2_dir = rotate_vector(v_back, -arrowhead_angle_rad)
                    arrow_p2 = p_end_disp + arrow_v2_dir * current_arrowhead_len
                    grid_dipole_arrow_vertices[grid_dipole_arrow_segment_count * 2] = arrow_p2
                    grid_dipole_arrow_vertices[grid_dipole_arrow_segment_count * 2 + 1] = p_end_disp
                    grid_dipole_arrow_segment_count += 1

        if grid_dipole_arrow_segment_count > 0:
            temp_grid_dipole_arrows_field = ti.Vector.field(dimension, dtype=ti.f32, shape=grid_dipole_arrow_segment_count * 2)
            temp_grid_dipole_arrows_field.from_numpy(grid_dipole_arrow_vertices[:grid_dipole_arrow_segment_count * 2])
            canvas.lines(temp_grid_dipole_arrows_field, color=(0.0, 0.8, 0.0), width=0.001)
        
        window.show()

if __name__ == "__main__":
    main_simulation_loop()
