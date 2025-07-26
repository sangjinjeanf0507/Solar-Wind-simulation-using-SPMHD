import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

# --- 1. 전역 상수 및 필드 선언 ---
mu_0 = 4 * np.pi * 1e-7 # 투자율 (상수, SI 단위: H/m 또는 N/A^2)
sigma_kernel = 10.0 / (7.0 * np.pi) # 2D Cubic Spline Kernel 정규화 상수 (무차원)
gamma = 5.0 / 3.0 # 비열비 (이상 기체 가정, 무차원)

dimension = 2
num_particles_max = 100000 # 최대 입자 수 (개)
dt = 7e-5# 시간 간격 (시뮬레이션 시간 단위, 현재 코드 맥락상 '초'로 가정 가능)

hex_spacing = 0.02

# 자기 확산 계수 (자기장 확산 제어, 시뮬레이션 스케일에 맞춰 조정된 값)
magnetic_diffusivity = 5e-4 

# 시뮬레이션 영역 (시뮬레이션 길이 단위)
domain_min = -10.0 # 변경됨: 도메인 크기 증가
domain_max = 10.0  # 변경됨: 도메인 크기 증가
domain_center = ti.Vector([0.0, 0.0]) # 시뮬레이션 영역의 중앙

# 중력 관련 상수 (중력 없음: 0으로 설정, 비활성화)
gravitational_constant = 5e-3 # 중력 상수 (비활성화)
center_mass = 0 # 중심 질량 (비활성화)

# 영구 자석 (자기 쌍극자) 관련 상수 (배경 자기장)
magnet_center = ti.Vector([0.0, 0.0])
initial_magnet_moment_strength = 2000.0 # 자기 쌍극자 모멘트 (SI 단위: Am^2 로 추정, 시뮬레이션 스케일에 맞춰 조정된 값)
magnet_moment_strength = initial_magnet_moment_strength
magnet_moment_direction = ti.Vector([-1.0, 0.0]) # X축 방향 (무차원)

# 시각화 관련 상수
magnetic_field_scale = 0.5 # GUI에서 보이는 자기장 벡터 시각화 스케일 (무차원)
num_grid_points = 50 # 전체 자기장 흐름 시각화를 위한 격자 해상도 (개)

# 각 입자의 자기장 크기의 기준이 됩니다. (테슬라, T)
base_B_magnitude = 0.05 # 시뮬레이션 스케일에 맞춰 조정된 값

# --- 배경 자기장 임계값 --- (테슬라, T)
B_magnitude_threshold = 5e-8 # 시뮬레이션 스케일에 맞춰 조정된 값 (예: 50 uT)

# --- 압력 상한선 --- (시뮬레이션 압력 단위, Pa로 추정)
max_pressure_cap = 100000000000000000000000.0

# --- 밀도 상한선 --- (실제 물리 단위: m^-3)
max_density_cap = 5.0 * 1e10 # 5e6 m^-3 (이는 5 cm^-3를 m^-3으로 변환한 값)
density_cap_activation_distance = 0.2 # (시뮬레이션 길이 단위)

# --- 속도 상한선 --- (미터/초, m/s)
max_velocity_cap = 1000000000000000000.0

# --- 새로운 입자 그룹 관련 상수 (특수 입자) ---
# 각 유형별 입자 수 (개)
num_ions = 5000
num_electron_core = 5000
num_electron_halo = 5000
num_special_particles = num_ions + num_electron_core + num_electron_halo # 총 특수 입자 수

# 특수 입자 초기화 및 재배치 스폰 영역 (왼쪽 끝, 위에서 아래까지)
special_particle_spawn_x_min = domain_min + 0.1 - 2.0 # 왼쪽 끝에서 약간 오른쪽으로
special_particle_spawn_x_max = domain_min + 0.5 - 2.0 # 왼쪽 끝에서 약간 오른쪽으로, 범위 조정
special_particle_spawn_y_min = domain_min
special_particle_spawn_y_max = domain_max

# 이온, 전자 코어, 전자 헤일로의 초기 속도 (시뮬레이션 단위)
# 실제 태양풍 속도: 300-800 km/s (일반), 1000-3000 km/s (빠른 태양풍)
ion_initial_velocity = ti.Vector([0.004, 0.0]) # 400 km/s -> 시뮬레이션 단위
electron_core_initial_velocity = ti.Vector([0.004, 0.0]) # 400 km/s -> 시뮬레이션 단위
electron_halo_initial_velocity_low = ti.Vector([0.002, 0.0]) # 200 km/s -> 시뮬레이션 단위
electron_halo_initial_velocity_high = ti.Vector([0.01, 0.0]) # 1000 km/s -> 시뮬레이션 단위

# 이온의 초기 자기장 (나노테슬라 -> 테슬라, nT -> T)
ion_initial_B_magnitude = 5e-9 # 5 nT -> 5e-9 T

# 입자 유형 식별을 위한 상수 (0: 일반, 1: 이온, 2: 전자 코어, 3: 전자 헤일로, 무차원)
PARTICLE_TYPE_NORMAL = 0
PARTICLE_TYPE_ION = 1
PARTICLE_TYPE_ELECTRON_CORE = 2
PARTICLE_TYPE_ELECTRON_HALO = 3

is_special_particle_type = ti.field(dtype=ti.i32, shape=num_particles_max) # 특수 입자인지 여부 및 유형 (0: 일반, 1: 이온, 2: 전자 코어, 3: 전자 헤일로)

# 자기장 없는 구역 및 입자 재배치 관련 상수 (시뮬레이션 길이 단위)
magnetic_free_zone_radius = 1.0 # 이 반경 내에는 입자 배치 안됨 / 침입 시 반사

# --- 사용자 조절 변수: 헥사곤 격자의 '최소 배치 반경' --- (시뮬레이션 길이 단위)
initial_placement_min_radius = 1.1 # 초기값 설정 (자기장 없는 구역 반경보다 약간 크게)

reposition_radius_min = magnetic_free_zone_radius + 0.1 # 재배치 시 최소 반경
reposition_radius_max = magnetic_free_zone_radius + 0.5 # 재배치 시 최대 반경

# --- 일반 입자 재배치 (Repopulation) 관련 상수 ---
repopulation_check_radius = 2.0  # 이 반경 내에서 입자 수를 확인 (시뮬레이션 길이 단위)
num_desired_particles_in_center = 2000 # 중앙 영역에 유지하고 싶은 최소 일반 입자 수
repopulation_search_attempts = 100 # 재배치 시도 횟수

# --- 특수 입자 재배치 (Repopulation) 관련 상수 --- (새로 추가)
special_particle_repop_check_center = ti.Vector([special_particle_spawn_x_min + (special_particle_spawn_x_max - special_particle_spawn_x_min) / 2.0, domain_center.y])
special_particle_repop_check_radius = 2.0 # 이 반경 내에서 특수 입자 수를 확인 (시뮬레이션 길이 단위)
num_desired_total_special_particles = num_ions + num_electron_core + num_electron_halo # 총 특수 입자 수 유지
special_repopulation_search_attempts = 100 # 재배치 시도 횟수

# 입자 간 최소 거리 (충돌 방지 및 고른 배치, 시뮬레이션 길이 단위)
min_particle_distance = 0.03
min_particle_distance_sq = min_particle_distance * min_particle_distance

# 격자점에서의 자기장 벡터를 저장할 필드
grid_pos = ti.Vector.field(dimension, dtype=ti.f32, shape=num_grid_points * num_grid_points) # (시뮬레이션 길이 단위)
grid_B_interpolated = ti.Vector.field(dimension, dtype=ti.f32, shape=num_grid_points * num_grid_points) # (테슬라, T)

# 순수한 영구 자석 자기장을 위한 격자 필드 (배경 자기장 시각화)
grid_B_dipole_only = ti.Vector.field(dimension, dtype=ti.f32, shape=num_grid_points * num_grid_points) # (테슬라, T)

# --- 2. Taichi 필드 정의 (모든 입자 데이터는 여기에 저장) ---
pos = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 위치 (시뮬레이션 길이 단위)
vel = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 속도 (m/s)
mass = ti.field(dtype=ti.f32, shape=num_particles_max) # 질량 (정규화된 값, 무차원)
u = ti.field(dtype=ti.f32, shape=num_particles_max) # 내부 에너지 (질량당 에너지, 시뮬레이션 에너지 단위)
rho = ti.field(dtype=ti.f32, shape=num_particles_max) # 밀도 (m^-3)
P_pressure = ti.field(dtype=ti.f32, shape=num_particles_max) # 압력 (시뮬레이션 압력 단위, Pa로 추정)
B = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 자기장 벡터 (테슬라, T)
acc = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 가속도 (시뮬레이션 가속도 단위, m/s^2로 추정)

# SPH MHD 특화 필드
etha_a_dt_field = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 내부 에너지 변화율
ae_k_field = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 운동 에너지
etha_a_field = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 총 에너지 (혹은 누적 에너지)
B_unit_field = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 자기장 단위 벡터 (무차원)

# 스트레스 텐서 필드
S_a_field = ti.Matrix.field(dimension, dimension, dtype=ti.f32, shape=num_particles_max)
S_b_field = ti.Matrix.field(dimension, dimension, dtype=ti.f32, shape=num_particles_max)

# 인공 점성 등을 위한 smoothing length (시뮬레이션 길이 단위)
h_smooth = ti.field(dtype=ti.f32, shape=num_particles_max)

# 자기장 업데이트를 위한 임시 필드 (dB/dt 저장)
dB_dt = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max)

# 실제로 초기화된 입자의 개수를 저장할 스칼라 필드
num_actual_particles = ti.field(dtype=ti.i32, shape=())

# --- 입자별 인공 점성 계수 필드 ---
alpha_visc_p = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 alpha_visc (무차원)
beta_visc_p = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 beta_visc (무차원)

# 전자의 전하량 상수 정의 (쿨롱, C)
electron_charge = -1.602176634e-19 # 전자의 기본 전하량 (C)
# SPH 입자당 유효 전하를 시뮬레이션 스케일에 맞게 조정
# 이 값은 물리적 정확성보다는 시뮬레이션 내 자기장 강도를 조절하는 용도로 사용될 수 있습니다.
# 예를 들어, 한 입자가 '특정 부피'의 전하를 대표한다고 가정.
# 여기에 시뮬레이션 스케일에 맞는 임의의 승수를 곱함.
effective_electron_charge_per_particle = electron_charge * 1e25 # 시뮬레이션 스케일에 맞는 유효 전하 (조정 가능, C)

# 비오-사바르 법칙에 의한 자기장 강도를 조절하는 스케일 팩터.
# 물리적으로 정확한 μ₀/2π를 시뮬레이션 스케일에 맞게 변환한 값
biot_savart_scale_factor = 6.4e6 # (물리적으로 정확한 스케일 팩터, μ₀/2π 기반)

# --- 3. SPH 커널 함수 (@ti.kernel 및 @ti.func) ---
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
    if r < 1e-9: # r=0에서의 0으로 나누기 방지
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
    if moment_strength_val >= 1e-9 and r_norm >= 1e-5: # 특이점 주변 피하기
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
    is_valid_flag = True # 플래그 변수

    dist_from_magnet_center = (candidate_pos - magnet_center).norm()

    # 1. 자기장 없는 구역 확인
    if dist_from_magnet_center < free_zone_radius_val:
        is_valid_flag = False

    # 2. 최소 배치 반경 확인 (free_zone_radius_val보다 큰 경우에만 의미 있음)
    if is_valid_flag and dist_from_magnet_center < min_placement_radius_val:
        is_valid_flag = False

    # 3. 기존 입자들과의 거리 확인 (is_valid_flag가 아직 True일 때만 검사)
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
    for i in range(num_el_core + num_el_halo): # 모든 전자 입자에 대해 반복
        current_electron_idx = initial_idx_offset + i

        target_B = ti.Vector([0.0, 0.0]) # 이 전자 입자가 받을 자기장

        # 다른 모든 전자 입자(j)가 현재 전자 입자(i) 위치에 생성하는 자기장을 합산
        for j in range(num_el_core + num_el_halo):
            if i == j:
                continue # 자기 자신은 제외

            other_electron_idx = initial_idx_offset + j

            r_vec = pos[current_electron_idx] - pos[other_electron_idx] # j에서 i로 향하는 벡터
            r = r_vec.norm()

            # 자기장 기여 계산 (2D 비오-사바르 간략화)
            if r > 1e-9: # 0으로 나누기 방지
                effective_current_x = charge_per_particle * vel[other_electron_idx].x
                effective_current_y = charge_per_particle * vel[other_electron_idx].y

                # r_vec is from j to i, so pos[i] - pos[j]
                dx = r_vec.x
                dy = r_vec.y
                r_sq = r * r

                if r_sq > 1e-9: # Avoid division by zero
                    # Bx contribution from current along X
                    dBx_from_cx = - effective_current_x * dy / r_sq
                    # Bx contribution from current along Y
                    dBx_from_cy = effective_current_y * dx / r_sq

                    # By contribution from current along X
                    dBy_from_cx = effective_current_x * dx / r_sq
                    # By contribution from current along Y
                    dBy_from_cy = - effective_current_y * dy / r_sq

                    # Sum up for the target B
                    target_B.x += (dBx_from_cx + dBx_from_cy) * bs_scale_factor
                    target_B.y += (dBy_from_cx + dBy_from_cy) * bs_scale_factor

        # 각 전자 입자의 B 필드에 계산된 비오-사바르 자기장 기여를 할당
        # 이 자기장은 초기 조건으로 한 번만 계산됩니다.
        if is_special_particle_type[current_electron_idx] == PARTICLE_TYPE_ELECTRON_CORE or \
           is_special_particle_type[current_electron_idx] == PARTICLE_TYPE_ELECTRON_HALO:
            B[current_electron_idx] = target_B
            # 자기장 크기 상한선 설정 (너무 커지는 것을 방지)
            if B[current_electron_idx].norm() > 1e-7: # 예시 상한선
                B[current_electron_idx] = B[current_electron_idx].normalized() * 1e-7


@ti.kernel
def init_particles_kernel(initial_mag_strength: ti.f32, start_row: ti.i32, end_row: ti.i32, initial_min_radius: ti.f32,
                          num_ions_param: ti.i32, num_electron_core_param: ti.i32, num_electron_halo_param: ti.i32,
                          ion_vel: ti.template(), electron_core_vel: ti.template(),
                          electron_halo_vel_low: ti.template(), electron_halo_vel_high: ti.template(),
                          ion_b_mag: ti.f32,
                          spawn_x_min: ti.f32, spawn_x_max: ti.f32, spawn_y_min: ti.f32, spawn_y_max: ti.f32): # 추가된 인자
    """
    모든 입자들을 초기화합니다 (육각형 격자 배치 및 자기장 강도에 따른 확률적 배치).
    start_row와 end_row는 상대적인 줄 번호로 사용됩니다.
    initial_min_radius는 중심으로부터 입자가 배치될 최소 거리입니다.
    """
    num_actual_particles[None] = 0
    max_placement_radius = ti.max(domain_max, domain_max) * 0.9 # 입자 분포의 최대 반경

    # --- 특수 입자 초기화 (이온, 전자 코어, 전자 헤일로) ---
    # 1. 이온 입자 초기화
    for i in range(num_ions_param):
        if num_actual_particles[None] < num_particles_max:
            idx = ti.atomic_add(num_actual_particles[None], 1)
            if idx < num_particles_max:
                is_special_particle_type[idx] = PARTICLE_TYPE_ION # 이온으로 설정
                alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1 # 낮은 점성
                vel[idx] = ion_vel
                # 특수 입자 위치: 왼쪽 끝에 위에서 아래까지 랜덤 배치
                pos[idx] = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                      spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                mass[idx] = 1.0 / num_particles_max
                u[idx] = 1.0
                # 수정된 부분: 밀도 변환 오류 수정 (5 cm^-3 -> 5e6 m^-3)
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
                B[idx] = current_B_at_pos_dir * ion_b_mag # 5 nT (5e-9 T)
            else:
                num_actual_particles[None] -= 1

    # 2. 전자 코어 입자 초기화
    electron_core_start_idx = num_actual_particles[None] # 전자 코어 시작 인덱스 저장
    for i in range(num_electron_core_param):
        if num_actual_particles[None] < num_particles_max:
            idx = ti.atomic_add(num_actual_particles[None], 1)
            if idx < num_particles_max:
                is_special_particle_type[idx] = PARTICLE_TYPE_ELECTRON_CORE # 전자 코어로 설정
                alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1 # 낮은 점성
                vel[idx] = electron_core_vel
                # 특수 입자 위치: 왼쪽 끝에 위에서 아래까지 랜덤 배치
                pos[idx] = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                      spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                mass[idx] = 1.0 / num_particles_max
                u[idx] = 1.0
                # 수정된 부분: 밀도 변환 오류 수정 (5 cm^-3 -> 5e6 m^-3)
                rho[idx] = 5.0 * 1e6
                acc[idx] = ti.Vector([0.0, 0.0])
                P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                h_smooth[idx] = 0.04
                etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                B_unit_field[idx] = ti.Vector([0.0, 0.0])
                S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                dB_dt[idx] = ti.Vector([0.0, 0.0])
                B[idx] = ti.Vector([0.0, 0.0]) # 초기에는 0으로 설정, 이후 compute_electron_initial_B_field에서 계산
            else:
                num_actual_particles[None] -= 1

    # 3. 전자 헤일로 입자 초기화
    electron_halo_start_idx = num_actual_particles[None] # 전자 헤일로 시작 인덱스 (이 값은 사용되지 않지만, 일관성을 위해 유지)
    for i in range(num_electron_halo_param):
        if num_actual_particles[None] < num_particles_max:
            idx = ti.atomic_add(num_actual_particles[None], 1)
            if idx < num_particles_max:
                is_special_particle_type[idx] = PARTICLE_TYPE_ELECTRON_HALO # 전자 헤일로로 설정
                alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1 # 낮은 점성

                if ti.random(ti.f32) < 0.5:
                    vel[idx] = electron_halo_vel_low # 200 km/s
                else:
                    vel[idx] = electron_halo_vel_high # 1000 km/s

                # 특수 입자 위치: 왼쪽 끝에 위에서 아래까지 랜덤 배치
                pos[idx] = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                      spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                mass[idx] = 1.0 / num_particles_max
                u[idx] = 1.0
                # 수정된 부분: 밀도 변환 오류 수정 (5 cm^-3 -> 5e6 m^-3)
                rho[idx] = 5.0 * 1e6
                acc[idx] = ti.Vector([0.0, 0.0])
                P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                h_smooth[idx] = 0.04
                etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                B_unit_field[idx] = ti.Vector([0.0, 0.0])
                S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                dB_dt[idx] = ti.Vector([0.0, 0.0])
                B[idx] = ti.Vector([0.0, 0.0]) # 초기에는 0으로 설정, 이후 compute_electron_initial_B_field에서 계산
            else:
                num_actual_particles[None] -= 1

    # --- 일반 입자 초기화 (헥사곤 격자) ---
    hex_rows_max = 1000 # 변경됨: 도메인 크기 증가에 따라 격자 행 수 증가
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

                    if idx < num_particles_max: # 여기서 다시 한번 최종 확인
                        B_initial_direction = ti.Vector([0.0, 0.0])
                        if current_B_norm >= B_magnitude_threshold:
                            B_initial_direction = current_B_at_pos / current_B_norm

                        is_special_particle_type[idx] = PARTICLE_TYPE_NORMAL # 일반 입자로 설정
                        alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                        vel[idx] = B_initial_direction * 0.001 # 속도 (시뮬레이션 단위)

                        pos[idx] = p_pos_candidate
                        mass[idx] = 1.0 / num_particles_max
                        u[idx] = 1.0
                        rho[idx] = 0.001# 일반 입자의 밀도 (m^-3)
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
                        num_actual_particles[None] -= 1 # 이미 증가된 카운트를 되돌립니다.
                
@ti.kernel
def add_special_particle_B_contributions():
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL:
            total_B = ti.Vector([0.0, 0.0])
            for j in range(num_actual_particles[None]):
                if is_special_particle_type[j] != PARTICLE_TYPE_NORMAL:
                    r_vec = pos[i] - pos[j]
                    r = r_vec.norm()
                    if r > 1e-6:
                        current_j = effective_electron_charge_per_particle * vel[j]
                        # 2D 비오-사바르 법칙 간이 계산
                        dB = biot_savart_scale_factor * ti.Vector([-current_j.y, current_j.x]) / (r * r)
                        total_B += dB
            B[i] += total_B

@ti.kernel
def induce_B_from_special_particles():
    """특수 입자의 전류 운동이 일반 입자에게 자기장을 유도하도록 합니다."""
    for i in range(num_actual_particles[None]):
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


@ti.kernel
def compute_sph_properties():
    """모든 입자의 밀도, 압력, 가속도 (압력, 점성, 자기장)를 계산합니다."""
    for i in range(num_actual_particles[None]):
        rho[i] = 0.0
        acc[i] = ti.Vector([0.0, 0.0])
        etha_a_dt_field[i] = 0.0
        dB_dt[i] = ti.Vector([0.0, 0.0])

    for i in range(num_actual_particles[None]):
        # 밀도 계산
        for j in range(num_actual_particles[None]):
            if i != j:
                r_vec = pos[i] - pos[j]
                r = r_vec.norm()
                h_ij = (h_smooth[i] + h_smooth[j]) / 2.0
                if r < 2.0 * h_ij:
                    density_contribution = mass[j] * W(r, h_ij)

                    target_rho = 1.0 # 일반 입자의 기본 밀도
                    if is_special_particle_type[i] == PARTICLE_TYPE_ION or \
                       is_special_particle_type[i] == PARTICLE_TYPE_ELECTRON_CORE or \
                       is_special_particle_type[i] == PARTICLE_TYPE_ELECTRON_HALO:
                        # 수정된 부분: 밀도 변환 오류 수정 (5 cm^-3 -> 5e6 m^-3)
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

        # 압력 계산
        if rho[i] > 1e-9:
            P_pressure[i] = (gamma - 1.0) * rho[i] * u[i]
        else:
            P_pressure[i] = 0.0

        P_pressure[i] = min(P_pressure[i], max_pressure_cap)

        # 스트레스 텐서 계산 (자기장 항 포함)
        B_i_norm_sq = B[i].dot(B[i])
        S_a_field[i] = P_pressure[i] * ti.Matrix.identity(ti.f32, dimension) + \
                                    (B[i].outer_product(B[i]) - 0.5 * B_i_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / mu_0

        acc_pressure_i, acc_visc_i, acc_magnetic_i = ti.Vector([0.0, 0.0]), ti.Vector([0.0, 0.0]), ti.Vector([0.0, 0.0])

        # 상호작용 계산 (압력, 점성, 자기장)
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
                # 압력 가속도
                acc_pressure_i += -mass[j] * (P_pressure[i] / rho[i]**2 + P_pressure[j] / rho[j]**2) * grad_Wij

                # 인공 점성 가속도
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

                # 일반 입자 또는 이온에만 자기장 가속도 영향을 받음
                if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL or \
                   is_special_particle_type[i] == PARTICLE_TYPE_ION:
                    # 자기장 가속도 (Lorentz force 등)
                    B_j_norm_sq = B[j].dot(B[j])
                    S_b_temp = P_pressure[j] * ti.Matrix.identity(ti.f32, dimension) + \
                                             (B[j].outer_product(B[j]) - 0.5 * B_j_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / mu_0
                    acc_magnetic_i += -mass[j] * ((S_a_field[i] / rho[i]**2 + S_b_temp / rho[j]**2) @ grad_Wij)

                # 내부 에너지 변화율
                etha_a_dt_field[i] += mass[j] * (P_pressure[i]/rho[i]**2 + P_pressure[j]/rho[j]**2 - Pi_ij) * (vel[i] - vel[j]).dot(grad_Wij)

                # 일반 입자 또는 이온에만 자기장 시간 미분 영향을 받음
                # (전자는 자기장을 유도하는 소스로서 작동하며, 자체 B 필드 변화는 직접적으로 다루지 않음)
                if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL or \
                   is_special_particle_type[i] == PARTICLE_TYPE_ION:
                    # 기존 자기장 시간 미분 (dB/dt) - advection 항
                    F_z_j = vel[j].x * B[j].y - vel[j].y * B[j].x
                    dB_dt_x_advection_j_contrib = mass[j] * F_z_j * grad_Wij.y / rho[j]
                    dB_dt_y_advection_j_contrib = -mass[j] * F_z_j * grad_Wij.x / rho[j]
                    dB_dt[i].x += dB_dt_x_advection_j_contrib
                    dB_dt[i].y += dB_dt_y_advection_j_contrib

                    # 자기장 시간 미분 (dB/dt) - 확산 항
                    diffusion_term_contrib = ti.Vector([0.0, 0.0])
                    if r_effective_for_stability > 1e-9:
                        diffusion_term_contrib = mass[j] * (B[j] - B[i]) * (r_vec.dot(grad_Wij)) / (rho[i] * r_effective_for_stability**2)
                    dB_dt[i] += magnetic_diffusivity * diffusion_term_contrib

        # 총 가속도
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

        # 일반 입자 또는 이온에 대해서만 자기장 업데이트
        # 전자 입자의 B 필드는 초기 조건으로만 설정되며, 이후는 직접적으로 업데이트되지 않습니다.
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

        # 속도 상한선 적용
        velocity_magnitude = vel[i].norm()
        if velocity_magnitude > max_velocity_cap:
            vel[i] = vel[i].normalized() * max_velocity_cap

        pos[i] += vel[i] * dt

        # --- 1. 중앙 자기장 0 구역 침범 시 입자 반사 ---
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


        # --- 2. 화면 밖으로 나간 입자 재배치 (일반 입자) ---
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
                    rho[i] = 1.0 # 일반 입자의 밀도 초기화
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
                    # 새로운 입자가 생성될 수 있는 영역을 repopulation_check_radius 근처로 제한
                    angle = ti.random(ti.f32) * 2 * math.pi
                    radius = ti.random(ti.f32) * (max_placement_rad_rep - current_initial_placement_min_radius) + current_initial_placement_min_radius
                    candidate_pos = magnet_center + ti.Vector([radius * ti.cos(angle), radius * ti.sin(angle)])

                    if not (domain_min <= candidate_pos.x <= domain_max and \
                                         domain_min <= candidate_pos.y <= domain_max):
                        attempts += 1
                        continue

                    if is_position_valid(candidate_pos, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                        idx = ti.atomic_add(num_actual_particles[None], 1)

                        if idx < num_particles_max: # 여기서 다시 한번 최종 확인
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
                            num_actual_particles[None] -= 1 # 이미 증가된 카운트를 되돌립니다.
                    attempts += 1

@ti.kernel
def count_current_special_particles_in_zone() -> ti.i32:
    """
    특수 입자 재배치 구역 내의 특수 입자 수를 세어 반환합니다.
    """
    count = 0
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] != PARTICLE_TYPE_NORMAL: # 특수 입자만 카운트
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
    spawn_x_min: ti.f32, spawn_x_max: ti.f32, spawn_y_min: ti.f32, spawn_y_max: ti.f32, # 스폰 영역
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

    # 이온 재배치
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

    # 전자 코어 재배치
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
                        B[idx] = ti.Vector([0.0, 0.0]) # 초기에는 0으로 설정
                        found_position = True
                    else:
                        num_actual_particles[None] -= 1
                attempts += 1

    # 전자 헤일로 재배치
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
                        B[idx] = ti.Vector([0.0, 0.0]) # 초기에는 0으로 설정
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

        # 자기장 없는 구역에 격자점이 있을 경우 자기장 0으로 설정
        if (p_grid_pos - magnet_center).norm() < magnetic_free_zone_radius:
            grid_B_interpolated[idx] = ti.Vector([0.0, 0.0])
            continue

        for i_particle in range(num_actual_particles[None]):
            r_vec = p_grid_pos - pos[i_particle]
            r = r_vec.norm()
            h_particle = h_smooth[i_particle]
            if r < 2.0 * h_particle:
                # 자기장 보간 시 이온과 일반 입자의 자기장만 사용
                if is_special_particle_type[i_particle] == PARTICLE_TYPE_NORMAL or \
                   is_special_particle_type[i_particle] == PARTICLE_TYPE_ION or \
                   is_special_particle_type[i_particle] == PARTICLE_TYPE_ELECTRON_CORE or \
                   is_special_particle_type[i_particle] == PARTICLE_TYPE_ELECTRON_HALO: # 전자의 자기장도 이제 보간에 포함
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

        # 자기장 없는 구역에 격자점이 있을 경우 자기장 0으로 설정
        if (p_grid_pos - magnet_center).norm() < magnetic_free_zone_radius:
            grid_B_dipole_only[idx] = ti.Vector([0.0, 0.0])
            continue

        grid_B_dipole_only[idx] = get_dipole_B_field(p_grid_pos, magnet_center, magnet_moment_direction, current_mag_strength)

# --- 4. 메인 시뮬레이션 루프 (Python) ---
def main_simulation_loop():
    global magnet_moment_strength
    global initial_placement_min_radius

    initial_placement_min_radius = 1.5

    # particles의 pos, vel, is_special_particle_type 필드 초기화
    init_particles_kernel(initial_magnet_moment_strength, -205, 205, initial_placement_min_radius, # 변경됨: 격자 행 수 증가
                          num_ions, num_electron_core, num_electron_halo,
                          ion_initial_velocity, electron_core_initial_velocity,
                          electron_halo_initial_velocity_low, electron_halo_initial_velocity_high,
                          ion_initial_B_magnitude,
                          special_particle_spawn_x_min, special_particle_spawn_x_max,
                          special_particle_spawn_y_min, special_particle_spawn_y_max) # 추가된 인자들

    print(f"Number of actual particles initialized: {num_actual_particles[None]}")

    # 전자 입자의 초기 자기장을 계산하는 커널 호출
    # 전자 코어 입자의 시작 인덱스를 계산합니다.
    electron_core_start_idx = num_ions
    compute_electron_initial_B_field(num_electron_core, num_electron_halo, electron_core_start_idx,
                                     effective_electron_charge_per_particle, biot_savart_scale_factor)
    # -------------------------------------------------------------

    window_width = 800
    window_height = 800
    window = ti.ui.Window("Taichi SPH MHD Simulation with Multi-Type Special Particles", (window_width, window_height), vsync=True)
    canvas = window.get_canvas()

    display_view_min = domain_min * 1.1 # 변경됨: 디스플레이 범위 조정
    display_view_max = domain_max * 1.1 # 변경됨: 디스플레이 범위 조정
    display_range = display_view_max - display_view_min

    def rotate_vector(vector, angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        vec_np = np.asarray(vector)
        x = vec_np[0] * c - vec_np[1] * s
        y = vec_np[0] * s + vec_np[1] * c
        return np.array([x, y], dtype=np.float32)

    # Define Earth's properties for visualization
    earth_radius_sim_units = magnetic_free_zone_radius # Use the magnetic_free_zone_radius as Earth's radius
    earth_color = (0.2, 0.4, 0.8) # Blueish color for Earth

    # Parameters for drawing the Earth (approximated by triangles)
    num_segments = 50 # Number of segments to approximate the circle
    earth_vertices = ti.Vector.field(dimension, dtype=ti.f32, shape=num_segments * 3) # Each segment is a triangle (center, point1, point2)

    @ti.kernel
    def generate_earth_vertices_kernel(center_x: ti.f32, center_y: ti.f32, radius: ti.f32, num_seg: ti.i32):
        for i in range(num_seg):
            angle1 = 2 * math.pi * i / num_seg
            angle2 = 2 * math.pi * (i + 1) / num_seg

            # Center of the Earth
            earth_vertices[i * 3] = ti.Vector([center_x, center_y])
            # First point on the circle
            earth_vertices[i * 3 + 1] = ti.Vector([center_x + radius * ti.cos(angle1), center_y + radius * ti.sin(angle1)])
            # Second point on the circle
            earth_vertices[i * 3 + 2] = ti.Vector([center_x + radius * ti.cos(angle2), center_y + radius * ti.sin(angle2)])

    frame_count = 0
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
        add_special_particle_B_contributions()
        induce_B_from_special_particles()  # 반드시 compute_sph_properties() 이전에 호출

        compute_sph_properties()
        update_particles(dt, initial_placement_min_radius)
        visualize_magnetic_field_grid_kernel(magnet_moment_strength)
        compute_static_dipole_field_kernel(magnet_moment_strength)

        # --- 일반 입자 재배치 로직 추가 ---
        current_normal_particles_in_center = count_normal_particles_in_radius(repopulation_check_radius)
        if current_normal_particles_in_center < num_desired_particles_in_center:
            # 부족한 입자 수만큼 재배치 시도
            repopulate_particles_kernel(initial_magnet_moment_strength, initial_placement_min_radius,
                                        num_desired_particles_in_center, repopulation_search_attempts,
                                        reposition_radius_max)
        # ------------------------------------

        # --- 특수 입자 재배치 로직 추가 (새로 추가) ---
        current_total_special_particles_in_zone = count_current_special_particles_in_zone()
        if current_total_special_particles_in_zone < num_desired_total_special_particles:
            repopulate_special_particles_kernel(initial_magnet_moment_strength, initial_placement_min_radius,
                                                num_ions, num_electron_core, num_electron_halo, # desired counts
                                                special_repopulation_search_attempts,
                                                special_particle_spawn_x_min, special_particle_spawn_x_max,
                                                special_particle_spawn_y_min, special_particle_spawn_y_max,
                                                ion_initial_velocity, electron_core_initial_velocity,
                                                electron_halo_initial_velocity_low, electron_halo_initial_velocity_high,
                                                ion_initial_B_magnitude)
        # ----------------------------------------------


        canvas.set_background_color((0.0, 0.0, 0.0))

        # Calculate Earth's display properties
        earth_center_disp_numpy = (magnet_center.to_numpy() - display_view_min) / display_range
        earth_radius_disp_numpy = earth_radius_sim_units / display_range

        # Generate Earth vertices in Taichi kernel
        generate_earth_vertices_kernel(earth_center_disp_numpy[0], earth_center_disp_numpy[1], earth_radius_disp_numpy, num_segments)

        # Draw Earth as a series of triangles
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
            canvas.circles(temp_normal_circles_field, radius=0.03 / display_range, color=(0.9, 0.9, 0.9)) # 일반 입자: 회색

        if len(ion_particles_pos) > 0:
            temp_ion_circles_field = ti.Vector.field(dimension, dtype=ti.f32, shape=len(ion_particles_pos))
            temp_ion_circles_field.from_numpy(np.array(ion_particles_pos, dtype=np.float32))
            canvas.circles(temp_ion_circles_field, radius=0.05 / display_range, color=(1.0, 0.0, 0.0)) # 이온: 빨간색

        if len(electron_core_particles_pos) > 0:
            temp_electron_core_circles_field = ti.Vector.field(dimension, dtype=ti.f32, shape=len(electron_core_particles_pos))
            temp_electron_core_circles_field.from_numpy(np.array(electron_core_particles_pos, dtype=np.float32))
            canvas.circles(temp_electron_core_circles_field, radius=0.05 / display_range, color=(1.0, 0.0, 0.0)) # 전자 코어: 파란색

        if len(electron_halo_particles_pos) > 0:
            temp_electron_halo_circles_field = ti.Vector.field(dimension, dtype=ti.f32, shape=len(electron_halo_particles_pos))
            temp_electron_halo_circles_field.from_numpy(np.array(electron_halo_particles_pos, dtype=np.float32))
            canvas.circles(temp_electron_halo_circles_field, radius=0.05 / display_range, color=(1.0, 0.0, 0.0)) # 전자 헤일로: 청록색

        # 그리드 포지션도 numpy로 변환하여 시각화에 사용
        grid_pos_np = grid_pos.to_numpy()
        grid_pos_np_disp = (grid_pos_np - display_view_min) / display_range

        arrowhead_angle_rad = np.pi / 6

        # --- 개별 입자의 자기장 시각화 (파란색 화살표) ---
        B_unit_field_np = B_unit_field.to_numpy()[:num_actual_particles[None]]
        particle_magnetic_arrow_vertices = np.zeros((num_actual_particles[None] * 3 * 2, dimension), dtype=np.float32)
        particle_arrow_segment_count = 0
        fixed_arrowhead_len_particle = 0.004 / display_range

        for i in range(num_actual_particles[None]):
            # 이온, 전자 코어, 전자 헤일로, 일반 입자의 자기장 시각화
            # 전자 입자의 자기장은 초기 비오-사바르 법칙에 의해 결정됩니다.
            if is_particle_type_np[i] in [PARTICLE_TYPE_ION, PARTICLE_TYPE_NORMAL, PARTICLE_TYPE_ELECTRON_CORE, PARTICLE_TYPE_ELECTRON_HALO]:
                if np.linalg.norm(B_unit_field_np[i]) > B_magnitude_threshold:
                    p_start_disp = display_pos_np[i]

                    scale_factor = magnetic_field_scale * 0.5 / display_range
                    if is_particle_type_np[i] == PARTICLE_TYPE_ION:
                        scale_factor = magnetic_field_scale * 1.0 / display_range
                    elif is_particle_type_np[i] == PARTICLE_TYPE_ELECTRON_CORE or \
                         is_particle_type_np[i] == PARTICLE_TYPE_ELECTRON_HALO:
                        scale_factor = magnetic_field_scale * 0.8 / display_range # 전자 자기장 화살표 스케일

                    direction_vec_disp = B_unit_field_np[i] * scale_factor
                    p_end_disp = p_start_disp + direction_vec_disp

                    line_norm = np.linalg.norm(direction_vec_disp)
                    if line_norm > 1e-9:
                        # Draw main line
                        particle_magnetic_arrow_vertices[particle_arrow_segment_count * 2] = p_start_disp
                        particle_magnetic_arrow_vertices[particle_arrow_segment_count * 2 + 1] = p_end_disp
                        particle_arrow_segment_count += 1

                        # Draw arrowhead
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
            # 특수 입자 자기장 화살표만 따로 시각화
            for type_id, color in [
                (PARTICLE_TYPE_ION, (1.0, 0.2, 0.2)),  # 이온: 붉은색
                (PARTICLE_TYPE_ELECTRON_CORE, (1.0, 0.2, 0.2)),  # 전자 코어: 파랑
                (PARTICLE_TYPE_ELECTRON_HALO, (1.0, 0.2, 0.2)),  # 전자 헤일로: 청록
                (PARTICLE_TYPE_NORMAL, (0.6, 0.6, 1.0)),  # 일반 입자: 회색
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
                            # 화살촉 2개
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


        # --- 보간된 자기장 그리드 시각화 (흰색 화살표) ---
        grid_B_interpolated_np = grid_B_interpolated.to_numpy()
        grid_arrow_vertices = np.zeros((num_grid_points * num_grid_points * 3 * 2, dimension), dtype=np.float32)
        grid_arrow_segment_count = 0
        fixed_arrowhead_len_grid = 0.003 / display_range # 고정된 화살촉 길이

        for i_grid in range(num_grid_points * num_grid_points):
            p_start_disp = grid_pos_np_disp[i_grid]
            current_B_norm = np.linalg.norm(grid_B_interpolated_np[i_grid])

            if current_B_norm > B_magnitude_threshold: # 일정 자기장 이상일 때만 표시
                # normalize and scale for display
                B_normalized_disp = (grid_B_interpolated_np[i_grid] / current_B_norm) * magnetic_field_scale * 0.2 / display_range # 일반 그리드 화살표 스케일

                p_end_disp = p_start_disp + B_normalized_disp
                line_norm = np.linalg.norm(B_normalized_disp)

                if line_norm > 1e-9:
                    # Draw main line
                    grid_arrow_vertices[grid_arrow_segment_count * 2] = p_start_disp
                    grid_arrow_vertices[grid_arrow_segment_count * 2 + 1] = p_end_disp
                    grid_arrow_segment_count += 1

                    # Draw arrowhead
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
            canvas.lines(temp_grid_arrows_field, color=(0.8, 0.8, 0.8), width=0.001) # width 추가

        # --- 영구 자석 자기장 그리드 시각화 (노란색 화살표 -> 초록색으로 변경됨) ---
        grid_B_dipole_only_np = grid_B_dipole_only.to_numpy()
        grid_dipole_arrow_vertices = np.zeros((num_grid_points * num_grid_points * 3 * 2, dimension), dtype=np.float32)
        grid_dipole_arrow_segment_count = 0
        fixed_arrowhead_len_dipole = 0.003 / display_range

        for i_grid in range(num_grid_points * num_grid_points):
            p_start_disp = grid_pos_np_disp[i_grid]
            
            # Use np.linalg.norm for NumPy arrays
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
            canvas.lines(temp_grid_dipole_arrows_field, color=(0.0, 0.8, 0.0), width=0.001) # 초록색으로 변경
        
        window.show()

if __name__ == "__main__":
    main_simulation_loop()