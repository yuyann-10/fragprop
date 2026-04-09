import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
from diffrax import DiscreteTerminatingEvent
import numpy as np
import jax
import jax.numpy as jnp
import diffrax 
from diffrax import diffeqsolve, PIDController, ODETerm, Tsit5, Dopri5, SaveAt
import matplotlib
matplotlib.use('TkAgg') 
import os
import gc 
 
# --- 全局字体设置 ---
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  
jax.config.update("jax_enable_x64", True)

# --- 常量和特征值 ---
mu = 0.01215058 
M = 5.9722e24 + 7.346e22 
L = 389703e3 # m
G = 6.67430e-11 
vv = 1017.551785 # m/s
tt = 382981 # s
tt_days = tt / (24 * 3600) 

# --- BCR4BP 常量 ---
mu3 = 1.989e30 / M  
R3 = 149.6e9 / L    
n_s = 0.9252         

 

# -------------------------------------------------------------
#                   BCR4BP 动力学
# -------------------------------------------------------------
@jax.jit
def bcr4bp_rhs(t, state, args):
    x, y, z, vx, vy, vz = state
    mu = args["mu"]
    mu3 = args["mu3"]
    R3 = args["R3"]
    n_s = args["n_s"]
    x_moon = 1.0 - mu
    r_moon_norm = 1738.0 / 389703.0 
    
    dist_to_moon_sq = (x - x_moon)**2 + y**2 + z**2
    
    def derivatives(_):
        mu1 = 1.0 - mu
        r1_sq = (x + mu)**2 + y**2 + z**2 + 1e-18
        r2_sq = (x - mu1)**2 + y**2 + z**2 + 1e-18
        
        # --- 加速修改点 ---
        inv_r1 = jax.lax.rsqrt(r1_sq) # 直接得到 1/sqrt(r1_sq)
        inv_r1_3 = inv_r1 * inv_r1 * inv_r1
        inv_r2 = jax.lax.rsqrt(r2_sq)
        inv_r2_3 = inv_r2 * inv_r2 * inv_r2
        
        psi = n_s * t 
        R_sun_x, R_sun_y = R3 * jnp.cos(psi), R3 * jnp.sin(psi)
        r3_sq = (x - R_sun_x)**2 + (y - R_sun_y)**2 + z**2 + 1e-18
        
        inv_r3 = jax.lax.rsqrt(r3_sq)
        inv_r3_3 = inv_r3 * inv_r3 * inv_r3
        
        ax_cr3bp = x + 2.0 * vy - mu1 * (x + mu) * inv_r1_3 - mu * (x - mu1) * inv_r2_3 
        ay_cr3bp = y - 2.0 * vx - mu1 * y * inv_r1_3 - mu * y * inv_r2_3 
        az_cr3bp = -mu1 * z * inv_r1_3 - mu * z * inv_r2_3 
        
        ax_sun = -mu3 * (x - R_sun_x) * inv_r3_3 - mu3 * jnp.cos(psi) / R3**2
        ay_sun = -mu3 * (y - R_sun_y) * inv_r3_3 - mu3 * jnp.sin(psi) / R3**2
        az_sun = -mu3 * z * inv_r3_3
        # ------------------
        
        return jnp.array([vx, vy, vz, ax_cr3bp + ax_sun, ay_cr3bp + ay_sun, az_cr3bp + az_sun])
    def frozen(_):
        return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return jax.lax.cond(dist_to_moon_sq < r_moon_norm**2, frozen, derivatives, None)

import equinox as eqx  

@eqx.filter_jit
def single_fragment_propagation(X0: jnp.ndarray, args: dict, duration: float, save_ts: jnp.ndarray) -> jnp.ndarray:
    base_term = ODETerm(bcr4bp_rhs) 
    solver = Dopri5()
    saveat = SaveAt(ts=save_ts)
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-8)
    x_moon = 1.0 - args['mu']
    r_moon_norm = 1738.0 / 389703.0
 
 
    sol = diffeqsolve(
        base_term, solver, t0=0.0, t1=duration, y0=X0, 
        dt0=5e-3, saveat=saveat, args=args, 
        stepsize_controller=stepsize_controller, 
        max_steps=10000, throw=False  
    )
    pos = sol.ys[:, 0:3]
    dist = jnp.sqrt((pos[...,0] - x_moon)**2 + pos[...,1]**2 + pos[...,2]**2)

    mask = dist > r_moon_norm
    sol_clean = jnp.where(mask[..., None], sol.ys, 0.0)
    return sol_clean 


@jax.jit
def calculate_PKH_spherical(v_total_kms, mass_kg):
    """球体模型的旧版 PKH (用于复现体积法静态对比)"""
    kl, tw, sigma, CL, tb, rho_b, rho_p = 1.9, 0.32, 57.0, 0.37, 0.127, 2.70, 2.70
    mass_g = mass_kg * 1000.0
    vol_p = mass_g / rho_p
    dp_cm = 2.0 * ((3.0 * vol_p) / (4.0 * jnp.pi))**(1.0/3.0) 
    bracket = tw * (sigma / 40.0)**0.5 + CL * tb * rho_b
    base_dc = kl * bracket * (rho_p**-0.5) * (v_total_kms**(-2.0/3.0))
    angles_deg = jnp.linspace(0, 90, 100) 
    angles_rad = jnp.clip(angles_deg, 0, 65) * jnp.pi / 180.0
    dc_array = base_dc * (jnp.cos(angles_rad))**(-11.0/6.0)
    failed_fraction = jnp.sum(dp_cm > dc_array) / 100.0
    return 0.5 * failed_fraction
'''
@jax.jit
def calculate_PKH_face_jax(v_total_kms, v_perp_kms, mass_kg):
    """
    精确的各面弹道极限 PKH (带有实际撞击角)
    v_total_kms: 绝对相对速度大小
    v_perp_kms: 垂直于表面的速度分量大小
    """
    kl, tw, sigma, CL, tb, rho_b, rho_p = 1.9, 0.32, 57.0, 0.37, 0.127, 2.70, 2.70
    mass_g = mass_kg * 1000.0
    vol_p = mass_g / rho_p
    dp_cm = 2.0 * ((3.0 * vol_p) / (4.0 * jnp.pi))**(1.0/3.0) 
    bracket = tw * (sigma / 40.0)**0.5 + CL * tb * rho_b
    
    # 防止除零
    v_total_safe = jnp.maximum(v_total_kms, 1e-6)
    
    # 计算撞击角的余弦值: cos(theta) = v_perp / v_total
    cos_theta = jnp.clip(v_perp_kms / v_total_safe, 0.0, 1.0)
    
    # Christiansen 规定超过 65度 的斜击按 65度 计算
    cos_65 = jnp.cos(65.0 * jnp.pi / 180.0)
    cos_theta = jnp.maximum(cos_theta, cos_65)
    
    # 临界穿透直径 (代入真实撞击角)
    dc_cm = kl * bracket * (rho_p**-0.5) * (v_total_safe**(-2.0/3.0)) * (cos_theta**(-11.0/6.0))
    
    # 击穿则有 50% 概率导致失效
    return jnp.where(dp_cm > dc_cm, 0.5, 0.0)
'''
@jax.jit
def extract_batch_components(trajs_batch, sc_pos_batch, sc_vel_batch, frag_masses, 
                                           area_sphere_km2, R_DZ_lu, vv_kms, V_DZ_km3):
    """
    针对 50m 直径球体航天器优化的风险评估算子
    """
    # 1. 危险区判定
    rel_pos = trajs_batch[:, :, 0:3] - sc_pos_batch[None, :, :]
    in_dz_mask = jnp.sum(rel_pos**2, axis=-1) < R_DZ_lu**2
    N_in_dz_t = jnp.sum(in_dz_mask, axis=0)

    # 2. 计算相对速度模长 (km/s)
    v_rel_vu = trajs_batch[:, :, 3:6] - sc_vel_batch[None, :, :]
    v_total_kms = jnp.linalg.norm(v_rel_vu, axis=-1) * vv_kms
    sum_v_rel_in_dz = jnp.sum(v_total_kms * in_dz_mask, axis=0)
    '''
    # 3. 计算球体PKH
    vmap_pkh_sph = jax.vmap(jax.vmap(calculate_PKH_spherical, in_axes=(0, None)), in_axes=(0, 0))
    pkh_sph_matrix = vmap_pkh_sph(v_total_kms, frag_masses)
    Sum_PKH_Spherical_t = jnp.sum(pkh_sph_matrix * in_dz_mask, axis=0)  
    '''
    # 4. 修正的通量计算
    # 单个碎片的通量贡献 = (相对速度 × 面积) / 危险区体积
    # 然后对所有碎片求和
    e_rate_per_fragment = v_total_kms * area_sphere_km2 / V_DZ_km3 
    Sum_E_Rate_Flux_t = jnp.sum(e_rate_per_fragment * in_dz_mask, axis=0)  
    
    return N_in_dz_t,  Sum_E_Rate_Flux_t, sum_v_rel_in_dz
# =========================================================================
#                   主函数 (Main)  
# =========================================================================
# ... (前面的导入和常量部分保持不变) ...

@eqx.filter_jit
def propagate_segments(X_init, args, duration_days, num_steps):
    """通用传播函数，返回末状态和全过程轨迹"""
    save_ts = jnp.linspace(0, duration_days / tt_days, num_steps)
    base_term = ODETerm(bcr4bp_rhs)
    solver = Dopri5()
    stepsize_controller = PIDController(rtol=1e-7, atol=1e-9)
    
    sol = diffeqsolve(
        base_term, solver, t0=0.0, t1=duration_days / tt_days, 
        y0=X_init, dt0=1e-3, saveat=SaveAt(ts=save_ts), 
        args=args, stepsize_controller=stepsize_controller,
        max_steps=20000, throw=False
    )
    return sol.ys  # (num_steps, 6)

def main():
    bcr4bp_args = {"mu": mu, "mu3": mu3, "R3": R3, "n_s": n_s}
    PRE_PROP_DAYS = 7.0  # 碎片独立预传播时间
    
    # --- 1. 碎片与轨迹准备 ---
    points_file = 'sampled_LLO_breakup_8points.npy'
    sampled_states = np.load(points_file, allow_pickle=True)
    if sampled_states.ndim == 1: sampled_states = np.stack(sampled_states)
    
    # 归一化 LLO 点
    sampled_states_normalized = np.copy(sampled_states)
    if np.any(sampled_states > 10.0):
        sampled_states_normalized[:, 0:3] /= L
        sampled_states_normalized[:, 3:6] /= vv

    # 加载碎片属性
    #SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    #file_name = os.path.join(SCRIPT_DIR, 'fragment_properties_schemeA_point5.txt')
    
    file_name = "/home/user1/python_work/fragment_results_8points/fragment_properties_10k_A_point5.txt"
    df_frags = pd.read_csv(file_name, sep=',', comment='#', header=None,
                           names=['ID', 'dVx', 'dVy', 'dVz', 'Vmag', 'Mass'])
    v_data = df_frags[['dVx', 'dVy', 'dVz']].values
    mass_data = jnp.array(df_frags['Mass'].values)
    N_fragments = len(mass_data)

    # 加载 Artemis 轨道
    df_artemis = pd.read_csv("artemis_bcr4bp_input.csv")
    artemis_times = df_artemis['Time_Days'].values
    artemis_states = df_artemis[['x','y','z','vx','vy','vz']].values
    sc_interp = interp1d(artemis_times, artemis_states, axis=0, bounds_error=False, fill_value="extrapolate")
    TOTAL_END = np.max(artemis_times)

    # 采样 8 个爆炸时刻 t_exp
    NUM_RUNS = 8
    # 确保 t_exp + 7天 不会超过任务总时长
    t_exp_samples = np.linspace(0.0, TOTAL_END - PRE_PROP_DAYS - 1.0, NUM_RUNS)

    # 常量
    R_DZ_km = 100.0
    R_DZ_lu = R_DZ_km / (L / 1000.0)
    V_DZ_km3 = (4/3) * np.pi * (R_DZ_km**3)
    AREA_SC_KM2 = jnp.pi * (0.01**2) 
    vv_kms = vv / 1000.0
    
    vmap_prop = eqx.filter_jit(jax.vmap(propagate_segments, in_axes=(0, None, None, None)))
    results_archive = {}

    # --- 2. 核心循环 ---
    for run_idx, t_exp in enumerate(t_exp_samples):
        print(f"\n[Run {run_idx+1}/8] 爆炸时刻: {t_exp:.2f} d | 评估起点: {t_exp + PRE_PROP_DAYS:.2f} d")
        
        # A. 准备碎片初始状态 (在 t_exp 时刻)
        X0_breakup = jnp.asarray(sampled_states_normalized[4]) # Point 5
        frags_init_at_texp = jnp.stack([
            jnp.full(N_fragments, X0_breakup[0]), jnp.full(N_fragments, X0_breakup[1]), jnp.full(N_fragments, X0_breakup[2]),
            v_data[:,0]/vv, v_data[:,1]/vv, v_data[:,2]/vv
        ], axis=1)

        # B. 评估阶段的时间跨度
        t_start_eval = t_exp + PRE_PROP_DAYS
        eval_duration = TOTAL_END - t_start_eval
        NUM_STEPS_EVAL = 5000
        T_EVAL_REL = np.linspace(0, eval_duration, NUM_STEPS_EVAL) # 相对 t_start_eval 的时间
        dt_sec = (T_EVAL_REL[1] - T_EVAL_REL[0]) * 24 * 3600
        
        # 获取航天器在评估阶段的轨迹
        SC_DATA = sc_interp(t_start_eval + T_EVAL_REL)
        SC_POS_JAX = jnp.array(SC_DATA[:, 0:3])
        SC_VEL_JAX = jnp.array(SC_DATA[:, 3:6])

        # 初始化统计
        global_Sum_Flux = np.zeros(NUM_STEPS_EVAL)
        global_N_in_dz = np.zeros(NUM_STEPS_EVAL)
        
        BATCH_SIZE = 1000
        num_batches = (N_fragments // BATCH_SIZE)
        
        for b_idx in tqdm(range(num_batches), desc="  Processing Batches", leave=False):
            start = b_idx * BATCH_SIZE
            end = start + BATCH_SIZE
            fb = frags_init_at_texp[start:end]
            mb = mass_data[start:end]
            
            # --- 第一阶段：预传播 7 天 ---
            # 我们只需要 7 天后的末状态作为第二阶段的初值
            # 为了节省显存，我们只取末状态
            pre_trajs = vmap_prop(fb, bcr4bp_args, PRE_PROP_DAYS, 2) 
            frags_at_7d = pre_trajs[:, -1, :] # 获取第 7 天的末状态 (batch, 6)
            
            # --- 第二阶段：风险评估传播 ---
            # 从第 7 天状态开始，传播评估时长
            eval_trajs = vmap_prop(frags_at_7d, bcr4bp_args, eval_duration, NUM_STEPS_EVAL)
            eval_trajs = jnp.where(jnp.isnan(eval_trajs), 1e6, eval_trajs)
            
            # 计算风险
            n_t, _, flux_t, _ = extract_batch_components(
                eval_trajs, SC_POS_JAX, SC_VEL_JAX, mb,
                AREA_SC_KM2, R_DZ_lu, vv_kms, V_DZ_km3
            )
            global_N_in_dz += np.array(n_t)
            global_Sum_Flux += np.array(flux_t)

        # 结果保存
        E_total = np.sum(global_Sum_Flux * dt_sec)
        results_archive[f"run_{run_idx}"] = {
            "t_exp": t_exp,
            "t_start_eval": t_start_eval,
            "time_rel": T_EVAL_REL, # 这是从第 7 天开始计时的相对时间
            "n_in_dz": global_N_in_dz,
            "flux_rate_ts": global_Sum_Flux,
            "e_total_flux": E_total,
            "p_hz_flux": 1.0 - np.exp(-E_total)
        }
        
        # 中途保存防止崩溃
        np.savez_compressed("artemis_p5_delayed_7d_hazard.npz", **results_archive)
        print(f"   Done. E_total: {E_total:.4e} | P_total: {1.0 - np.exp(-E_total):.4e}")
        gc.collect()

if __name__ == "__main__":
    main()