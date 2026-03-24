# sound/py/plot_simulation_with_params_cuda.py

import numpy as np
import matplotlib
# 设置Matplotlib使用Agg后端，这是一个非交互式后端，适合在没有GUI的环境中使用
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime
import logging
import os
import torch
import sys
import json
from scipy.signal import find_peaks

# 导入CUDA工具模块
sys.path.append('sound/py')
from cuda_utils import setup_cuda_for_multiprocessing, print_cuda_info, get_optimal_device

# === 日志设置 ===
# 创建logs目录
logs_dir = 'sound/logs'
os.makedirs(logs_dir, exist_ok=True)

# 生成唯一的日志文件名（基于时间戳）
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(logs_dir, f'plot_sim_cuda_{timestamp}.log')

# 配置日志记录器
logger = logging.getLogger('plot_sim_cuda')
logger.setLevel(logging.DEBUG)

# 文件处理器
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)

# 控制台处理器（仅显示INFO及以上级别）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置日志格式
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# 添加处理器到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"开始记录日志，文件: {log_filename}")

# === 初始化CUDA环境 ===
# 设置多进程启动方法为'spawn'以避免CUDA在fork中的问题
setup_cuda_for_multiprocessing()

# 获取最佳设备
device = get_optimal_device()
print_cuda_info()
logger.info(f"使用设备: {device}")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Create directories if they don't exist
img_dir = 'sound/img'
fit_dir = 'sound/fit_results'
os.makedirs(img_dir, exist_ok=True)
os.makedirs(fit_dir, exist_ok=True)
logger.debug(f"已创建目录: {img_dir}, {fit_dir}")

# === 日志记录实验参数 ===
def log_parameters(params_dict):
    """将参数记录到日志文件中"""
    logger.info("仿真参数:")
    for key, value in params_dict.items():
        logger.info(f"  {key}: {value}")
    
    # 同时保存为JSON以便于后续分析
    params_json_path = os.path.join(logs_dir, f'sim_params_{timestamp}.json')
    with open(params_json_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    logger.debug(f"参数已保存至: {params_json_path}")

# === PyTorch Calculation Function (with Attenuation) ===
def calculate_pressure_at_points_pytorch(field_points_pt, source_positions_pt, source_amplitudes_pt, k_pt, gamma_pt):
    """Calculates complex pressure at specific field points including propagation attenuation."""
    field_points_expanded = field_points_pt.unsqueeze(1)
    source_positions_expanded = source_positions_pt.unsqueeze(0)
    diff = field_points_expanded - source_positions_expanded
    r_sq = torch.sum(torch.square(diff), dim=-1)
    r = torch.sqrt(r_sq)
    r = torch.clamp(r, min=1e-9)
    r_complex = r.to(torch.complex64)
    j_pt = torch.tensor(1j, dtype=torch.complex64, device=k_pt.device)
    exp_term = torch.exp(j_pt * k_pt * r_complex)
    attenuation_factor = torch.exp(-gamma_pt * r)
    if source_amplitudes_pt.ndim == 0:
        source_amplitudes_expanded = source_amplitudes_pt
    elif source_amplitudes_pt.ndim == 1:
        source_amplitudes_expanded = source_amplitudes_pt.unsqueeze(0)
    else:
        source_amplitudes_expanded = source_amplitudes_pt
    P_contributions = source_amplitudes_expanded * exp_term * attenuation_factor / r_complex
    P_total = torch.sum(P_contributions, dim=1)
    return P_total.squeeze()

# === 批量处理函数 ===
def calculate_pressure_curve_batch(l_values_batch_pt, all_sources_pt, all_amplitudes_pt, k_pt, gamma_pt):
    """批量计算多个位置点的声压，提高GPU利用率"""
    batch_size = l_values_batch_pt.shape[0]
    # 创建批量的场点坐标 [batch_size, 3]
    field_points_batch = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    field_points_batch[:, 2] = l_values_batch_pt  # 设置z坐标为l值
    
    # 计算压力（向量化操作）
    pressure_batch = calculate_pressure_at_points_pytorch(
        field_points_batch, all_sources_pt, all_amplitudes_pt, k_pt, gamma_pt
    )
    
    return pressure_batch

# --- Get Parameters from User Input ---
def get_float_input(prompt, default_value):
    """Helper function to get float input with error handling and default value."""
    while True:
        try:
            user_input = input(f"{prompt} [默认: {default_value}]: ")
            if not user_input: # User pressed Enter
                return default_value
            return float(user_input)
        except ValueError:
            logger.warning(f"  错误：请输入一个有效的数值。")
            print("  错误：请输入一个有效的数值。")

# 新增的函数：用于交互式调整实验数据的缩放
def get_exp_data_scaling_params():
    """获取实验数据缩放参数"""
    print("\n实验数据缩放参数:")
    scale_factor = get_float_input("  缩放因子 (乘以实验数据幅度)", 1.0)
    vertical_offset = get_float_input("  垂直偏移量 (加到缩放后的实验数据)", 0.0)
    use_auto_scaling = input("  使用自动缩放匹配模拟曲线? (y/n) [y]: ").strip().lower()
    auto_scaling = (use_auto_scaling == '' or use_auto_scaling == 'y')
    
    return scale_factor, vertical_offset, auto_scaling

def try_load_params_from_file():
    """尝试从文件加载参数，便于重复使用拟合结果"""
    # 首先尝试加载最新的参数文件
    latest_params_path = os.path.join(fit_dir, 'best_fit_peaks_params_cuda_latest.txt')
    if os.path.exists(latest_params_path):
        logger.info(f"检测到最新参数文件: {latest_params_path}")
        use_latest = input("是否使用最新的拟合参数? (y/n) [y]: ").strip().lower()
        if use_latest == '' or use_latest == 'y':
            return load_params_from_file(latest_params_path)
    
    # 其次尝试其他可能的参数文件
    fit_params_path = os.path.join(fit_dir, 'best_fit_peaks_params_cuda.txt')
    if not os.path.exists(fit_params_path):
        fit_params_path = os.path.join(fit_dir, 'best_fit_peaks_params_full.txt')
    
    if os.path.exists(fit_params_path):
        logger.info(f"检测到参数文件: {fit_params_path}")
        load_file = input("是否加载此文件的参数? (y/n) [y]: ").strip().lower()
        if load_file == '' or load_file == 'y':
            return load_params_from_file(fit_params_path)
            
    # 如果想手动选择参数文件
    custom_file = input("是否手动指定参数文件? (y/n) [n]: ").strip().lower()
    if custom_file == 'y':
        file_path = input("请输入参数文件路径: ").strip()
        if os.path.exists(file_path):
            return load_params_from_file(file_path)
        else:
            logger.warning(f"文件不存在: {file_path}")
    
    return None

def load_params_from_file(file_path):
    """从指定文件加载参数"""
    logger.info(f"正在从文件加载参数: {file_path}")
    params = {}
    metadata = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                if '=' in line:
                    key, value = line.split('=')
                    key = key.strip()
                    value = value.strip()
                    
                    # 尝试将值转换为浮点数
                    try:
                        value = float(value)
                    except ValueError:
                        # 如果不是浮点数，保留为字符串（元数据）
                        metadata[key] = value
                        continue
                    
                    params[key] = value
        
        if all(key in params for key in ['R1', 'phi1_rad', 'gamma1', 'R2', 'phi2_rad', 'gamma2', 'R3', 'phi3_rad', 'gamma3', 'A2_rel', 'A3_rel']):
            param_list = [
                params['R1'], params['phi1_rad'], params['gamma1'],
                params['R2'], params['phi2_rad'], params['gamma2'], 
                params['R3'], params['phi3_rad'], params['gamma3'],
                params['A2_rel'], params['A3_rel']
            ]
            
            # 记录元数据（如果有）
            if metadata:
                logger.info("参数文件元数据:")
                for key, value in metadata.items():
                    logger.info(f"  {key}: {value}")
            
            return param_list
        else:
            logger.warning("  参数文件格式不完整，将使用手动输入。")
            missing_keys = []
            for key in ['R1', 'phi1_rad', 'gamma1', 'R2', 'phi2_rad', 'gamma2', 'R3', 'phi3_rad', 'gamma3', 'A2_rel', 'A3_rel']:
                if key not in params:
                    missing_keys.append(key)
            logger.warning(f"  缺少参数: {', '.join(missing_keys)}")
    except Exception as e:
        logger.error(f"  读取参数文件失败: {e}", exc_info=True)
    
    return None

# === 主执行流程 ===
try:
    # === 加载CSV实验数据 ===
    l_peaks_exp_mm = None
    amp_peaks_exp = None
    exp_data_path = 'sound/csv/submax.csv'
    try:
        # 加载CSV数据：第一列是位置(mm)，第二列是幅度
        exp_data = np.loadtxt(exp_data_path, delimiter=',')
        if exp_data.ndim == 1:  # 处理只有一行的情况
            exp_data = exp_data.reshape(1, -1)
        # 按位置（第一列）排序
        exp_data = exp_data[exp_data[:, 0].argsort()]
        l_peaks_exp_mm = exp_data[:, 0]
        amp_peaks_exp = exp_data[:, 1]
        
        # 记录加载的实验数据
        logger.info(f"成功加载实验数据: {len(l_peaks_exp_mm)} 个点 (位置和幅值)")
        logger.info(f"实验数据范围: {l_peaks_exp_mm[0]:.2f} mm 到 {l_peaks_exp_mm[-1]:.2f} mm")
    except Exception as e:
        logger.warning(f"未能加载实验数据文件 '{exp_data_path}': {e}")
        logger.warning("将只绘制仿真结果。")
    
    # === Fixed Simulation Parameters ===
    c = 346.0
    f1 = 36981.0
    a = 0.0191
    N_points_per_radius = 50
    N_pairs = 50

    # === Derived Acoustic Parameters ===
    f2 = 2 * f1
    f3 = 3 * f1
    lambda1 = c / f1
    lambda2 = c / f2
    lambda3 = c / f3
    k1_np = 2 * np.pi / lambda1
    k2_np = 2 * np.pi / lambda2
    k3_np = 2 * np.pi / lambda3
    k1_pt = torch.tensor(k1_np, dtype=torch.complex64, device=device)
    k2_pt = torch.tensor(k2_np, dtype=torch.complex64, device=device)
    k3_pt = torch.tensor(k3_np, dtype=torch.complex64, device=device)

    # === Transducer Discretization ===
    source_points_list = []
    xs_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1, dtype=np.float32)
    ys_grid = np.linspace(-a, a, 2 * N_points_per_radius + 1, dtype=np.float32)
    for xs in xs_grid:
        for ys in ys_grid:
            if xs**2 + ys**2 <= a**2:
                source_points_list.append(np.array([xs, ys, 0], dtype=np.float32))
    source_points_orig_np = np.array(source_points_list, dtype=np.float32)
    num_source_points = len(source_points_orig_np)
    source_points_orig_pt = torch.tensor(source_points_orig_np, dtype=torch.float32, device=device)
    logger.info(f"换能器离散化: {num_source_points} 点")
    
    # === Simulation l-range for Plotting ===
    l_min_plot_m = 0.0001  # 起始点稍微偏移0以避免奇异点
    l_max_plot_m = 0.020   # 20mm 上限
    num_l_points_plot = 1000  # 设置为1000个点
    l_values_plot_np = np.linspace(l_min_plot_m, l_max_plot_m, num_l_points_plot, dtype=np.float32)
    l_values_plot_mm = l_values_plot_np * 1000
    logger.info(f"绘图 L 范围: {l_min_plot_m*1000:.2f} mm 到 {l_max_plot_m*1000:.2f} mm ({num_l_points_plot} 点)")
    
    # === Simulation Function with Expanded Parameters and CUDA Optimization ===
    def simulate_pressure_curve(l_values_np, params):
        # Expanded parameters for each harmonic
        # [R1, phi1, gamma1, R2, phi2, gamma2, R3, phi3, gamma3, A2_rel, A3_rel]
        R1_val, phi1_rad, gamma1, R2_val, phi2_rad, gamma2, R3_val, phi3_rad, gamma3, A2_rel, A3_rel = params
        
        # 将整个l值数组转移到GPU
        l_values_pt = torch.tensor(l_values_np, dtype=torch.float32, device=device)
        
        # Complex reflection coefficients for each harmonic
        R1_complex_np = (R1_val * np.exp(1j * phi1_rad)).astype(np.complex64)
        R2_complex_np = (R2_val * np.exp(1j * phi2_rad)).astype(np.complex64)
        R3_complex_np = (R3_val * np.exp(1j * phi3_rad)).astype(np.complex64)
        
        # Transfer to PyTorch tensors
        R1_0_pt = torch.tensor(R1_complex_np, dtype=torch.complex64, device=device)
        R1_l_pt = torch.tensor(R1_complex_np, dtype=torch.complex64, device=device)
        
        R2_0_pt = torch.tensor(R2_complex_np, dtype=torch.complex64, device=device)
        R2_l_pt = torch.tensor(R2_complex_np, dtype=torch.complex64, device=device)
        
        R3_0_pt = torch.tensor(R3_complex_np, dtype=torch.complex64, device=device)
        R3_l_pt = torch.tensor(R3_complex_np, dtype=torch.complex64, device=device)
        
        # Attenuation coefficients for each harmonic
        gamma1_pt = torch.tensor(gamma1, dtype=torch.float32, device=device)
        gamma2_pt = torch.tensor(gamma2, dtype=torch.float32, device=device)
        gamma3_pt = torch.tensor(gamma3, dtype=torch.float32, device=device)
        
        # Amplitude coefficients
        A1_point_pt = torch.tensor(1.0, dtype=torch.complex64, device=device)
        A2_point_pt = torch.tensor(A2_rel, dtype=torch.complex64, device=device)
        A3_point_pt = torch.tensor(A3_rel, dtype=torch.complex64, device=device)
        
        base_amplitude1_per_point_pt = torch.full((num_source_points,), A1_point_pt.item(), dtype=torch.complex64, device=device)
        base_amplitude2_per_point_pt = torch.full((num_source_points,), A2_point_pt.item(), dtype=torch.complex64, device=device)
        base_amplitude3_per_point_pt = torch.full((num_source_points,), A3_point_pt.item(), dtype=torch.complex64, device=device)
        
        # 预分配结果张量在GPU上
        P_total_scenario_at_l_pt = torch.zeros(len(l_values_np), dtype=torch.complex64, device=device)
        
        # 批处理大小
        batch_size = min(100, len(l_values_np))  # 可以根据GPU内存调整
        
        # 处理每个谐波
        for h_idx, (k_pt_h, gamma_pt_h, base_amp_pt_h, amp_rel_h, R0_pt_h, Rl_pt_h) in enumerate([
            (k1_pt, gamma1_pt, base_amplitude1_per_point_pt, 1.0, R1_0_pt, R1_l_pt),
            (k2_pt, gamma2_pt, base_amplitude2_per_point_pt, A2_rel, R2_0_pt, R2_l_pt),
            (k3_pt, gamma3_pt, base_amplitude3_per_point_pt, A3_rel, R3_0_pt, R3_l_pt)
        ]):
            if amp_rel_h < 1e-6:
                continue
            
            # 分批处理l值
            for batch_start in range(0, len(l_values_np), batch_size):
                batch_end = min(batch_start + batch_size, len(l_values_np))
                l_batch = l_values_pt[batch_start:batch_end]
                batch_len = len(l_batch)
                
                # 为这批l值预分配结果
                P_batch = torch.zeros(batch_len, dtype=torch.complex64, device=device)
                
                # 对每个l值进行处理
                for i, l_val_pt in enumerate(l_batch):
                    if l_val_pt < 1e-9:
                        continue
                        
                    all_source_positions_list = [source_points_orig_pt]
                    all_source_amplitudes_list = [base_amp_pt_h]
                    current_amplitude_factor_pos_pt = Rl_pt_h.clone()
                    current_amplitude_factor_neg_pt = (Rl_pt_h * R0_pt_h).clone()
                    
                    for n in range(1, N_pairs + 1):
                        z_offset_pos = 2.0 * n * l_val_pt
                        amp_factor_pos = current_amplitude_factor_pos_pt
                        src_pos_plus_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_pos]], dtype=torch.float32, device=device)
                        all_source_positions_list.append(src_pos_plus_pt)
                        all_source_amplitudes_list.append(base_amp_pt_h * amp_factor_pos)
                        current_amplitude_factor_pos_pt *= (R0_pt_h * Rl_pt_h)
                        
                        z_offset_neg = -2.0 * n * l_val_pt
                        amp_factor_neg = current_amplitude_factor_neg_pt
                        src_pos_neg_pt = source_points_orig_pt + torch.tensor([[0, 0, z_offset_neg]], dtype=torch.float32, device=device)
                        all_source_positions_list.append(src_pos_neg_pt)
                        all_source_amplitudes_list.append(base_amp_pt_h * amp_factor_neg)
                        current_amplitude_factor_neg_pt *= (Rl_pt_h * R0_pt_h)
                    
                    all_src_pt = torch.cat(all_source_positions_list, dim=0)
                    all_amp_pt = torch.cat(all_source_amplitudes_list, dim=0)
                    
                    # 计算单个场点
                    target_point_l_pt = torch.tensor([[0, 0, l_val_pt]], dtype=torch.float32, device=device)
                    P_batch[i] += calculate_pressure_at_points_pytorch(
                        target_point_l_pt, all_src_pt, all_amp_pt, k_pt_h, gamma_pt_h
                    )
                
                # 将批处理结果添加到总结果中
                P_total_scenario_at_l_pt[batch_start:batch_end] += P_batch
        
        # 计算振幅并传回CPU
        P_amplitude = torch.abs(P_total_scenario_at_l_pt).cpu().numpy()
        return P_amplitude

    # 尝试从文件加载参数
    params_from_file = try_load_params_from_file()

    if params_from_file:
        params = params_from_file
        logger.info("\n使用文件中的参数:")
    else:
        logger.info("\n请输入仿真参数:")
        print("\n请输入仿真参数:")
        print("\n基频参数:")
        R1_val     = get_float_input("  R1 (基频反射系数, 0 到 1)", 0.95)
        phi1_rad   = get_float_input("  phi1 (基频相位, 弧度)", 0.0)
        gamma1     = get_float_input("  gamma1 (基频衰减系数, Np/m)", 1.0)

        print("\n二次谐波参数:")
        R2_val     = get_float_input("  R2 (二次谐波反射系数, 0 到 1)", 0.95)
        phi2_rad   = get_float_input("  phi2 (二次谐波相位, 弧度)", 0.0)
        gamma2     = get_float_input("  gamma2 (二次谐波衰减系数, Np/m)", 4.0)
        A2_rel     = get_float_input("  A2_rel (二次谐波相对幅值, >=0)", 0.1)

        print("\n三次谐波参数:")
        R3_val     = get_float_input("  R3 (三次谐波反射系数, 0 到 1)", 0.95)
        phi3_rad   = get_float_input("  phi3 (三次谐波相位, 弧度)", 0.0)
        gamma3     = get_float_input("  gamma3 (三次谐波衰减系数, Np/m)", 9.0)
        A3_rel     = get_float_input("  A3_rel (三次谐波相对幅值, >=0)", 0.05)

        # Assign to params list
        params = [R1_val, phi1_rad, gamma1, R2_val, phi2_rad, gamma2, R3_val, phi3_rad, gamma3, A2_rel, A3_rel]

    # 记录参数
    params_dict = {
        "R1": params[0],
        "phi1_rad": params[1],
        "phi1_deg": np.degrees(params[1]),
        "gamma1": params[2],
        "R2": params[3],
        "phi2_rad": params[4],
        "phi2_deg": np.degrees(params[4]),
        "gamma2": params[5],
        "R3": params[6],
        "phi3_rad": params[7],
        "phi3_deg": np.degrees(params[7]),
        "gamma3": params[8],
        "A2_rel": params[9],
        "A3_rel": params[10],
        "c": c,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda3": lambda3,
        "a": a,
        "N_points_per_radius": N_points_per_radius,
        "N_pairs": N_pairs,
        "l_min_plot_m": l_min_plot_m,
        "l_max_plot_m": l_max_plot_m,
        "num_l_points_plot": num_l_points_plot
    }
    log_parameters(params_dict)

    # === Run Simulation with Parameters ===
    logger.info("开始计算仿真曲线...")
    sim_start_time = time.time()
    
    # 主曲线仿真
    P_amplitude_sim = simulate_pressure_curve(l_values_plot_np, params)
    
    sim_end_time = time.time()
    sim_time = sim_end_time - sim_start_time
    logger.info(f"主曲线仿真完成，耗时: {sim_time:.2f} 秒")

    # === Find Peaks in Simulation ===
    peak_finding_distance_points = 5
    sim_peaks_indices, _ = find_peaks(P_amplitude_sim,
                                      distance=peak_finding_distance_points,
                                      height=np.max(P_amplitude_sim)*0.05)
    l_peaks_sim_mm = l_values_plot_mm[sim_peaks_indices]
    logger.info(f"在仿真曲线中找到 {len(l_peaks_sim_mm)} 个极大值.")
    
    # 记录峰值位置
    peaks_dict = {
        "l_peaks_sim_mm": l_peaks_sim_mm.tolist(),
        "peak_amplitudes": P_amplitude_sim[sim_peaks_indices].tolist()
    }
    peaks_json_path = os.path.join(logs_dir, f'sim_peaks_{timestamp}.json')
    with open(peaks_json_path, 'w') as f:
        json.dump(peaks_dict, f, indent=2)
    logger.debug(f"峰值位置已保存至: {peaks_json_path}")

    # === Save Simulation Results ===
    # 保存仿真结果数据
    sim_data = {
        "l_values_mm": l_values_plot_mm.tolist(),
        "amplitude": P_amplitude_sim.tolist(),
        "params": params_dict
    }
    sim_data_path = os.path.join(logs_dir, f'sim_data_{timestamp}.npz')
    np.savez(sim_data_path, **sim_data)
    logger.info(f"仿真数据已保存至: {sim_data_path}")

    # === Plotting ===
    logger.info("开始绘图...")
    
    # 获取实验数据缩放参数
    scaling_factor, vertical_offset, auto_scale = get_exp_data_scaling_params()
    
    # 创建图形
    plt.figure(figsize=(14, 7))
    
    # 绘制仿真曲线
    plt.plot(l_values_plot_mm, P_amplitude_sim, label='仿真结果', color='dodgerblue', linewidth=1.5)
    plt.plot(l_peaks_sim_mm, P_amplitude_sim[sim_peaks_indices], '^', markersize=8, label='仿真极大值', color='blue', alpha=0.9)
    
    # 绘制实验数据点（如果有）
    if l_peaks_exp_mm is not None and amp_peaks_exp is not None:
        # 只显示绘图范围内的实验数据
        valid_exp_indices = (l_peaks_exp_mm >= l_values_plot_mm[0]) & (l_peaks_exp_mm <= l_values_plot_mm[-1])
        l_peaks_exp_valid = l_peaks_exp_mm[valid_exp_indices]
        amp_peaks_exp_valid = amp_peaks_exp[valid_exp_indices]
        
        if auto_scale:
            # 自动缩放：使实验数据的幅度范围与模拟数据匹配
            max_sim_amp = np.max(P_amplitude_sim)
            max_exp_amp = np.max(amp_peaks_exp_valid)
            auto_scale_factor = max_sim_amp / max_exp_amp if max_exp_amp > 0 else 1.0
            logger.info(f"自动缩放因子: {auto_scale_factor:.4f}")
            
            # 应用自动缩放因子和用户指定的额外缩放
            amp_peaks_exp_scaled = amp_peaks_exp_valid * auto_scale_factor * scaling_factor + vertical_offset
        else:
            # 仅应用用户指定的缩放和偏移
            amp_peaks_exp_scaled = amp_peaks_exp_valid * scaling_factor + vertical_offset
        
        # 绘制缩放后的实验数据
        plt.plot(l_peaks_exp_valid, amp_peaks_exp_scaled, 'o', markersize=8, 
                label='实验数据点(已缩放)', color='red', alpha=0.7, markerfacecolor='none', mew=1.5)
        
        # 记录缩放参数到日志
        logger.info(f"实验数据缩放: 缩放因子={scaling_factor}, 偏移量={vertical_offset}, 自动缩放={auto_scale}")
        
        # 计算并显示均方根误差
        if len(l_peaks_exp_valid) > 0:
            # 为每个实验点找到最近的模拟点
            interp_sim_amp = np.interp(l_peaks_exp_valid, l_values_plot_mm, P_amplitude_sim)
            rmse = np.sqrt(np.mean((amp_peaks_exp_scaled - interp_sim_amp) ** 2))
            logger.info(f"实验数据与模拟数据的均方根误差: {rmse:.4f}")
    
    # 设置图表标题和标签
    plt.xlabel('反射面距离 l (mm)')
    plt.ylabel('相对声压振幅')
    param_str = rf'R1={params[0]:.3f}, R2={params[3]:.3f}, R3={params[6]:.3f}, A2={params[9]:.2f}, A3={params[10]:.2f}'
    title_str = f'仿真声压 vs 距离 ({l_values_plot_mm[0]:.1f}-{l_values_plot_mm[-1]:.1f}mm, {param_str})'
    if l_peaks_exp_mm is not None:
        title_str += f'\n缩放因子={scaling_factor:.2f}, 偏移量={vertical_offset:.2f}'
    plt.title(title_str)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim(l_values_plot_mm[0], l_values_plot_mm[-1])
    plt.ylim(bottom=0)
    
    # 添加理论标记（可选）
    lambda1_half_mm = (lambda1 / 2) * 1000
    y_max_plot = plt.ylim()[1]
    for k in range(int(l_values_plot_mm[-1] / lambda1_half_mm) + 1):
        l_mark = k * lambda1_half_mm
        if l_mark > l_values_plot_mm[0]:
            plt.axvline(l_mark, color='grey', linestyle=':', alpha=0.5, linewidth=1)
    
    # 保存图表
    plot_filename = f'sim_plot_{timestamp}_R1{params[0]:.2f}_R2{params[3]:.2f}_R3{params[6]:.2f}_A2{params[9]:.2f}_A3{params[10]:.2f}.pdf'
    final_plot_path = os.path.join(img_dir, plot_filename)
    try:
        plt.savefig(final_plot_path, format='pdf', bbox_inches='tight')
        logger.info(f"绘图已保存至: {final_plot_path}")
    except Exception as e:
        logger.error(f"保存绘图失败: {e}")
    
    plt.show()

except Exception as e:
    logger.error(f"执行过程中发生错误: {e}", exc_info=True)

finally:
    logger.info("绘图程序结束。")
    # 关闭日志处理器
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

# 删除以下行
# matplotlib.use('Agg') 