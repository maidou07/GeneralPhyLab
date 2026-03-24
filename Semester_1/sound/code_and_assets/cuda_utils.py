#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CUDA 工具函数模块
提供 CUDA 相关的实用函数，特别是针对多进程环境的配置。
"""

import os
import sys
import torch
import multiprocessing
import numpy as np

def setup_cuda_for_multiprocessing():
    """
    为多进程环境正确设置 CUDA。
    在使用 multiprocessing 之前调用此函数。
    """
    try:
        # 在 Unix 系统上，使用 'spawn' 方法来避免 CUDA 在 fork 中重新初始化的问题
        multiprocessing.set_start_method('spawn', force=True)
        print("已将多进程启动方法设置为 'spawn'")
        return True
    except RuntimeError as e:
        print(f"注意：无法设置启动方法为 'spawn'（可能已经设置）：{e}")
        return False

def is_cuda_available():
    """
    检查 CUDA 是否可用。
    
    返回:
        bool: 如果 CUDA 可用则为 True，否则为 False
    """
    return torch.cuda.is_available()

def get_cuda_device_count():
    """
    获取可用的 CUDA 设备数量。
    
    返回:
        int: CUDA 设备的数量
    """
    if not is_cuda_available():
        return 0
    return torch.cuda.device_count()

def get_current_cuda_device():
    """
    获取当前使用的 CUDA 设备的索引。
    
    返回:
        int: 当前 CUDA 设备的索引，如果 CUDA 不可用则为 -1
    """
    if not is_cuda_available():
        return -1
    return torch.cuda.current_device()

def select_best_cuda_device():
    """
    选择最佳的 CUDA 设备（具有最大可用内存的设备）。
    
    返回:
        torch.device: 选择的设备
    """
    if not is_cuda_available():
        print("CUDA 不可用，返回 CPU 设备")
        return torch.device('cpu')
    
    device_count = get_cuda_device_count()
    if device_count == 1:
        return torch.device('cuda:0')
    
    # 如果有多个 GPU，选择内存最大的设备
    max_free_memory = 0
    best_device = 0
    
    for i in range(device_count):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = i
    
    print(f"已选择 CUDA 设备 {best_device}，可用内存: {max_free_memory / 1024**3:.2f} GB")
    return torch.device(f'cuda:{best_device}')

def set_cuda_device_env(device_index=None):
    """
    设置 CUDA_VISIBLE_DEVICES 环境变量。
    
    参数:
        device_index (int, optional): 要使用的 CUDA 设备索引。
                                      如果为 None，则使用最佳设备。
    
    返回:
        torch.device: 设置的设备
    """
    if not is_cuda_available():
        print("CUDA 不可用，返回 CPU 设备")
        return torch.device('cpu')
    
    if device_index is None:
        device = select_best_cuda_device()
        device_index = device.index
    else:
        device = torch.device(f'cuda:{device_index}')
    
    # 设置环境变量，使得只有选定的 GPU 对进程可见
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_index)
    print(f"已将 CUDA_VISIBLE_DEVICES 设置为 {device_index}")
    
    return device

def get_optimal_device():
    """
    获取最优的计算设备（CUDA 如果可用，否则 CPU）。
    
    返回:
        torch.device: 最优的计算设备
    """
    if is_cuda_available():
        return select_best_cuda_device()
    else:
        print("CUDA 不可用，使用 CPU")
        return torch.device('cpu')

def print_cuda_info():
    """打印 CUDA 和 GPU 的详细信息。"""
    if not is_cuda_available():
        print("CUDA 不可用")
        return
    
    print(f"CUDA 可用，版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"发现的 CUDA 设备数量: {get_cuda_device_count()}")
    print(f"当前 CUDA 设备: {get_current_cuda_device()}")
    
    for i in range(get_cuda_device_count()):
        props = torch.cuda.get_device_properties(i)
        free_memory = props.total_memory - torch.cuda.memory_allocated(i)
        print(f"设备 {i}: {props.name}")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  可用内存: {free_memory / 1024**3:.2f} GB")
        print(f"  多处理器数量: {props.multi_processor_count}")

def safe_cuda_computation(func, *args, **kwargs):
    """
    安全地执行可能使用 CUDA 的计算，捕获并处理常见的 CUDA 错误。
    
    参数:
        func: 要执行的函数
        *args, **kwargs: 传递给函数的参数
    
    返回:
        函数的返回值，如果出现 CUDA 错误，则回退到 CPU 计算
    """
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA 错误: {e}")
            print("尝试在 CPU 上重新计算...")
            
            # 临时设置所有张量到 CPU
            old_device = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    old_device[key] = value.device
                    kwargs[key] = value.cpu()
            
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.is_cuda:
                    new_args.append(arg.cpu())
                else:
                    new_args.append(arg)
            
            result = func(*new_args, **kwargs)
            
            # 如果原始结果应该在 GPU 上，将结果移回 GPU
            if any(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args) or \
               any(isinstance(value, torch.Tensor) and key in old_device for key, value in kwargs.items()):
                if isinstance(result, torch.Tensor):
                    device = next((d for d in old_device.values()), 
                                 next((arg.device for arg in args if isinstance(arg, torch.Tensor) and arg.is_cuda), 
                                     torch.device('cuda:0')))
                    result = result.to(device)
            
            return result
        else:
            raise  # 如果不是 CUDA 错误，则重新抛出异常

# 示例使用
if __name__ == "__main__":
    setup_cuda_for_multiprocessing()
    print_cuda_info()
    device = get_optimal_device()
    print(f"选择的最优设备: {device}") 