#!/usr/bin/env python3
"""
实时坐标变换微调GUI工具

功能:
1. 加载PLY点云和cam_to_ee标定文件
2. 实时从机器人获取ee_to_base变换
3. 可视化变换后的点云和坐标轴
4. GUI界面微调cam_to_ee的平移和旋转参数
5. 实时更新显示，方便人工对齐
6. 保存微调后的标定文件

作者: GitHub Copilot
日期: 2025-10-30
"""

import numpy as np
import torch
import open3d as o3d
import json
import os
import time
import threading
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 从point_trans_ply2pt.py导入函数
from point_trans_ply2pt import (
    pose_to_homogeneous_matrix, 
    load_transform_json,
    get_current_ee_to_base_transform
)


class TransformCalibrationGUI:
    """实时坐标变换微调GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("实时坐标变换微调工具")
        self.root.geometry("500x800")
        
        # 数据
        self.points_original = None  # 原始点云 (相机坐标系)
        self.colors = None
        self.cam_to_ee_original = None  # 原始标定
        self.ee_to_base = None  # 实时从机器人获取
        
        # 微调参数 (相对于原始标定的增量)
        self.delta_translation = [0.0, 0.0, 0.0]  # [dx, dy, dz]
        self.delta_rotation = [0.0, 0.0, 0.0]  # [roll, pitch, yaw] in degrees
        
        # 可视化
        self.vis = None
        self.vis_thread = None
        self.is_running = False
        self.needs_update = False  # 标记是否需要更新可视化
        self.last_update_time = 0.0
        
        # 机器人连接
        self.robot_ip = "192.168.56.101"
        self.robot_connected = False
        self.auto_update_ee = False
        
        # 文件路径
        self.ply_file = None
        self.cam_to_ee_file = None
        
        # 构建UI
        self.build_ui()
        
    def build_ui(self):
        """构建用户界面"""
        
        # === 1. 文件加载区域 ===
        file_frame = ttk.LabelFrame(self.root, text="1. 加载文件", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # PLY文件
        ttk.Label(file_frame, text="PLY点云:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ply_entry = ttk.Entry(file_frame, width=40)
        self.ply_entry.insert(0, "/home/hkcrc/DCIM/rs1028_2/cloud.ply")
        self.ply_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=self.browse_ply).grid(row=0, column=2, pady=2)
        
        # cam_to_ee文件
        ttk.Label(file_frame, text="cam_to_ee标定:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cam_to_ee_entry = ttk.Entry(file_frame, width=40)
        self.cam_to_ee_entry.insert(0, "cam_to_ee2.json")
        self.cam_to_ee_entry.grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="浏览", command=self.browse_cam_to_ee).grid(row=1, column=2, pady=2)
        
        # 加载按钮
        ttk.Button(file_frame, text="加载数据", command=self.load_data, 
                  style="Accent.TButton").grid(row=2, column=1, pady=10)
        
        # === 2. 机器人连接区域 ===
        robot_frame = ttk.LabelFrame(self.root, text="2. 机器人连接", padding=10)
        robot_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(robot_frame, text="机器人IP:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.robot_ip_entry = ttk.Entry(robot_frame, width=20)
        self.robot_ip_entry.insert(0, self.robot_ip)
        self.robot_ip_entry.grid(row=0, column=1, padx=5, pady=2)
        
        self.connect_btn = ttk.Button(robot_frame, text="连接机器人", command=self.connect_robot)
        self.connect_btn.grid(row=0, column=2, padx=5, pady=2)
        
        self.robot_status_label = ttk.Label(robot_frame, text="未连接", foreground="red")
        self.robot_status_label.grid(row=0, column=3, padx=5, pady=2)
        
        # 自动更新选项
        self.auto_update_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(robot_frame, text="实时更新ee_to_base (每2秒)", 
                       variable=self.auto_update_var,
                       command=self.toggle_auto_update).grid(row=1, column=1, columnspan=2, pady=5)
        
        # === 3. 微调参数区域 ===
        adjust_frame = ttk.LabelFrame(self.root, text="3. 微调cam_to_ee变换", padding=10)
        adjust_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 平移调整
        ttk.Label(adjust_frame, text="平移增量 (米):", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 5))
        
        self.sliders = {}
        
        # X平移
        ttk.Label(adjust_frame, text="ΔX:").grid(row=1, column=0, sticky=tk.E, padx=5)
        self.sliders['tx'] = tk.Scale(adjust_frame, from_=-0.1, to=0.1, resolution=0.001,
                                      orient=tk.HORIZONTAL, length=250, command=self.on_slider_change)
        self.sliders['tx'].set(0.0)
        self.sliders['tx'].grid(row=1, column=1, columnspan=2, padx=5)
        self.tx_label = ttk.Label(adjust_frame, text="0.000")
        self.tx_label.grid(row=1, column=3, padx=5)
        
        # Y平移
        ttk.Label(adjust_frame, text="ΔY:").grid(row=2, column=0, sticky=tk.E, padx=5)
        self.sliders['ty'] = tk.Scale(adjust_frame, from_=-0.1, to=0.1, resolution=0.001,
                                      orient=tk.HORIZONTAL, length=250, command=self.on_slider_change)
        self.sliders['ty'].set(0.0)
        self.sliders['ty'].grid(row=2, column=1, columnspan=2, padx=5)
        self.ty_label = ttk.Label(adjust_frame, text="0.000")
        self.ty_label.grid(row=2, column=3, padx=5)
        
        # Z平移
        ttk.Label(adjust_frame, text="ΔZ:").grid(row=3, column=0, sticky=tk.E, padx=5)
        self.sliders['tz'] = tk.Scale(adjust_frame, from_=-0.1, to=0.1, resolution=0.001,
                                      orient=tk.HORIZONTAL, length=250, command=self.on_slider_change)
        self.sliders['tz'].set(0.0)
        self.sliders['tz'].grid(row=3, column=1, columnspan=2, padx=5)
        self.tz_label = ttk.Label(adjust_frame, text="0.000")
        self.tz_label.grid(row=3, column=3, padx=5)
        
        # 旋转调整
        ttk.Label(adjust_frame, text="旋转增量 (度):", font=("", 10, "bold")).grid(
            row=4, column=0, columnspan=4, sticky=tk.W, pady=(15, 5))
        
        # Roll
        ttk.Label(adjust_frame, text="Roll:").grid(row=5, column=0, sticky=tk.E, padx=5)
        self.sliders['roll'] = tk.Scale(adjust_frame, from_=-30, to=30, resolution=0.1,
                                        orient=tk.HORIZONTAL, length=250, command=self.on_slider_change)
        self.sliders['roll'].set(0.0)
        self.sliders['roll'].grid(row=5, column=1, columnspan=2, padx=5)
        self.roll_label = ttk.Label(adjust_frame, text="0.0")
        self.roll_label.grid(row=5, column=3, padx=5)
        
        # Pitch
        ttk.Label(adjust_frame, text="Pitch:").grid(row=6, column=0, sticky=tk.E, padx=5)
        self.sliders['pitch'] = tk.Scale(adjust_frame, from_=-30, to=30, resolution=0.1,
                                         orient=tk.HORIZONTAL, length=250, command=self.on_slider_change)
        self.sliders['pitch'].set(0.0)
        self.sliders['pitch'].grid(row=6, column=1, columnspan=2, padx=5)
        self.pitch_label = ttk.Label(adjust_frame, text="0.0")
        self.pitch_label.grid(row=6, column=3, padx=5)
        
        # Yaw
        ttk.Label(adjust_frame, text="Yaw:").grid(row=7, column=0, sticky=tk.E, padx=5)
        self.sliders['yaw'] = tk.Scale(adjust_frame, from_=-30, to=30, resolution=0.1,
                                       orient=tk.HORIZONTAL, length=250, command=self.on_slider_change)
        self.sliders['yaw'].set(0.0)
        self.sliders['yaw'].grid(row=7, column=1, columnspan=2, padx=5)
        self.yaw_label = ttk.Label(adjust_frame, text="0.0")
        self.yaw_label.grid(row=7, column=3, padx=5)
        
        # 快捷按钮
        button_frame = ttk.Frame(adjust_frame)
        button_frame.grid(row=8, column=0, columnspan=4, pady=15)
        
        ttk.Button(button_frame, text="重置", command=self.reset_adjustments).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="精细模式", command=self.fine_mode).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="粗调模式", command=self.coarse_mode).pack(side=tk.LEFT, padx=5)
        
        # === 4. 可视化控制 ===
        vis_frame = ttk.LabelFrame(self.root, text="4. 可视化", padding=10)
        vis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(vis_frame, text="启动3D可视化", command=self.start_visualization,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(vis_frame, text="强制更新", command=self.force_update_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(vis_frame, text="停止可视化", command=self.stop_visualization).pack(side=tk.LEFT, padx=5)
        
        # === 5. 保存区域 ===
        save_frame = ttk.LabelFrame(self.root, text="5. 保存标定", padding=10)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(save_frame, text="保存为:").pack(side=tk.LEFT, padx=5)
        self.save_entry = ttk.Entry(save_frame, width=30)
        self.save_entry.insert(0, "cam_to_ee_calibrated.json")
        self.save_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="保存", command=self.save_calibration).pack(side=tk.LEFT, padx=5)
        
    def browse_ply(self):
        """浏览PLY文件"""
        filename = filedialog.askopenfilename(
            title="选择PLY点云文件",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
        )
        if filename:
            self.ply_entry.delete(0, tk.END)
            self.ply_entry.insert(0, filename)
    
    def browse_cam_to_ee(self):
        """浏览cam_to_ee文件"""
        filename = filedialog.askopenfilename(
            title="选择cam_to_ee标定文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.cam_to_ee_entry.delete(0, tk.END)
            self.cam_to_ee_entry.insert(0, filename)
    
    def load_data(self):
        """加载点云和标定文件"""
        try:
            self.ply_file = self.ply_entry.get()
            self.cam_to_ee_file = self.cam_to_ee_entry.get()
            
            # 加载PLY点云
            print(f"加载点云: {self.ply_file}")
            pcd = o3d.io.read_point_cloud(self.ply_file)
            self.points_original = np.asarray(pcd.points)
            self.colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else np.ones((len(self.points_original), 3))
            
            # 采样点云 (如果太大)
            if len(self.points_original) > 100000:
                print(f"  点云过大 ({len(self.points_original)}点), 采样到100000点")
                indices = np.random.choice(len(self.points_original), 100000, replace=False)
                self.points_original = self.points_original[indices]
                self.colors = self.colors[indices]
            
            print(f"  ✓ 加载点云: {self.points_original.shape}")
            
            # 加载cam_to_ee标定
            print(f"加载标定: {self.cam_to_ee_file}")
            cam_pos, cam_quat = load_transform_json(self.cam_to_ee_file)
            self.cam_to_ee_original = pose_to_homogeneous_matrix(cam_pos, cam_quat)
            print(f"  ✓ 原始cam_to_ee标定加载完成")
            print(f"    位置: {cam_pos}")
            print(f"    四元数: {cam_quat}")
            
            messagebox.showinfo("成功", "数据加载完成！\n现在可以连接机器人并启动可视化。")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载失败:\n{e}")
            print(f"加载错误: {e}")
    
    def connect_robot(self):
        """连接机器人"""
        try:
            self.robot_ip = self.robot_ip_entry.get()
            
            # 测试连接并获取一次姿态
            print(f"连接机器人: {self.robot_ip}")
            ee_pos, ee_quat = get_current_ee_to_base_transform(self.robot_ip)
            
            if ee_pos is None:
                raise RuntimeError("无法获取机器人姿态")
            
            self.ee_to_base = pose_to_homogeneous_matrix(ee_pos, ee_quat)
            self.robot_connected = True
            self.robot_status_label.config(text="已连接", foreground="green")
            self.connect_btn.config(text="重新连接")
            
            print(f"  ✓ 机器人连接成功")
            print(f"    末端位置: {ee_pos}")
            
            messagebox.showinfo("成功", "机器人连接成功！")
            
        except Exception as e:
            self.robot_connected = False
            self.robot_status_label.config(text="连接失败", foreground="red")
            messagebox.showerror("错误", f"机器人连接失败:\n{e}")
            print(f"连接错误: {e}")
    
    def toggle_auto_update(self):
        """切换自动更新ee_to_base"""
        self.auto_update_ee = self.auto_update_var.get()
        if self.auto_update_ee:
            print("启动自动更新ee_to_base (每2秒)")
            self.start_auto_update_thread()
        else:
            print("停止自动更新")
    
    def start_auto_update_thread(self):
        """启动自动更新线程"""
        def update_loop():
            while self.auto_update_ee and self.robot_connected:
                try:
                    ee_pos, ee_quat = get_current_ee_to_base_transform(self.robot_ip)
                    if ee_pos is not None:
                        self.ee_to_base = pose_to_homogeneous_matrix(ee_pos, ee_quat)
                        print(f"[{time.strftime('%H:%M:%S')}] 更新ee_to_base: {ee_pos[:2]}...")
                        if self.is_running:
                            self.needs_update = True
                except Exception as e:
                    print(f"自动更新错误: {e}")
                time.sleep(2)
        
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
    
    def on_slider_change(self, value):
        """滑块变化回调"""
        self.delta_translation[0] = self.sliders['tx'].get()
        self.delta_translation[1] = self.sliders['ty'].get()
        self.delta_translation[2] = self.sliders['tz'].get()
        
        self.delta_rotation[0] = self.sliders['roll'].get()
        self.delta_rotation[1] = self.sliders['pitch'].get()
        self.delta_rotation[2] = self.sliders['yaw'].get()
        
        # 更新标签
        self.tx_label.config(text=f"{self.delta_translation[0]:.3f}")
        self.ty_label.config(text=f"{self.delta_translation[1]:.3f}")
        self.tz_label.config(text=f"{self.delta_translation[2]:.3f}")
        self.roll_label.config(text=f"{self.delta_rotation[0]:.1f}")
        self.pitch_label.config(text=f"{self.delta_rotation[1]:.1f}")
        self.yaw_label.config(text=f"{self.delta_rotation[2]:.1f}")
        
        # 标记需要更新
        self.needs_update = True
        print(f"调整: T=[{self.delta_translation[0]:.3f}, {self.delta_translation[1]:.3f}, {self.delta_translation[2]:.3f}] "
              f"R=[{self.delta_rotation[0]:.1f}, {self.delta_rotation[1]:.1f}, {self.delta_rotation[2]:.1f}]")
    
    def reset_adjustments(self):
        """重置所有调整"""
        for key in self.sliders:
            self.sliders[key].set(0.0)
        self.on_slider_change(None)
    
    def fine_mode(self):
        """精细调整模式"""
        self.sliders['tx'].config(from_=-0.01, to=0.01, resolution=0.0001)
        self.sliders['ty'].config(from_=-0.01, to=0.01, resolution=0.0001)
        self.sliders['tz'].config(from_=-0.01, to=0.01, resolution=0.0001)
        self.sliders['roll'].config(from_=-5, to=5, resolution=0.01)
        self.sliders['pitch'].config(from_=-5, to=5, resolution=0.01)
        self.sliders['yaw'].config(from_=-5, to=5, resolution=0.01)
        print("切换到精细模式: 平移±1cm/0.1mm步长, 旋转±5度/0.01度步长")
    
    def coarse_mode(self):
        """粗调模式"""
        self.sliders['tx'].config(from_=-0.1, to=0.1, resolution=0.001)
        self.sliders['ty'].config(from_=-0.1, to=0.1, resolution=0.001)
        self.sliders['tz'].config(from_=-0.1, to=0.1, resolution=0.001)
        self.sliders['roll'].config(from_=-30, to=30, resolution=0.1)
        self.sliders['pitch'].config(from_=-30, to=30, resolution=0.1)
        self.sliders['yaw'].config(from_=-30, to=30, resolution=0.1)
        print("切换到粗调模式: 平移±10cm/1mm步长, 旋转±30度/0.1度步长")
    
    def get_adjusted_cam_to_ee(self):
        """计算微调后的cam_to_ee变换"""
        # 增量平移矩阵
        T_delta_trans = np.eye(4)
        T_delta_trans[:3, 3] = self.delta_translation
        
        # 增量旋转矩阵 (欧拉角 -> 旋转矩阵)
        delta_rot_rad = np.deg2rad(self.delta_rotation)
        R_delta = R.from_euler('xyz', delta_rot_rad).as_matrix()
        T_delta_rot = np.eye(4)
        T_delta_rot[:3, :3] = R_delta
        
        # 组合: 先旋转后平移
        T_delta = T_delta_trans @ T_delta_rot
        
        # 应用到原始标定
        T_cam_ee_adjusted = T_delta @ self.cam_to_ee_original
        
        return T_cam_ee_adjusted
    
    def start_visualization(self):
        """启动3D可视化"""
        if self.points_original is None:
            messagebox.showwarning("警告", "请先加载点云数据！")
            return
        
        if not self.robot_connected:
            messagebox.showwarning("警告", "请先连接机器人！")
            return
        
        if self.is_running:
            messagebox.showinfo("提示", "可视化已在运行")
            return
        
        self.is_running = True
        
        # 在新线程中运行可视化
        def vis_thread_func():
            self.run_visualization()
        
        self.vis_thread = threading.Thread(target=vis_thread_func, daemon=True)
        self.vis_thread.start()
        
        print("3D可视化已启动")
    
    def run_visualization(self):
        """运行Open3D可视化循环"""
        # 创建可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="实时坐标变换微调", width=1200, height=900)
        
        # 初始化几何体
        self.pcd_vis = o3d.geometry.PointCloud()
        self.coord_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.coord_ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.coord_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        
        self.vis.add_geometry(self.pcd_vis)
        self.vis.add_geometry(self.coord_base)
        self.vis.add_geometry(self.coord_ee)
        self.vis.add_geometry(self.coord_camera)
        
        # 初始更新
        print(f"[{time.strftime('%H:%M:%S')}] 初始化可视化...")
        self.needs_update = True
        self.update_visualization()
        
        # 获取视图控制
        view_control = self.vis.get_view_control()
        
        # 渲染循环
        self.last_update_time = time.time()
        update_interval = 0.1  # 更新间隔（秒）
        
        while self.is_running:
            current_time = time.time()
            
            # 如果有更新标志或定期检查
            if self.needs_update and (current_time - self.last_update_time) > update_interval:
                print(f"[{time.strftime('%H:%M:%S')}] 更新可视化...")
                self.update_visualization()
                self.needs_update = False
                self.last_update_time = current_time
            
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.033)  # ~30 FPS
        
        self.vis.destroy_window()
    
    def update_visualization(self):
        """更新可视化内容"""
        if not self.is_running or self.vis is None:
            return
        
        try:
            # 获取微调后的cam_to_ee
            T_cam_ee_adjusted = self.get_adjusted_cam_to_ee()
            
            # 计算完整变换链: camera -> ee -> base
            T_cam_base = self.ee_to_base @ T_cam_ee_adjusted
            
            # 变换点云到base坐标系
            points_homo = np.hstack([self.points_original, np.ones((self.points_original.shape[0], 1))])
            points_base = (T_cam_base @ points_homo.T).T[:, :3]
            
            # 更新点云 - 完全重新设置
            self.vis.remove_geometry(self.pcd_vis, reset_bounding_box=False)
            self.pcd_vis = o3d.geometry.PointCloud()
            self.pcd_vis.points = o3d.utility.Vector3dVector(points_base)
            self.pcd_vis.colors = o3d.utility.Vector3dVector(self.colors)
            self.vis.add_geometry(self.pcd_vis, reset_bounding_box=False)
            
            # 更新Base坐标系（始终在原点）
            self.vis.remove_geometry(self.coord_base, reset_bounding_box=False)
            self.coord_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            self.vis.add_geometry(self.coord_base, reset_bounding_box=False)
            
            # 更新EE坐标系
            self.vis.remove_geometry(self.coord_ee, reset_bounding_box=False)
            self.coord_ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.coord_ee.transform(self.ee_to_base)
            self.vis.add_geometry(self.coord_ee, reset_bounding_box=False)
            
            # 更新Camera坐标系
            self.vis.remove_geometry(self.coord_camera, reset_bounding_box=False)
            self.coord_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            self.coord_camera.transform(T_cam_base)
            self.vis.add_geometry(self.coord_camera, reset_bounding_box=False)
            
        except Exception as e:
            print(f"可视化更新错误: {e}")
            import traceback
            traceback.print_exc()
    
    def force_update_visualization(self):
        """强制更新可视化"""
        if not self.is_running:
            messagebox.showwarning("警告", "可视化未运行！请先启动3D可视化。")
            return
        
        print(f"\n[{time.strftime('%H:%M:%S')}] 强制更新可视化...")
        self.needs_update = True
        self.last_update_time = 0  # 重置时间以立即更新
    
    def stop_visualization(self):
        """停止可视化"""
        self.is_running = False
        print("停止可视化")
    
    def save_calibration(self):
        """保存微调后的标定"""
        if self.cam_to_ee_original is None:
            messagebox.showwarning("警告", "请先加载原始标定数据！")
            return
        
        try:
            # 获取微调后的变换矩阵
            T_cam_ee_adjusted = self.get_adjusted_cam_to_ee()
            
            # 提取位置和四元数
            position = T_cam_ee_adjusted[:3, 3].tolist()
            rotation_matrix = T_cam_ee_adjusted[:3, :3]
            quaternion = R.from_matrix(rotation_matrix).as_quat().tolist()
            
            # 保存为JSON
            output_file = self.save_entry.get()
            calibration_data = {
                "position": position,
                "quaternion": quaternion,
                "adjustments": {
                    "delta_translation": self.delta_translation,
                    "delta_rotation": self.delta_rotation
                },
                "timestamp": time.time(),
                "source_file": self.cam_to_ee_file,
                "note": "微调后的cam_to_ee标定"
            }
            
            with open(output_file, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            
            print(f"\n✓ 标定保存成功: {output_file}")
            print(f"  位置: {position}")
            print(f"  四元数: {quaternion}")
            print(f"  平移增量: {self.delta_translation}")
            print(f"  旋转增量: {self.delta_rotation}")
            
            messagebox.showinfo("成功", f"标定已保存到:\n{output_file}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败:\n{e}")
            print(f"保存错误: {e}")


def main():
    """主函数"""
    root = tk.Tk()
    app = TransformCalibrationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    print("=" * 60)
    print("实时坐标变换微调GUI工具")
    print("=" * 60)
    print("\n功能说明:")
    print("1. 加载PLY点云和cam_to_ee标定文件")
    print("2. 连接机器人获取实时ee_to_base变换")
    print("3. 通过GUI滑块微调cam_to_ee的平移和旋转")
    print("4. 实时3D可视化查看调整效果")
    print("5. 保存微调后的标定文件")
    print("\n使用流程:")
    print("① 输入PLY文件和cam_to_ee标定文件路径")
    print("② 点击'加载数据'")
    print("③ 输入机器人IP并点击'连接机器人'")
    print("④ 点击'启动3D可视化'")
    print("⑤ 调整滑块微调变换，观察点云对齐效果")
    print("⑥ 满意后点击'保存'保存新的标定文件")
    print("\n提示:")
    print("- 红色坐标轴: Base坐标系")
    print("- 绿色坐标轴: 末端执行器坐标系")
    print("- 蓝色坐标轴: 相机坐标系")
    print("- 可选择'精细模式'进行更精确的调整")
    print("- 可启用'实时更新'自动获取机器人姿态")
    print("=" * 60)
    print()
    
    main()
