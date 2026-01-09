"""
EgoAllo可视化 - 导出为视频文件
用于模型对比，将推理结果渲染为MP4视频
"""

import numpy as np
import torch
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import trimesh
from typing import Optional

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.transforms import SE3, SO3
from egoallo.network import EgoDenoiseTraj


class VideoRenderer:
    """简单的3D渲染器，用于生成视频帧"""
    
    def __init__(self, width=1920, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # 相机参数
        self.camera_distance = 3.0
        self.camera_height = 1.5
        self.camera_angle = 0.0  # 初始角度
        
    def project_3d_to_2d(self, points_3d: np.ndarray, camera_pos: np.ndarray, camera_target: np.ndarray):
        """简单的3D投影到2D"""
        # 相机坐标系
        forward = camera_target - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # 视图矩阵
        view_matrix = np.eye(4)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = -camera_pos
        
        # 投影矩阵 (简单透视)
        fov = 60.0
        aspect = self.width / self.height
        near = 0.1
        far = 100.0
        f = 1.0 / np.tan(np.radians(fov) / 2)
        
        proj_matrix = np.zeros((4, 4))
        proj_matrix[0, 0] = f / aspect
        proj_matrix[1, 1] = f
        proj_matrix[2, 2] = (far + near) / (near - far)
        proj_matrix[2, 3] = (2 * far * near) / (near - far)
        proj_matrix[3, 2] = -1
        
        # 齐次坐标
        points_homo = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
        
        # 变换
        points_cam = points_homo @ view_matrix.T
        points_clip = points_cam @ proj_matrix.T
        
        # NDC坐标
        points_ndc = points_clip[:, :3] / points_clip[:, 3:4]
        
        # 屏幕坐标
        points_screen = np.zeros((len(points_3d), 2))
        points_screen[:, 0] = (points_ndc[:, 0] + 1) * self.width / 2
        points_screen[:, 1] = (1 - points_ndc[:, 1]) * self.height / 2
        
        return points_screen.astype(np.int32), points_clip[:, 2]
    
    def render_skeleton(self, frame, joints_3d: np.ndarray, camera_pos: np.ndarray, camera_target: np.ndarray):
        """渲染骨架"""
        # SMPL-H骨架连接
        skeleton_connections = [
            (0, 1), (0, 2), (0, 3),  # Pelvis to hips and spine
            (1, 4), (2, 5),  # Hips to knees
            (4, 7), (5, 8),  # Knees to ankles
            (3, 6), (6, 9),  # Spine to chest to neck
            (9, 12), (9, 13), (9, 14),  # Neck to shoulders and head
            (12, 15), (13, 16),  # Shoulders to elbows
            (15, 18), (16, 19),  # Elbows to wrists
            (18, 20), (19, 21),  # Wrists to hands (simplified)
        ]
        
        # 投影关节
        joints_2d, depths = self.project_3d_to_2d(joints_3d, camera_pos, camera_target)
        
        # 绘制连接线
        for i, j in skeleton_connections:
            if i < len(joints_2d) and j < len(joints_2d):
                if 0 <= joints_2d[i, 0] < self.width and 0 <= joints_2d[i, 1] < self.height and \
                   0 <= joints_2d[j, 0] < self.width and 0 <= joints_2d[j, 1] < self.height:
                    # 根据深度调整颜色（远的更暗）
                    depth_factor = max(0.3, min(1.0, 5.0 / (depths[i] + 1)))
                    color = (int(100 * depth_factor), int(200 * depth_factor), int(255 * depth_factor))
                    cv2.line(frame, tuple(joints_2d[i]), tuple(joints_2d[j]), color, 2)
        
        # 绘制关节点
        for idx, (joint_2d, depth) in enumerate(zip(joints_2d, depths)):
            if 0 <= joint_2d[0] < self.width and 0 <= joint_2d[1] < self.height:
                depth_factor = max(0.3, min(1.0, 5.0 / (depth + 1)))
                color = (int(50 * depth_factor), int(150 * depth_factor), int(200 * depth_factor))
                cv2.circle(frame, tuple(joint_2d), 5, color, -1)
        
        return frame
    
    def render_frame(self, joints_3d: np.ndarray, frame_idx: int, total_frames: int, info_text: str = ""):
        """渲染单帧"""
        # 创建空白帧
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        
        # 相机旋转（环绕拍摄）
        angle = self.camera_angle + (frame_idx / total_frames) * 360
        camera_pos = np.array([
            self.camera_distance * np.cos(np.radians(angle)),
            self.camera_distance * np.sin(np.radians(angle)),
            self.camera_height
        ])
        
        # 相机目标（身体中心）
        camera_target = np.mean(joints_3d, axis=0)
        camera_target[2] = 1.0  # 固定高度
        
        # 渲染骨架
        frame = self.render_skeleton(frame, joints_3d, camera_pos, camera_target)
        
        # 添加信息文本
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        
        if info_text:
            cv2.putText(frame, info_text, (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # 绘制地面网格
        self.draw_ground_grid(frame, camera_pos, camera_target)
        
        return frame
    
    def draw_ground_grid(self, frame, camera_pos, camera_target):
        """绘制地面网格"""
        grid_size = 10
        grid_step = 0.5
        
        grid_points = []
        for x in np.arange(-grid_size/2, grid_size/2 + grid_step, grid_step):
            for y in np.arange(-grid_size/2, grid_size/2 + grid_step, grid_step):
                grid_points.append([x, y, 0.0])
        
        grid_points = np.array(grid_points)
        grid_2d, depths = self.project_3d_to_2d(grid_points, camera_pos, camera_target)
        
        # 绘制网格点
        for point, depth in zip(grid_2d, depths):
            if 0 <= point[0] < self.width and 0 <= point[1] < self.height:
                cv2.circle(frame, tuple(point), 1, (180, 180, 180), -1)


def visualize_npz_to_video(
    npz_path: Path,
    output_video_path: Path,
    body_model: fncsmpl.SmplhModel,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30
):
    """
    将EgoAllo的NPZ输出渲染为视频
    
    Args:
        npz_path: EgoAllo输出的NPZ文件路径
        output_video_path: 输出视频路径
        body_model: SMPL-H身体模型
        width: 视频宽度
        height: 视频高度
        fps: 视频帧率
    """
    print(f"加载NPZ文件: {npz_path}")
    data = np.load(npz_path)
    
    # 提取数据
    Ts_world_cpf = torch.from_numpy(data['Ts_world_cpf'])  # [T, 7]
    body_quats = torch.from_numpy(data['body_quats'])  # [1, T, 21, 4]
    betas = torch.from_numpy(data['betas'])  # [1, T, 16]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Ts_world_cpf = Ts_world_cpf.to(device)
    body_quats = body_quats.to(device)
    betas = betas.to(device)
    body_model = body_model.to(device)
    
    timesteps = Ts_world_cpf.shape[0]
    print(f"总帧数: {timesteps}")
    
    # 应用body模型获取关节位置
    if 'left_hand_quats' in data and 'right_hand_quats' in data:
        left_hand_quats = torch.from_numpy(data['left_hand_quats']).to(device)
        right_hand_quats = torch.from_numpy(data['right_hand_quats']).to(device)
    else:
        left_hand_quats = None
        right_hand_quats = None
    
    # 前向运动学
    shaped = body_model.with_shape(betas.mean(dim=1, keepdim=True))
    fk_outputs = shaped.with_pose_decomposed(
        T_world_root=SE3.identity(device=device, dtype=body_quats.dtype).parameters(),
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
    )
    
    # 计算世界坐标系下的根节点位置
    T_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        fk_outputs,
        Ts_world_cpf[None, ...],
    )
    fk_outputs = fk_outputs.with_new_T_world_root(T_world_root)
    
    # 提取关节位置 [num_samples, T, num_joints, 3]
    joints_world = fk_outputs.Ts_world_joint[..., 4:7].cpu().numpy()  # [1, T, num_joints, 3]
    joints_world = joints_world[0]  # [T, num_joints, 3]
    
    print(f"关节位置形状: {joints_world.shape}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # 创建渲染器
    renderer = VideoRenderer(width, height, fps)
    
    # 提取序列名称
    sequence_name = npz_path.parent.name
    
    print(f"渲染视频: {output_video_path}")
    for t in tqdm(range(timesteps), desc="渲染帧"):
        # 渲染帧
        frame = renderer.render_frame(
            joints_world[t, :21],  # 只使用body关节
            t,
            timesteps,
            info_text=f"Sequence: {sequence_name}"
        )
        
        # 写入视频
        video_writer.write(frame)
    
    video_writer.release()
    print(f"✓ 视频已保存: {output_video_path}")


def batch_visualize_outputs(
    search_dir: Path,
    output_dir: Path,
    body_model: fncsmpl.SmplhModel,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30
):
    """批量可视化输出目录中的所有NPZ文件"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有NPZ文件
    npz_files = list(search_dir.rglob("*.npz"))
    print(f"\n找到 {len(npz_files)} 个NPZ文件")
    
    for npz_path in npz_files:
        # 生成输出视频路径
        relative_path = npz_path.relative_to(search_dir)
        video_name = f"{relative_path.parent.name}_{npz_path.stem}.mp4"
        output_video_path = output_dir / video_name
        
        print(f"\n处理: {npz_path.name}")
        try:
            visualize_npz_to_video(
                npz_path,
                output_video_path,
                body_model,
                width,
                height,
                fps
            )
        except Exception as e:
            print(f"❌ 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='EgoAllo可视化 - 导出为视频')
    parser.add_argument('--npz-path', type=str,
                        help='单个NPZ文件路径')
    parser.add_argument('--search-dir', type=str,
                        default='./ee4d_inference_output',
                        help='搜索NPZ文件的目录（批量处理）')
    parser.add_argument('--output-dir', type=str,
                        default='./ee4d_videos',
                        help='输出视频目录')
    parser.add_argument('--smplh-npz-path', type=str,
                        default='./data/smplh/neutral/model.npz',
                        help='SMPL-H模型路径')
    parser.add_argument('--width', type=int, default=1920,
                        help='视频宽度')
    parser.add_argument('--height', type=int, default=1080,
                        help='视频高度')
    parser.add_argument('--fps', type=int, default=30,
                        help='视频帧率')
    
    args = parser.parse_args()
    
    # 加载body模型
    print(f"加载SMPL-H模型: {args.smplh_npz_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)
    
    if args.npz_path:
        # 单文件处理
        npz_path = Path(args.npz_path)
        output_video_path = Path(args.output_dir) / f"{npz_path.stem}.mp4"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        visualize_npz_to_video(
            npz_path,
            output_video_path,
            body_model,
            args.width,
            args.height,
            args.fps
        )
    else:
        # 批量处理
        batch_visualize_outputs(
            Path(args.search_dir),
            Path(args.output_dir),
            body_model,
            args.width,
            args.height,
            args.fps
        )


if __name__ == '__main__':
    main()
