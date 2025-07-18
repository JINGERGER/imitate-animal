import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import imageio
import os
import signal
import sys
from PIL import Image  # 用于图像尺寸统一

# ===== 1. 定义3D坐标系和基本参数 =====
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 设置固定的画布尺寸（像素）
fig.set_dpi(100)  # 设置DPI确保尺寸一致
fixed_width, fixed_height = 1200, 1000  # 固定宽度和高度

ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_zlim(0, 3)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Quadruped-Arm-Camera Perception System', fontsize=14)
ax.set_box_aspect([8, 8, 3])

# ===== 参数设置 =====
dog_radius = 0.38
arm_base_height = 0.42
arm_length = 1.0
fov_vertical = 57
fov_horizontal = 86
detection_range = 1.5
radius = 0.9
z_offset = 0.6

# 动作帧数设置
num_frames_lift = 30          # 0-29: 抬起阶段
num_frames_rotate = 60        # 30-89: 旋转阶段
num_frames_view_up = 30       # 90-119: 视野上偏
num_frames_view_back = 30     # 120-149: 视野回位
num_frames_lower = 30         # 150-179: 放下阶段
num_frames_move = 30          # 180-209: 移动阶段

# 计算各阶段起始帧
lift_end = num_frames_lift
rotate_end = lift_end + num_frames_rotate
view_up_end = rotate_end + num_frames_view_up
view_back_end = view_up_end + num_frames_view_back
lower_end = view_back_end + num_frames_lower
move_end = lower_end + num_frames_move

# 计算动作总帧数
frames_per_action = move_end  # 210帧
num_steps = 5
total_frames = frames_per_action * num_steps  # 1050帧

start_pos = np.array([0.75, 0.75, 0])

# 存储点云数据
current_points = np.empty((0, 3))  # 当前扫描点
clustered_mesh_points = []         # 聚类后的历史点组
max_current_points = 1500         # 当前点的最大数量
mesh_update_interval = 60         # 每多少帧更新一次网格
cluster_eps = 0.3                 # DBSCAN聚类的距离阈值
cluster_min_samples = 10          # 形成聚类的最小点数
smoothing_factor = 0.5            # 平滑因子

# 用于保存GIF的变量
frames = []  # 存储所有帧的列表
save_gif = True  # 是否保存GIF
animation_running = True  # 动画运行状态标志

# ===== Ctrl+C信号处理函数 =====
# def handle_sigint(signal, frame):
#     """捕获Ctrl+C信号，保存已捕获的帧为GIF并退出"""
#     global animation_running
#     print("\n检测到Ctrl+C，正在保存当前帧为GIF...")
#     animation_running = False
    
#     # 保存已捕获的帧
#     if save_gif and len(frames) > 0:
#         # 确保所有帧尺寸一致
#         normalized_frames = normalize_frame_sizes(frames)
#         imageio.mimsave('quadruped_perception_partial.gif', normalized_frames, fps=12, loop=0)
#         print(f"已保存部分动画为: quadruped_perception_partial.gif (共{len(normalized_frames)}帧)")
#     else:
#         print("没有可保存的帧数据")
    
#     # 关闭所有窗口并退出
#     plt.close('all')
#     sys.exit(0)

# # 注册信号处理器
# signal.signal(signal.SIGINT, handle_sigint)

def normalize_frame_sizes(frame_list):
    """将所有帧调整为统一尺寸"""
    if not frame_list:
        return []
    
    # 使用第一帧的尺寸作为标准（或使用固定尺寸）
    target_size = (fixed_height, fixed_width)
    
    normalized = []
    for frame in frame_list:
        # 将numpy数组转换为PIL图像
        img = Image.fromarray(frame)
        # 调整尺寸
        img_resized = img.resize(target_size[::-1], Image.LANCZOS)  # 注意宽度和高度的顺序
        # 转换回numpy数组
        normalized.append(np.array(img_resized))
    
    return normalized

def calculate_optimal_eps(points, k=5):
    if len(points) <= k:
        return cluster_eps
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    distances = np.sort(distances[:, k-1])
    return np.mean(distances[-10:])

def generate_scanned_points(position, view_direction, fov_v, fov_h, max_range):
    global current_points
    dir_norm = view_direction / np.linalg.norm(view_direction)
    
    right = np.cross(dir_norm, np.array([0, 0, 1]))
    right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([1, 0, 0])
    up = np.cross(dir_norm, right)
    up = up / np.linalg.norm(up)
    
    num_points = 30
    new_points = []
    for _ in range(num_points):
        dist = np.random.uniform(0.1, max_range)
        theta = np.random.uniform(-np.radians(fov_h/2), np.radians(fov_h/2))
        phi = np.random.uniform(-np.radians(fov_v/2), np.radians(fov_v/2))
        
        point_local = np.array([
            dist * np.cos(phi) * np.cos(theta),
            dist * np.cos(phi) * np.sin(theta),
            dist * np.sin(phi)
        ])
        
        axis = np.cross(np.array([1, 0, 0]), dir_norm)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.array([0, 1, 0])
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(np.array([1, 0, 0]), dir_norm))
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        point_world = position + R @ point_local
        new_points.append(point_world)
    
    current_points = np.vstack((current_points, new_points))
    if len(current_points) > max_current_points:
        current_points = current_points[-max_current_points:]

def cluster_points(points):
    if len(points) < cluster_min_samples:
        return [points]
    optimal_eps = calculate_optimal_eps(points)
    dbscan = DBSCAN(eps=optimal_eps, min_samples=cluster_min_samples)
    labels = dbscan.fit_predict(points)
    
    clusters = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        clusters.append(points[labels == label])
    
    return clusters if clusters else [points]

def update_mesh(frame):
    global current_points, clustered_mesh_points
    if frame % mesh_update_interval == 0 and frame > 0 and len(current_points) > 0:
        new_clusters = cluster_points(current_points)
        for new_cluster in new_clusters:
            is_new = True
            center_new = np.mean(new_cluster, axis=0)
            for existing_cluster in clustered_mesh_points:
                center_existing = np.mean(existing_cluster, axis=0)
                if np.linalg.norm(center_new - center_existing) < cluster_eps * 1.5:
                    is_new = False
                    break
            if is_new:
                clustered_mesh_points.append(new_cluster)
        current_points = np.empty((0, 3))

def smooth_mesh(points, factor=0.2):
    if len(points) < 5:
        return points
    nbrs = NearestNeighbors(n_neighbors=8).fit(points)
    _, indices = nbrs.kneighbors(points)
    smoothed = np.zeros_like(points)
    for i in range(len(points)):
        neighbors = points[indices[i]]
        smoothed[i] = (1 - factor) * points[i] + factor * np.mean(neighbors, axis=0)
    return smoothed

def create_mesh_from_cluster(cluster):
    if len(cluster) < 4:
        return None
    smoothed_cluster = smooth_mesh(cluster, smoothing_factor)
    sample_size = min(150, len(smoothed_cluster))
    if len(smoothed_cluster) > sample_size:
        step = len(smoothed_cluster) // sample_size
        cluster_sample = smoothed_cluster[::step][:sample_size]
    else:
        cluster_sample = smoothed_cluster
    
    try:
        tri = Delaunay(cluster_sample[:, :2])
        mesh = Poly3DCollection(cluster_sample[tri.simplices], alpha=0.4, edgecolor='none')
        z_values = cluster_sample[:, 2]
        if len(z_values) > 0:
            norm = plt.Normalize(z_values.min(), z_values.max())
            mesh.set_facecolor(cm.Oranges(norm(z_values[tri.simplices].mean(axis=1))))
        return mesh
    except:
        return None

def draw_frustum(ax, position, view_direction, fov_v, fov_h, max_range):
    h = max_range * np.tan(np.radians(fov_h/2))
    v = max_range * np.tan(np.radians(fov_v/2))
    dir_norm = view_direction / np.linalg.norm(view_direction)
    
    right = np.cross(dir_norm, np.array([0, 0, 1]))
    right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([1, 0, 0])
    up = np.cross(dir_norm, right)
    up = up / np.linalg.norm(up)
    
    points = np.array([
        [0, 0, 0],
        position + dir_norm * max_range + right * h + up * v,
        position + dir_norm * max_range - right * h + up * v,
        position + dir_norm * max_range - right * h - up * v,
        position + dir_norm * max_range + right * h - up * v
    ])
    points[1:] -= position
    
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    for line in lines:
        ax.plot3D(
            points[line, 0] + position[0],
            points[line, 1] + position[1],
            points[line, 2] + position[2],
            color='red',
            alpha=0.6
        )
    
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = position[0] + detection_range * np.cos(u) * np.sin(v)
    y = position[1] + detection_range * np.sin(u) * np.sin(v)
    z = position[2] + detection_range * np.cos(v)
    ax.plot_wireframe(x, y, z, color="cyan", alpha=0.2)

def update(frame):
    if not animation_running:
        return []
        
    ax.cla()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(0, 3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Quadruped-Arm-Camera Perception System', fontsize=14)
    ax.set_box_aspect([8, 8, 3])

    update_mesh(frame)
    
    # 计算当前周期和帧位置
    step_idx = frame // frames_per_action
    frame_in_action = frame % frames_per_action  # 0-209

    # 显示当前阶段
    stage_text = "Unknown"
    if frame_in_action < lift_end:
        stage_text = f"Lifting (0-{lift_end-1})"
    elif frame_in_action < rotate_end:
        stage_text = f"Rotating ({lift_end}-{rotate_end-1})"
    elif frame_in_action < view_up_end:
        stage_text = f"Action 1: View Up ({rotate_end}-{view_up_end-1})"
    elif frame_in_action < view_back_end:
        stage_text = f"Action 2: View Back ({view_up_end}-{view_back_end-1})"
    elif frame_in_action < lower_end:
        stage_text = f"Lowering ({view_back_end}-{lower_end-1})"
    else:
        stage_text = f"Moving ({lower_end}-{move_end-1})"

    # 计算四足机器人位置
    move_progress = 0.0
    if frame_in_action >= lower_end:  # 移动阶段
        move_frame = frame_in_action - lower_end
        move_progress = move_frame / num_frames_move
    dog_pos = start_pos + np.array([step_idx + move_progress, 0, 0])
    arm_base = np.array([dog_pos[0], dog_pos[1], arm_base_height])

    # 绘制四足机器人
    dog_body = Circle((dog_pos[0], dog_pos[1]), dog_radius, color='gray', alpha=0.7)
    ax.add_patch(dog_body)
    art3d.pathpatch_2d_to_3d(dog_body, z=0.1, zdir='z')
    ax.text(dog_pos[0], dog_pos[1], 0.3, "Quadruped", color='black', fontsize=9)
    ax.text(arm_base[0], arm_base[1], arm_base[2]+0.1, "Arm Base", color='blue', fontsize=9)
    ax.text(4, 7, 2.5, f"Cycle: {step_idx+1}/{num_steps}", color='green', fontsize=10)
    ax.text(4, 6.7, 2.5, f"Frame: {frame_in_action}/{frames_per_action-1}", color='black', fontsize=10)
    ax.text(4, 6.4, 2.5, stage_text, color='purple', fontsize=10)

    # 初始化机械臂参数
    arm_end = arm_base.copy()
    view_direction = np.array([1, 0, 0])
    view_angle = 0

    # 执行对应阶段动作
    if frame_in_action < lift_end:
        # 抬起阶段
        lift_ratio = frame_in_action / num_frames_lift
        arm_end = arm_base + np.array([radius, 0, z_offset * lift_ratio])
        view_direction = np.array([1, 0, 0])

    elif frame_in_action < rotate_end:
        # 旋转阶段
        rotate_frame = frame_in_action - lift_end
        angle = 2 * np.pi * rotate_frame / num_frames_rotate
        arm_end = arm_base + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            z_offset
        ])
        view_direction = np.array([np.cos(angle), np.sin(angle), 0])

    elif frame_in_action < view_up_end:
        # 动作1：视野上偏
        view_up_frame = frame_in_action - rotate_end
        arm_end = arm_base + np.array([radius, 0, z_offset])  # 位置固定
        angle_ratio = view_up_frame / num_frames_view_up
        view_angle = 135 * angle_ratio  # 0°→135°
        rad_angle = np.radians(view_angle)
        
        view_direction = np.array([
            np.cos(rad_angle),  # X分量
            0,                  # Y分量
            np.sin(rad_angle)   # Z分量（上偏）
        ])
        ax.text(arm_end[0], arm_end[1], arm_end[2]+0.5, 
                f"View angle: {int(view_angle)}° up", 
                color='purple', fontsize=9)

    elif frame_in_action < view_back_end:
        # 动作2：视野回位
        view_back_frame = frame_in_action - view_up_end
        arm_end = arm_base + np.array([radius, 0, z_offset])
        angle_ratio = view_back_frame / num_frames_view_back
        view_angle = 135 * (1 - angle_ratio)  # 135°→0°
        rad_angle = np.radians(view_angle)
        
        view_direction = np.array([
            np.cos(rad_angle),
            0,
            np.sin(rad_angle)
        ])
        ax.text(arm_end[0], arm_end[1], arm_end[2]+0.5, 
                f"View angle: {int(view_angle)}° up", 
                color='purple', fontsize=9)

    elif frame_in_action < lower_end:
        # 放下阶段
        lower_frame = frame_in_action - view_back_end
        lower_ratio = 1 - (lower_frame / num_frames_lower)
        arm_end = arm_base + np.array([radius, 0, z_offset * lower_ratio])
        view_direction = np.array([1, 0, 0])

    else:
        # 移动阶段
        arm_end = arm_base + np.array([radius, 0, 0])
        view_direction = np.array([1, 0, 0])

    # 生成扫描点（所有阶段都执行）
    generate_scanned_points(arm_end, view_direction, fov_vertical, fov_horizontal, detection_range)

    # 绘制机械臂和相机
    ax.plot([arm_base[0], arm_end[0]],
            [arm_base[1], arm_end[1]],
            [arm_base[2], arm_end[2]],
            'o-', color='blue', linewidth=3)
    ax.scatter(arm_end[0], arm_end[1], arm_end[2], color='red', s=100, marker='o', label='Depth Camera')
    ax.quiver(arm_end[0], arm_end[1], arm_end[2], 0.2, 0, 0, 
              color='green', linewidth=2, label='Camera orientation')
    draw_frustum(ax, arm_end, view_direction, fov_vertical, fov_horizontal, detection_range)
    ax.text(arm_end[0], arm_end[1], arm_end[2]+0.1, "Depth Camera", color='red', fontsize=9)

    # 绘制网格和当前扫描点
    for cluster in clustered_mesh_points:
        mesh_surface = create_mesh_from_cluster(cluster)
        if mesh_surface:
            ax.add_collection3d(mesh_surface)
    if len(current_points) > 0:
        ax.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2], 
                   color='yellow', s=5, alpha=0.8, label='Current Scan')

    ax.legend()

    # 捕获当前帧并添加到列表
    if save_gif:
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        
        # 确保图像尺寸正确
        width, height = fig.canvas.get_width_height()
        expected_size = width * height * 3
        
        if len(image_data) > expected_size:
            image_data = image_data[:expected_size]
        elif len(image_data) < expected_size:
            padding = np.zeros(expected_size - len(image_data), dtype='uint8')
            image_data = np.concatenate([image_data, padding])
        
        image = image_data.reshape((height, width, 3))
        frames.append(image)

    return ax,

# 运行动画
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=80, blit=False)

# 显示动画
plt.tight_layout()  
plt.show()

# # 正常结束时保存完整GIF
# if save_gif and len(frames) > 0 and animation_running:
#     # 确保所有帧尺寸一致
#     normalized_frames = normalize_frame_sizes(frames)
#     imageio.mimsave('quadruped_perception.gif', normalized_frames, fps=12, loop=0)
#     print("GIF已保存为: quadruped_perception.gif")
