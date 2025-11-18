import open3d as o3d
import numpy as np

# 可选：固定随机种子，便于复现
# np.random.seed(42)

pcd = o3d.io.read_point_cloud("bunny1.ply")  # 输入文件路径

# 为每个点生成随机 RGB，范围 [0,1]
rand_colors = np.random.rand(len(pcd.points), 3).astype(np.float64)

pcd.colors = o3d.utility.Vector3dVector(rand_colors)

o3d.io.write_point_cloud("bunny_random.ply", pcd, write_ascii=True)
print("完成! 已为每个点赋随机颜色并导出 bunny_random.ply")