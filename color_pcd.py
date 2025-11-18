import open3d as o3d
import numpy as np

# 读取PLY文件

pcd = o3d.io.read_point_cloud("bunny1.ply")  # 修改为你的输入文件路径

# 创建蓝色颜色数组 (R=0, G=0, B=1.0 对应纯蓝色)
blue_colors = np.array([[0.0, 0.0, 1.0]] * len(pcd.points))

# 设置点云颜色
pcd.colors = o3d.utility.Vector3dVector(blue_colors)

# 导出为ASCII格式的PLY文件
o3d.io.write_point_cloud("bunny_blue.ply", pcd, write_ascii=True)

print("完成! 所有点已设置为蓝色并导出为ASCII格式")