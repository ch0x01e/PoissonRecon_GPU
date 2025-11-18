import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import copy

def improved_plane_detection(pcd, distance_threshold=0.02, min_plane_points=500):
    """
    改进的平面检测，使用多尺度RANSAC和法向量一致性检查
    """
    planes = []
    remaining_pcd = pcd
    max_iterations = 10
    
    # 预先计算法向量
    remaining_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    for i in range(max_iterations):
        if len(remaining_pcd.points) < min_plane_points:
            break
            
        # 使用RANSAC检测平面
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        plane_pcd = remaining_pcd.select_by_index(inliers)
        
        if len(plane_pcd.points) < min_plane_points:
            break
            
        # 检查法向量一致性（确保是真正的平面）
        plane_normal = np.array(plane_model[:3])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        point_normals = np.asarray(plane_pcd.normals)
        dot_products = np.abs(np.dot(point_normals, plane_normal))
        
        # 如果法向量一致性较差，可能不是真正的平面
        if np.mean(dot_products) < 0.8:  # 阈值可调整
            print(f"平面 {i+1} 法向量一致性较差，跳过")
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            continue
        
        planes.append({
            'points': plane_pcd,
            'model': plane_model,
            'normal': plane_normal
        })
        
        print(f"找到平面 {i+1}, 点数: {len(plane_pcd.points)}, 法向量一致性: {np.mean(dot_products):.3f}")
        
        # 移除已识别的平面点
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
    
    return planes, remaining_pcd

def create_improved_plane_mesh(plane_pcd, plane_model, expansion_factor=1.1):
    """
    创建改进的平面网格，更好地保持边界和颜色
    """
    points_3d = np.asarray(plane_pcd.points)
    colors_3d = np.asarray(plane_pcd.colors) if plane_pcd.has_colors() else None
    
    if len(points_3d) < 3:
        return None
    
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # 构建局部坐标系
    if abs(normal[0]) > abs(normal[1]):
        tangent = np.array([-normal[2], 0, normal[0]])
    else:
        tangent = np.array([0, normal[2], -normal[1]])
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    
    try:
        # 投影到2D
        points_2d = np.column_stack((
            np.dot(points_3d, tangent),
            np.dot(points_3d, bitangent)
        ))
        
        # 使用Alpha Shape替代凸包，更好地保持复杂边界
        from scipy.spatial import Delaunay
        
        # 扩展边界以避免裁剪
        center = np.mean(points_2d, axis=0)
        expanded_points = center + (points_2d - center) * expansion_factor
        
        # 计算凸包（作为基础）
        hull = ConvexHull(expanded_points)
        boundary_points_2d = expanded_points[hull.vertices]
        
        # 三角化
        tri = Delaunay(boundary_points_2d)
        
        # 创建网格顶点（在3D空间中）
        vertices_3d = []
        vertex_colors = []
        
        for i, v_2d in enumerate(boundary_points_2d):
            # 将2D坐标转换回3D
            point_3d = v_2d[0] * tangent + v_2d[1] * bitangent
            
            # 投影到平面上
            t = -(np.dot(normal, point_3d) + d) / np.dot(normal, normal)
            precise_point = point_3d + t * normal
            vertices_3d.append(precise_point)
            
            # 为顶点分配颜色 - 使用区域内最近点的颜色
            if colors_3d is not None:
                # 找到原始点云中最近的点
                distances = np.linalg.norm(points_3d - precise_point, axis=1)
                nearest_idx = np.argmin(distances)
                vertex_colors.append(colors_3d[nearest_idx])
        
        vertices_3d = np.array(vertices_3d)
        
        # 创建网格
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices_3d)
        plane_mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
        
        # 设置颜色
        if colors_3d is not None and len(vertex_colors) > 0:
            plane_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        else:
            # 计算平面点的平均颜色
            avg_color = np.mean(colors_3d, axis=0) if colors_3d is not None else [0.7, 0.7, 0.7]
            plane_mesh.paint_uniform_color(avg_color)
        
        # 设置法向量
        vertex_normals = np.tile(normal, (len(vertices_3d), 1))
        plane_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
        
        return plane_mesh
        
    except Exception as e:
        print(f"创建平面网格时出错: {e}")
        return None

def improved_poisson_reconstruction(pcd, depth=10):
    """
    改进的泊松重建，更好的参数和颜色处理
    """
    if len(pcd.points) < 1000:
        print("点云点数不足，跳过泊松重建")
        return o3d.geometry.TriangleMesh()
    
    # 确保有法向量
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
        )
    
    print("正在进行泊松重建...")
    
    # 使用更好的参数
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, 
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=False
    )
    
    # 基于密度过滤
    if len(densities) > 0:
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"过滤掉 {np.sum(vertices_to_remove)} 个低密度顶点")
    
    # 为泊松网格分配颜色
    if pcd.has_colors() and len(mesh.vertices) > 0:
        print("为泊松网格分配颜色...")
        mesh_colors = []
        poisson_vertices = np.asarray(mesh.vertices)
        
        # 使用Open3D的最近邻搜索
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        for vertex in poisson_vertices:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 3)  # 找3个最近点
            nearest_colors = np.asarray(pcd.colors)[idx]
            avg_color = np.mean(nearest_colors, axis=0)
            mesh_colors.append(avg_color)
        
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    
    return mesh

def hierarchical_reconstruction_improved(pcd_path, output_path="improved_reconstruction.ply"):
    """
    改进的分层重建算法
    """
    print("=== 改进的分层重建 ===")
    
    # 加载点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        print("错误：无法读取点云文件")
        return None
    
    print(f"原始点云: {len(pcd.points)} 个点")
    print(f"包含颜色: {pcd.has_colors()}")
    
    # 创建备份
    original_pcd = copy.deepcopy(pcd)
    
    # 预处理
    print("预处理点云...")
    pcd = pcd.voxel_down_sample(voxel_size=0.02)  # 适当的下采样
    
    # 移除离群点
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    print(f"预处理后: {len(pcd.points)} 个点")
    
    # 步骤1: 改进的平面检测
    print("\n=== 平面检测 ===")
    planes, non_plane_pcd = improved_plane_detection(
        pcd, 
        distance_threshold=0.03,  # 稍微宽松的阈值
        min_plane_points=300
    )
    
    print(f"找到 {len(planes)} 个平面结构")
    print(f"非平面部分: {len(non_plane_pcd.points)} 个点")
    
    # 步骤2: 创建平面网格
    print("\n=== 创建平面网格 ===")
    plane_meshes = []
    for i, plane_data in enumerate(planes):
        print(f"处理平面 {i+1}...")
        plane_mesh = create_improved_plane_mesh(
            plane_data['points'], 
            plane_data['model'],
            expansion_factor=1.05  # 轻微扩展避免缝隙
        )
        if plane_mesh is not None:
            plane_meshes.append(plane_mesh)
            print(f"  平面 {i+1} 网格: {len(plane_mesh.vertices)} 顶点")
    
    # 步骤3: 非平面部分重建
    print("\n=== 非平面部分重建 ===")
    if len(non_plane_pcd.points) > 1000:
        non_plane_mesh = improved_poisson_reconstruction(non_plane_pcd, depth=9)
        print(f"泊松重建: {len(non_plane_mesh.vertices)} 顶点")
    else:
        non_plane_mesh = o3d.geometry.TriangleMesh()
        print("非平面部分点数不足，跳过重建")
    
    # 步骤4: 合并网格
    print("\n=== 合并网格 ===")
    final_mesh = o3d.geometry.TriangleMesh()
    
    total_vertices = 0
    total_faces = 0
    
    for i, plane_mesh in enumerate(plane_meshes):
        final_mesh += plane_mesh
        total_vertices += len(plane_mesh.vertices)
        total_faces += len(plane_mesh.triangles)
        print(f"添加平面 {i+1}")
    
    if len(non_plane_mesh.vertices) > 0:
        final_mesh += non_plane_mesh
        total_vertices += len(non_plane_mesh.vertices)
        total_faces += len(non_plane_mesh.triangles)
        print("添加非平面部分")
    
    # 后处理
    if len(final_mesh.vertices) > 0:
        print("后处理网格...")
        final_mesh.remove_duplicated_vertices()
        final_mesh.remove_duplicated_triangles()
        final_mesh.remove_degenerate_triangles()
        final_mesh.remove_non_manifold_edges()
        
        # 确保颜色
        if not final_mesh.has_vertex_colors():
            final_mesh.paint_uniform_color([0.8, 0.8, 0.8])
        
        print(f"\n=== 最终结果 ===")
        print(f"总顶点数: {total_vertices}")
        print(f"总面片数: {total_faces}")
        print(f"最终网格: {len(final_mesh.vertices)} 顶点, {len(final_mesh.triangles)} 面片")
        
        # 保存
        o3d.io.write_triangle_mesh(output_path, final_mesh)
        print(f"保存到: {output_path}")
        
        # 可视化
        print("显示结果...")
        o3d.visualization.draw_geometries([final_mesh], 
                                        window_name="改进的重建结果",
                                        width=1200, 
                                        height=800)
        
        return final_mesh
    else:
        print("错误：未能生成有效网格")
        return None

def debug_visualization(original_pcd, planes, non_plane_pcd, final_mesh):
    """
    调试可视化，显示重建过程
    """
    geometries = []
    
    # 原始点云
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(original_pcd)
    
    # 平面部分（不同颜色）
    colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]
    for i, plane_data in enumerate(planes):
        debug_pcd = plane_data['points']
        debug_pcd.paint_uniform_color(colors[i % len(colors)])
        # 平移避免重叠
        points = np.asarray(debug_pcd.points) + [2, 0, 0]
        debug_pcd.points = o3d.utility.Vector3dVector(points)
        geometries.append(debug_pcd)
    
    # 非平面部分
    if len(non_plane_pcd.points) > 0:
        non_plane_pcd.paint_uniform_color([0, 0, 0])
        points = np.asarray(non_plane_pcd.points) + [4, 0, 0]
        non_plane_pcd.points = o3d.utility.Vector3dVector(points)
        geometries.append(non_plane_pcd)
    
    # 最终网格
    if final_mesh is not None:
        mesh_points = np.asarray(final_mesh.vertices) + [6, 0, 0]
        final_mesh.vertices = o3d.utility.Vector3dVector(mesh_points)
        geometries.append(final_mesh)
    
    o3d.visualization.draw_geometries(geometries, 
                                    window_name="调试视图 - 从左到右: 原始, 平面1, 平面2, ..., 非平面, 最终网格",
                                    width=1600, 
                                    height=800)

# 使用示例
if __name__ == "__main__":
    # 替换为你的点云路径
    input_file = "point.pcd"
    
    # 运行改进的重建
    result = hierarchical_reconstruction_improved(
        input_file, 
        "improved_reconstruction_result.ply"
    )
    
    # 如果需要调试视图，可以在这里加载数据并调用 debug_visualization