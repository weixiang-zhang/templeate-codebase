import open3d as o3d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

##### coding4nerf2 environment | open3D==0.16

input_mesh = "/home/ubuntu/projects/coding4paper/projects/libinr/log/040_sdf/dev_full_2024-11-18-00:36:33_doio/log_mesh_o3d/dragon_10000.ply"


# 读取 PLY 格式的 3D 模型
mesh = o3d.io.read_triangle_mesh(input_mesh)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# verts = np.asarray(mesh.vertices)
# faces = np.asarray(mesh.triangles)
# ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], linewidth=0.2, antialiased=True)
# plt.savefig("rendered_mesh.png")


