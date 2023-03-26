import os
# import bpy
import numpy as np
import trimesh


basis_mesh = trimesh.load(
    os.path.abspath("../../mouth.glb"), force="mesh"
)
np.save(
    os.path.abspath("indices.npy"), basis_mesh.faces,
)

# mesh = bpy.data.objects["object_mouth"]
# shape_keys = mesh.data.shape_keys.key_blocks
# base_coords = None
# deltas = np.zeros((len(shape_keys) - 1, len(shape_keys[0].data), 3))
# for i, key in enumerate(shape_keys):
#     if key.name == "Basis":
#         base_coords = np.empty(len(key.data) * 3)
#         key.data.foreach_get("co", base_coords)
#         base_coords = base_coords.reshape(-1, 3)
#         np.save(
#             os.path.join("data/neutral.npy"), base_coords,
#         )
#         continue
#
#     print(key.name)
#
#     coords = np.empty(len(key.data) * 3)
#     key.data.foreach_get("co", coords)
#     coords = coords.reshape(-1, 3)
#     deltas[i - 1] = coords - base_coords
#
# np.save(os.path.join("data/deltas.npy"), deltas)
# print(deltas.shape)
# print(basis_mesh.faces.shape)
