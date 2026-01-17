import bpy
import sys
import json
import os

# -------- argüman --------
# blender --python load_scene_from_json.py -- scene.json
argv = sys.argv
argv = argv[argv.index("--") + 1:]
scene_json_path = argv[0]

if not os.path.exists(scene_json_path):
    raise RuntimeError(f"scene.json bulunamadı: {scene_json_path}")

# -------- sahneyi temizle --------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# -------- json oku --------
with open(scene_json_path, "r", encoding="utf-8") as f:
    scene = json.load(f)

objects = scene.get("objects", [])

# -------- objeleri oluştur --------
for i, obj_data in enumerate(objects):
    label = obj_data.get("label", f"obj_{i}")
    scale = obj_data.get("scale", 1.0)
    loc = obj_data.get("location", {})

    x = loc.get("x", 0.0)
    y = loc.get("y", 0.0)
    z = loc.get("z", 0.0)

    # küp oluştur
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, z))
    cube = bpy.context.active_object
    cube.name = f"{label}_{i}"

    # uniform scale
    cube.scale = (scale, scale, scale)

# -------- viewport --------
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces[0].shading.type = 'SOLID'
