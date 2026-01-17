import bpy
import sys
import json
import os
import mathutils

def get_xy_bbox(obj):
    """
    Objeye ait world-space XY bounding box döner
    """
    world_bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    xs = [v.x for v in world_bbox]
    ys = [v.y for v in world_bbox]

    return min(xs), max(xs), min(ys), max(ys)

def bbox_overlap_xy(a, b):
    """
    XY düzleminde iki bbox çakışıyor mu
    """
    a_min_x, a_max_x, a_min_y, a_max_y = a
    b_min_x, b_max_x, b_min_y, b_max_y = b

    overlap_x = min(a_max_x, b_max_x) - max(a_min_x, b_min_x)
    overlap_y = min(a_max_y, b_max_y) - max(a_min_y, b_min_y)

    if overlap_x > 0 and overlap_y > 0:
        return overlap_x, overlap_y

    return None

def resolve_floor_collisions(objects,
                             max_iters=10,
                             padding=0.01,
                             correction_ratio=0.3,
                             max_step=0.2):
    """
    Floor objeleri arasındaki çakışmaları minimal hareketlerle çözer
    """

    floor_objs = [
        o["blender_obj"]
        for o in objects
        if o.get("placement") == "floor" and "blender_obj" in o
    ]

    for _ in range(max_iters):
        moved_any = False

        for i in range(len(floor_objs)):
            for j in range(i + 1, len(floor_objs)):
                a = floor_objs[i]
                b = floor_objs[j]

                bbox_a = get_xy_bbox(a)
                bbox_b = get_xy_bbox(b)

                overlap = bbox_overlap_xy(bbox_a, bbox_b)
                if not overlap:
                    continue

                overlap_x, overlap_y = overlap

                # hangi eksende daha az hareket gerekir
                if overlap_x < overlap_y:
                    axis = mathutils.Vector((1, 0, 0))
                    overlap_amt = overlap_x
                else:
                    axis = mathutils.Vector((0, 1, 0))
                    overlap_amt = overlap_y

                # sadece overlap'in bir kısmı kadar düzelt
                move_dist = min(overlap_amt * correction_ratio, max_step) + padding

                # yön
                direction = (b.location - a.location).dot(axis)
                axis = axis if direction >= 0 else -axis

                # SADECE b objesini it
                b.location += axis * move_dist

                moved_any = True

        if not moved_any:
            break

       
def normalize_mesh_to_unit_bbox(obj):
    mesh = obj.data

    # local-space bbox
    coords = [v.co for v in mesh.vertices]

    min_x = min(v.x for v in coords)
    max_x = max(v.x for v in coords)
    min_y = min(v.y for v in coords)
    max_y = max(v.y for v in coords)
    min_z = min(v.z for v in coords)
    max_z = max(v.z for v in coords)

    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z

    max_dim = max(size_x, size_y, size_z)
    if max_dim == 0:
        return

    scale = 1.0 / max_dim

    for v in mesh.vertices:
        v.co *= scale

    mesh.update()

def normalize_object_by_bbox(obj):
    bpy.context.view_layer.update()

    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    size_x = max(v.x for v in bbox) - min(v.x for v in bbox)
    size_y = max(v.y for v in bbox) - min(v.y for v in bbox)
    size_z = max(v.z for v in bbox) - min(v.z for v in bbox)

    max_dim = max(size_x, size_y, size_z)

    if max_dim > 0:
        factor = 1.0 / max_dim
        obj.scale = (factor, factor, factor)
        bpy.context.view_layer.update()


def normalize_object_scale(obj):
    dims = obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)

    if max_dim > 0:
        factor = 1.0 / max_dim
        obj.scale = (
            obj.scale.x * factor,
            obj.scale.y * factor,
            obj.scale.z * factor
        )

        # Blender’ın internal transformlarını güncelle
        bpy.context.view_layer.update()
    
def lift_object_above_floor(obj, floor_z=0.0, margin=0.01):
    world_bbox = [
        obj.matrix_world @ mathutils.Vector(corner)
        for corner in obj.bound_box
    ]
    min_z = min(v.z for v in world_bbox)
    print("################")
    print(min_z)
    if min_z < floor_z:
        obj.location.z += (floor_z - min_z + margin)

def assign_color_material(obj, color, name):
    """
    color: [R, G, B] 0–255 aralığında
    """
    # 🔹 normalize
    r = color[0] / 255.0
    g = color[1] / 255.0
    b = color[2] / 255.0

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

# -------- argüman --------
# -- <scene.json>

FLOOR_WIDTH = 15.0
FLOOR_DEPTH = 15.0

WALL_WIDTH = 15.0
WALL_HEIGHT = 10.0



argv = sys.argv
argv = argv[argv.index("--") + 1:]
scene_path = argv[0]

if not os.path.exists(scene_path):
    raise RuntimeError("scene.json bulunamadı")

# -------- sahneyi temizle --------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

with open(scene_path, "r", encoding="utf-8") as f:
    scene_data = json.load(f)

# -------- zemin (floor) --------
bpy.ops.mesh.primitive_plane_add(
    size=1,
    location=(0.0, 0.0, 0.0)
)
floor = bpy.context.active_object
floor.name = "Floor"

# Plane default size=1 → 2x2, o yüzden scale yarısı
floor.scale = (
    FLOOR_WIDTH / 1.0,
    FLOOR_DEPTH / 1.0,
    1.0
)

assign_color_material(
    floor,
    scene_data["floor"]["color"],
    "FloorMaterial"
)

# -------- duvar (wall) --------
bpy.ops.mesh.primitive_plane_add(
    size=1,
    location=(
        -1* FLOOR_DEPTH / 2.0,
        0.0,   # zeminin arka kenarı
        WALL_HEIGHT / 4.0    # zeminden yukarı
    )
)
wall = bpy.context.active_object
wall.name = "Wall"

# XZ plane olacak şekilde döndür
wall.rotation_euler = (1.5708, 0.0, 1.5708)  # 90 derece X ekseni

wall.scale = (
    WALL_WIDTH ,
    WALL_HEIGHT / 2.0,
    1.0
)

assign_color_material(
    wall,
    scene_data["wall"]["color"],
    "WallMaterial"
)


objects = scene_data.get("objects", [])
if not objects:
    raise RuntimeError("scene.json içinde object yok")

# -------- objeleri yükle --------
for obj_data in objects:
    ply_path = obj_data["ply_path"]

    if not os.path.exists(ply_path):
        print(f"[WARN] Dosya yok: {ply_path}")
        continue

    # import
    bpy.ops.wm.ply_import(filepath=ply_path)
    obj = bpy.context.selected_objects[0]
    mesh = obj.data

    # ---------- 1) normalize ----------
    # 🔴 transform ile uğraşma
    obj.scale = (1,1,1)

    # ✅ mesh normalize
    normalize_mesh_to_unit_bbox(obj)
    obj_data["blender_obj"] = obj
    # artık JSON scale anlamlı
    s = obj_data["scale"]
    obj.scale = (s, s, s)

    obj.rotation_euler = (0.0, 0.0, 1.5708)
    # ---------- 3) location ----------
    loc = obj_data["location"]
    obj.location = (
        loc.get("x", 0.0),
        loc.get("y", 0.0),
        loc.get("z", 0.0)
    )

    bpy.context.view_layer.update()

    # ---------- 4) floor collision ----------
    if obj_data.get("placement") == "floor":
         print("xxxxxx")
         lift_object_above_floor(obj)

   
    # ---------- 5) material ----------
    if not mesh.color_attributes:
        print(f"[WARN] Vertex color yok: {ply_path}")
        continue

    color_attr = mesh.color_attributes[0].name

    mat = bpy.data.materials.new(name=f"VC_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = color_attr

    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.append(mat)
    
resolve_floor_collisions(objects)

# -------- viewport --------
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces[0].shading.type = 'MATERIAL'
