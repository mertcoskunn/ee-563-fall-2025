import bpy
import sys
import os


# -------- argüman --------
# -- <folder_path>
argv = sys.argv
argv = argv[argv.index("--") + 1:]
folder_path = argv[0]

# -------- sahneyi temizle --------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# -------- dosyaları al --------
files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith(".ply")
])

if not files:
    raise RuntimeError("Klasörde .ply dosyası yok")

spacing = 5.0
x_offset = 0.0

for fname in files:
    path = os.path.join(folder_path, fname)

    # import
    bpy.ops.wm.ply_import(filepath=path)

    obj = bpy.context.selected_objects[0]
    mesh = obj.data

    # konum
    obj.location.x = x_offset
    x_offset += spacing

    # vertex color kontrol
    if not mesh.color_attributes:
        print(f"[WARN] {fname} vertex color yok, atlandı")
        continue

    color_attr = mesh.color_attributes[0].name

    # material
    mat = bpy.data.materials.new(name=f"VC_{fname}")
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

# -------- viewport --------
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces[0].shading.type = 'MATERIAL'
