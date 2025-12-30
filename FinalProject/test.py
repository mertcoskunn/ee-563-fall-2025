import subprocess
import os

# Blender executable path
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
# Linux / Mac örnekleri:
# BLENDER_PATH = "/usr/bin/blender"
# BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"

# Blender içinde çalışacak script
BLENDER_SCRIPT = os.path.abspath("show_in_blender.py")

# Gösterilecek model (GLB / PLY)
MESH_PATH = os.path.abspath("output/couch_0.ply")
print(MESH_PATH)
print("###################33")
subprocess.run([
    BLENDER_PATH,
    "--python", "show_in_blender.py",
    "--",
    os.path.abspath("output")
])