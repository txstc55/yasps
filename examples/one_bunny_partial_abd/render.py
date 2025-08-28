import bpy, os, sys, argparse

def parse_cli():
    # Everything after `--` goes to Python
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="OBJ → PNG sequence renderer")
    ap.add_argument("--start", type=int, required=True, help="first frame index (inclusive)")
    ap.add_argument("--end",   type=int, required=True, help="last frame index (inclusive)")
    ap.add_argument("--step",  type=int, default=1,     help="step size for the range")
    ap.add_argument("--base_dir", default="/home/xuan/Desktop/research/yasps/examples/one_bunny_partial_abd/outputs")
    ap.add_argument("--file_pattern", default="bunny1_{:04d}.obj")
    ap.add_argument("--out_pattern",  default="bunny1_{:04d}.png")
    ap.add_argument("--n1", type=int, default=20)
    ap.add_argument("--n2", type=int, default=1500)
    return ap.parse_args(argv)

ARGS = parse_cli()

# Plug args into your existing vars
BASE_DIR     = ARGS.base_dir
FILE_PATTERN = ARGS.file_pattern
OUT_DIR      = BASE_DIR
OUT_PATTERN  = ARGS.out_pattern
N1, N2       = ARGS.n1, ARGS.n2
RANGE        = range(ARGS.start, ARGS.end + 1, ARGS.step)

# Existing material names (must already exist in the .blend)
MAT_NAME_A = "bunny_material_0"  # v < N1
MAT_NAME_B = "bunny_material_1"  # v < N2 (and none < N1)
MAT_NAME_C = "bunny_material_2"  # else

# ==============================
# Helpers (Blender 4.5+ OBJ import)
# ==============================
def import_obj(filepath: str):
    """Use Blender 4.5+ OBJ importer. Fallback to legacy if available."""
    if hasattr(bpy.ops.wm, "obj_import"):
        res = bpy.ops.wm.obj_import(filepath=filepath)
        if res != {'FINISHED'}:
            raise RuntimeError(f"wm.obj_import failed: {filepath}")
    elif hasattr(bpy.ops.import_scene, "obj"):
        res = bpy.ops.import_scene.obj(filepath=filepath)
        if res != {'FINISHED'}:
            raise RuntimeError(f"import_scene.obj failed: {filepath}")
    else:
        raise RuntimeError("No OBJ importer operator registered.")

def get_required_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        raise RuntimeError(f"Required material '{name}' not found in this file.")
    return mat

def ensure_material_slot(obj: bpy.types.Object, mat: bpy.types.Material) -> int:
    for idx, m in enumerate(obj.data.materials):
        if m == mat:
            return idx
    obj.data.materials.append(mat)
    return len(obj.data.materials) - 1

def remove_objects(objs):
    """Remove objects and their data if orphaned."""
    for o in objs:
        data = getattr(o, "data", None)
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass
        if data and hasattr(data, "users") and data.users == 0:
            try:
                # Mesh datablock cleanup
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                # (If other types ever appear, add branches here)
            except Exception:
                pass

def assign_materials_by_vertex_threshold(obj, n1, n2, matA, matB, matC):
    """Per-face assignment: if any vert < n1 -> A; elif any vert < n2 -> B; else C."""
    mesh = obj.data
    # Ensure in OBJECT mode
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    # (Re)attach only our three materials to this object (optional clean first)
    if obj.material_slots:
        for i in range(len(obj.material_slots)-1, -1, -1):
            obj.active_material_index = i
            bpy.ops.object.material_slot_remove()

    slot_A = ensure_material_slot(obj, matA)
    slot_B = ensure_material_slot(obj, matB)
    slot_C = ensure_material_slot(obj, matC)

    for poly in mesh.polygons:
        v_idx = poly.vertices
        if any(vid < n1 for vid in v_idx):
            poly.material_index = slot_A
        elif any(vid < n2 for vid in v_idx):
            poly.material_index = slot_B
        else:
            poly.material_index = slot_C

# ==============================
# Prep
# ==============================
os.makedirs(OUT_DIR, exist_ok=True)

scene = bpy.context.scene
# Make sure we write PNGs
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'  # keep alpha if you want

# Cache required materials once
mat_A = get_required_material(MAT_NAME_A)
mat_B = get_required_material(MAT_NAME_B)
mat_C = get_required_material(MAT_NAME_C)

last_imported_mesh_objs = []

# ==============================
# Main loop
# ==============================
for i in RANGE:
    # 1) Clean up previously imported objs
    if last_imported_mesh_objs:
        remove_objects(last_imported_mesh_objs)
        last_imported_mesh_objs = []

    # 2) Import new OBJ
    obj_path = os.path.join(BASE_DIR, FILE_PATTERN.format(i))
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    bpy.ops.object.select_all(action='DESELECT')
    import_obj(obj_path)

    # Collect imported mesh objects from selection/active
    imported_meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not imported_meshes and bpy.context.active_object and bpy.context.active_object.type == 'MESH':
        imported_meshes = [bpy.context.active_object]

    if not imported_meshes:
        raise RuntimeError(f"Imported '{obj_path}', but no mesh objects were detected.")

    # If multiple mesh objects were created, assign to each (common for some OBJs)
    for obj in imported_meshes:
        assign_materials_by_vertex_threshold(obj, N1, N2, mat_A, mat_B, mat_C)

    last_imported_mesh_objs = imported_meshes

    # 3) Render still to unique filepath
    png_path = os.path.join(OUT_DIR, OUT_PATTERN.format(i))
    scene.render.filepath = png_path
    bpy.ops.render.render(write_still=True)
    print(f"[OK] Rendered {png_path}")

# Optional: clear last import after finishing
# remove_objects(last_imported_mesh_objs)
print("[DONE] Sequence render complete.")
