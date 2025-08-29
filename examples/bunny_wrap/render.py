import bpy
import os

# ==============================
# CONFIG — EDIT THESE
# ==============================
OBJ_DIR     = "/home/xuan/Desktop/research/yasps/examples/bunny_wrap/outputs"  # folder with sphere_0000.obj, etc.
OUT_DIR     = "/home/xuan/Desktop/research/yasps/examples/bunny_wrap/outputs"  # render output folder
START_INDEX = 0
END_INDEX   = 2                                   # inclusive; loads 0..2
OBJ_PATTERN = "sphere_{:04d}.obj"                 # file name pattern
OUTPUT_NAME = "frame_{:04d}.png"                  # per-frame render name

GEONODES_GROUP_NAME = "FaceAreaNodes"             # must exist in the .blend
MATERIAL_NAME       = "membrane"                  # must exist in the .blend

# OBJ import options (leave as-is unless you need axis tweaks)
OBJ_IMPORT_KW = dict()  # e.g., {"forward_axis": 'NEGATIVE_Z', "up_axis": 'Y'}

# ==============================
# VALIDATION
# ==============================
gn_group = bpy.data.node_groups.get(GEONODES_GROUP_NAME)
if gn_group is None:
    raise RuntimeError(f"Geometry Node Group '{GEONODES_GROUP_NAME}' not found.")

membrane_mat = bpy.data.materials.get(MATERIAL_NAME)
if membrane_mat is None:
    raise RuntimeError(f"Material '{MATERIAL_NAME}' not found.")

os.makedirs(OUT_DIR, exist_ok=True)
scene = bpy.context.scene

# ==============================
# HELPERS
# ==============================
def import_obj(filepath: str):
    """Import an OBJ (Blender 4.5+) and return the newly created objects."""
    before = set(bpy.data.objects)
    res = bpy.ops.wm.obj_import(filepath=filepath, **OBJ_IMPORT_KW)
    if 'CANCELLED' in res:
        raise RuntimeError(f"Import cancelled or failed for: {filepath}")
    after = set(bpy.data.objects)
    created = [ob for ob in (after - before)]
    return created

def shade_smooth_and_auto(ob):
    """
    Make mesh faces smooth, then apply 'Shade Auto Smooth' (by angle) in 4.5+.
    Uses an override context for the operator; falls back to Edge Split if needed.
    """
    if ob.type != 'MESH':
        return
    me = ob.data

    # Smooth shading without operators (no context needed)
    if len(me.polygons):
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))

    # Ensure OBJECT mode & selection for operator
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob

    # Build a VIEW_3D override (some environments require it)
    override = bpy.context.copy()
    override["active_object"] = ob
    override["object"] = ob
    override["selected_objects"] = [ob]
    # Try to attach a 3D view area/region
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            override["area"] = area
            for region in area.regions:
                if region.type == 'WINDOW':
                    override["region"] = region
                    break
            break

    # Try the operator (preferred in 4.x)
    try:
        bpy.ops.object.shade_auto_smooth(override, angle=1.0471975512)  # 60° in radians
    except Exception as e:
        # Fallback: Edge Split modifier to simulate auto-smooth by angle
        try:
            es = ob.modifiers.new(name="Autosmooth_Fallback", type='EDGE_SPLIT')
            es.use_edge_angle = True
            es.split_angle = 1.0471975512
        except Exception as ee:
            print(f"[WARN] Auto smooth failed for {ob.name}: {e} / Fallback failed: {ee}")

def apply_geo_and_material(objs, node_group, material):
    """Add geometry-nodes modifier, assign material, and enable smooth shading + auto smooth."""
    for ob in objs:
        if ob.type != 'MESH':
            continue

        # Add Geo Nodes modifier
        mod = ob.modifiers.new(name="FaceArea", type='NODES')
        mod.node_group = node_group

        # Assign/replace material(s)
        me = ob.data
        if not me.materials:
            me.materials.append(material)
        else:
            for i in range(len(me.materials)):
                me.materials[i] = material

        # Shade Smooth + Auto Smooth
        shade_smooth_and_auto(ob)

def remove_objects(objs):
    """Remove ONLY the given objects (safe for the rest of the scene)."""
    # Deselect everything
    bpy.ops.object.select_all(action='DESELECT')

    # Select targets
    for ob in objs:
        if ob and ob.name in bpy.data.objects:
            try:
                ob.select_set(True)
            except:
                pass

    # Remove selected objects and unlink data
    for ob in objs:
        try:
            bpy.data.objects.remove(ob, do_unlink=True)
        except:
            pass

    # # Optionally purge orphans (comment out if you want to keep data blocks)
    # try:
    #     bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=False, do_recursive=True)
    # except:
    #     pass

# ==============================
# MAIN LOOP
# ==============================
last_imported = []

for i in range(START_INDEX, END_INDEX + 1):
    # 1) Cleanup previous iteration (ONLY previously imported objs)
    if last_imported:
        remove_objects(last_imported)
        last_imported = []

    # 2) Import this iteration's OBJ
    obj_path = os.path.join(OBJ_DIR, OBJ_PATTERN.format(i))
    if not os.path.isfile(obj_path):
        print(f"[WARN] File not found, skipping: {obj_path}")
        continue

    print(f"[INFO] Importing: {obj_path}")
    new_objs = import_obj(obj_path)

    # (Optional) rename imported objects to match iteration
    for ob in new_objs:
        try:
            ob.name = f"sphere_{i:04d}_{ob.name}"
        except:
            pass

    # 3) Apply Geo Nodes + Material + Smooth shading
    apply_geo_and_material(new_objs, gn_group, membrane_mat)

    # 4) Render still
    scene.render.filepath = os.path.join(OUT_DIR, OUTPUT_NAME.format(i))
    print(f"[INFO] Rendering to: {scene.render.filepath}")
    bpy.ops.render.render(write_still=True)

    # 5) Keep reference to delete next round (so your camera/lights/etc. are safe)
    last_imported = new_objs

print("[DONE] All iterations complete.")
