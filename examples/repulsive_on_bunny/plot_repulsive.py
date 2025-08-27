import bpy, os

# ==============================
# USER SETTINGS
# ==============================
USE_ACTIVE_OBJECT = False   # we’re importing per-iteration
LINE_OBJ_PATH = "/home/xuan/Desktop/research/yasps/examples/repulsive_on_bunny/mapped/mapped_loop_000083.obj"
OUTPUT_DIR = "/home/xuan/Desktop/research/yasps/examples/repulsive_on_bunny/mapped/"    # use an absolute path if you prefer

# Geometry look & feel
RESAMPLE_COUNT = 100000
RADIUS_START   = 0.010
RADIUS_END     = 0.002
PROFILE_RADIUS = 0.002
USE_RIBBON     = False     # True => flat ribbon profile

# Material look & feel
USE_DASH           = True
DASH_REPEAT        = 30.0
DASH_LENGTH        = 0.40
EMISSION_STRENGTH  = 2.0
COLOR_A = (0, 0.655023, 1, 1)  # RGBA
COLOR_B = (0, 0.655023, 1, 1)

# ==============================
# Helpers
# ==============================
# ---- Put this near the top (after imports), or anywhere above render_sequence() ----
def use_cycles(device_preference="GPU", samples=256, preview_samples=64, denoise=True):
    """Configure Cycles (Blender 4.5.2-safe). Tries GPU; falls back to CPU."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    c = scene.cycles
    # Device (scene-level)
    if hasattr(c, "device"):
        c.device = 'GPU' if device_preference.upper() == "GPU" else 'CPU'

    # Try to prefer OPTIX/Metal/CUDA if available (won't crash if not)
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        if bpy.app.build_options.optix:
            prefs.compute_device_type = 'OPTIX'
        elif bpy.app.build_options.cuda:
            prefs.compute_device_type = 'CUDA'
        elif bpy.app.build_options.metal:
            prefs.compute_device_type = 'METAL'
    except Exception:
        pass  # preferences may be locked; that's fine

    # Sampling
    c.use_adaptive_sampling = True
    c.samples = samples               # final render
    c.preview_samples = preview_samples

    # Denoising
    if denoise:
        # Render denoiser (final)
        if hasattr(c, "denoiser"):
            # 'OPENIMAGEDENOISE' (CPU) is reliable; OPTIX is fast on RTX
            c.denoiser = 'OPENIMAGEDENOISE'
        # Per-view layer denoise toggle
        if hasattr(bpy.context.view_layer, "cycles"):
            bpy.context.view_layer.cycles.use_denoising = True

    # Bounce limits (a bit conservative for speed)
    c.max_bounces = 12
    c.diffuse_bounces = 4
    c.glossy_bounces = 4
    c.transmission_bounces = 8
    c.transparent_max_bounces = 8

    # Reduce fireflies (esp. with glass)
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.sample_clamp_indirect = 2.0     # 0 disables clamping

    # Film settings (optional)
    # scene.view_layers["View Layer"].use_pass_combined = True
    # scene.render.film_transparent = True  # enable if you want a transparent bg

    print("✓ Cycles configured (device:", getattr(c, "device", "unknown"), ")")


def set_material_flags_version_safe(mat):
    if hasattr(mat, "blend_method"):
        mat.blend_method = 'BLEND'
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = 'NONE'
    elif hasattr(mat, "shadow_mode"):
        mat.shadow_mode = 'NONE'
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False

def enable_bloom_version_safe():
    try:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except Exception:
        try:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        except Exception:
            return
    eevee = getattr(bpy.context.scene, "eevee", None)
    if not eevee:
        return
    for name, value in (
        ("use_bloom", True),
        ("bloom_intensity", 0.05),
        ("bloom_threshold", 0.8),
        ("bloom_radius", 6.5),
        ("bloom_color", (1.0, 1.0, 1.0)),
        ("bloom_clamp", 0.0),
        ("bloom_knee", 0.5),
    ):
        if hasattr(eevee, name):
            try:
                setattr(eevee, name, value)
            except Exception:
                pass

def link(tree, a, a_out, b, b_in):
    """Robust link helper: accepts socket names or indices for both ends."""
    out_sock = a.outputs[a_out] if isinstance(a_out, str) else a.outputs[int(a_out)]
    in_sock  = b.inputs[b_in]   if isinstance(b_in, str)  else b.inputs[int(b_in)]
    tree.links.new(out_sock, in_sock)

def import_obj(filepath):
    """Import an OBJ and return the list of newly created objects (robust)."""
    before = set(bpy.data.objects)
    # 4.x importer, fallback to legacy if needed
    try:
        bpy.ops.wm.obj_import(filepath=filepath)
    except Exception:
        bpy.ops.import_scene.obj(filepath=filepath)
    after = set(bpy.data.objects)
    new_objs = list(after - before)
    # Make sure something is active for downstream ops
    if new_objs:
        for o in new_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new_objs[0]
    return new_objs

# ==============================
# Geometry Nodes (Blender 4.5.2)
# ==============================
def build_geo_nodes(obj, material, name="Line_Geo"):
    # Replace existing group (avoids editing interface in place)
    old = bpy.data.node_groups.get(name)
    if old:
        bpy.data.node_groups.remove(old, do_unlink=True)
    ng = bpy.data.node_groups.new(name=name, type='GeometryNodeTree')

    # ---- Interface: define sockets fresh
    iface = ng.interface
    iface.new_socket(name="Geometry",       in_out='INPUT',  socket_type='NodeSocketGeometry')
    iface.new_socket(name="Resample Count", in_out='INPUT',  socket_type='NodeSocketInt').default_value  = RESAMPLE_COUNT
    iface.new_socket(name="Radius Start",   in_out='INPUT',  socket_type='NodeSocketFloat').default_value = RADIUS_START
    iface.new_socket(name="Radius End",     in_out='INPUT',  socket_type='NodeSocketFloat').default_value = RADIUS_END
    iface.new_socket(name="Profile Radius", in_out='INPUT',  socket_type='NodeSocketFloat').default_value = PROFILE_RADIUS
    iface.new_socket(name="Geometry",       in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # Group IO nodes
    n_in  = ng.nodes.new("NodeGroupInput");  n_in.location  = (-1200, 0)
    n_out = ng.nodes.new("NodeGroupOutput"); n_out.location = (900, 0)

    # Mesh → Curve
    n_mesh_to_curve = ng.nodes.new("GeometryNodeMeshToCurve"); n_mesh_to_curve.location = (-1000, 0)
    link(ng, n_in, "Geometry", n_mesh_to_curve, "Mesh")

    # Resample
    n_resample = ng.nodes.new("GeometryNodeResampleCurve"); n_resample.location = (-800, 0)
    n_resample.mode = 'COUNT'
    link(ng, n_in, "Resample Count", n_resample, "Count")
    link(ng, n_mesh_to_curve, "Curve", n_resample, "Curve")

    # Along-curve parameter t
    n_spline_param = ng.nodes.new("GeometryNodeSplineParameter"); n_spline_param.location = (-820, -240)

    # Taper: map t (0..1) to radius in [End..Start], then multiply by Profile Radius
    n_map = ng.nodes.new("ShaderNodeMapRange"); n_map.location = (-600, -240)
    n_map.inputs["From Min"].default_value = 0.0
    n_map.inputs["From Max"].default_value = 1.0
    link(ng, n_in, "Radius End",   n_map, "To Min")
    link(ng, n_in, "Radius Start", n_map, "To Max")
    link(ng, n_spline_param, "Factor", n_map, "Value")

    n_mul_profile = ng.nodes.new("ShaderNodeMath"); n_mul_profile.location = (-400, -240)
    n_mul_profile.operation = 'MULTIPLY'
    link(ng, n_map, "Result", n_mul_profile, 0)
    link(ng, n_in, "Profile Radius", n_mul_profile, 1)

    n_set_radius = ng.nodes.new("GeometryNodeSetCurveRadius"); n_set_radius.location = (-580, 0)
    link(ng, n_resample, "Curve", n_set_radius, "Curve")
    link(ng, n_mul_profile, 0, n_set_radius, "Radius")  # math->Value (idx 0)

    # Profile: ribbon or circle
    if USE_RIBBON:
        n_profile = ng.nodes.new("GeometryNodeCurvePrimitiveLine"); n_profile.location = (-420, -420)
        n_profile.inputs["Start"].default_value = (0.0, 0.0, 0.0)
        n_profile.inputs["End"].default_value   = (0.0, 0.0, 0.001)
        profile_socket = ("Curve", n_profile)
    else:
        n_profile = ng.nodes.new("GeometryNodeCurvePrimitiveCircle"); n_profile.location = (-420, -420)
        n_profile.inputs["Radius"].default_value = 0.2  # actual thickness comes from Set Curve Radius
        profile_socket = ("Curve", n_profile)

    n_curve_to_mesh = ng.nodes.new("GeometryNodeCurveToMesh"); n_curve_to_mesh.location = (-220, 0)
    link(ng, n_set_radius, "Curve", n_curve_to_mesh, "Curve")
    link(ng, profile_socket[1], profile_socket[0], n_curve_to_mesh, "Profile Curve")
    n_curve_to_mesh.inputs["Fill Caps"].default_value = True

    # Store along-curve factor "t" directly
    n_store = ng.nodes.new("GeometryNodeStoreNamedAttribute"); n_store.location = (200, 0)
    if hasattr(n_store, "domain"):    n_store.domain    = 'POINT'
    if hasattr(n_store, "data_type"): n_store.data_type = 'FLOAT'
    n_store.inputs["Name"].default_value = "t"
    link(ng, n_curve_to_mesh, "Mesh", n_store, "Geometry")
    link(ng, n_spline_param, "Factor", n_store, "Value")

    # Smooth shading
    n_set_smooth = ng.nodes.new("GeometryNodeSetShadeSmooth"); n_set_smooth.location = (420, 0)
    n_set_smooth.inputs["Shade Smooth"].default_value = True
    link(ng, n_store, "Geometry", n_set_smooth, "Geometry")

    # Set Material
    n_set_mat = ng.nodes.new("GeometryNodeSetMaterial"); n_set_mat.location = (620, 0)
    n_set_mat.inputs["Material"].default_value = material
    link(ng, n_set_smooth, "Geometry", n_set_mat, "Geometry")

    # Realize (safe)
    n_realize = ng.nodes.new("GeometryNodeRealizeInstances"); n_realize.location = (760, 0)
    link(ng, n_set_mat, "Geometry", n_realize, "Geometry")

    # Group output
    link(ng, n_realize, "Geometry", n_out, "Geometry")

    # Attach modifier
    mod = obj.modifiers.get(name) or obj.modifiers.new(name=name, type='NODES')
    mod.node_group = ng

    # Ensure the material is in the object's slots (check by name)
    ms = obj.data.materials
    if ms.get(material.name) is None:
        ms.append(material)

    return ng

# ==============================
# Material (Glass + Emission + optional dash mask)
# ==============================
def ensure_material(name="Line_FX"):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    return mat

def build_material(mat):
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    def nlink(a, a_out, b, b_in):
        out_sock = a.outputs[a_out] if isinstance(a_out, str) else a.outputs[int(a_out)]
        in_sock  = b.inputs[b_in]   if isinstance(b_in, str)  else b.inputs[int(b_in)]
        nt.links.new(out_sock, in_sock)

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (900, 0)

    # Attribute "t"
    n_attr = nt.nodes.new("ShaderNodeAttribute"); n_attr.location = (-820, 80)
    n_attr.attribute_name = "t"

    # Gradient along t
    n_ramp = nt.nodes.new("ShaderNodeValToRGB"); n_ramp.location = (-560, 80)
    n_ramp.color_ramp.elements[0].position = 0.0
    n_ramp.color_ramp.elements[0].color = COLOR_A
    n_ramp.color_ramp.elements[1].position = 1.0
    n_ramp.color_ramp.elements[1].color = COLOR_B

    # Emission
    n_emis = nt.nodes.new("ShaderNodeEmission"); n_emis.location = (160, 140)
    n_emis.inputs["Strength"].default_value = EMISSION_STRENGTH

    # Fresnel rim
    n_fres = nt.nodes.new("ShaderNodeFresnel"); n_fres.location = (-150, -180)
    n_fres.inputs["IOR"].default_value = 1.25
    n_fres_ramp = nt.nodes.new("ShaderNodeValToRGB"); n_fres_ramp.location = (40, -180)
    n_fres_ramp.color_ramp.elements[0].position = 0.6
    n_fres_ramp.color_ramp.elements[1].position = 0.95

    # Glass instead of Transparent
    n_glass = nt.nodes.new("ShaderNodeBsdfGlass"); n_glass.location = (420, -140)
    n_glass.inputs["Color"].default_value = (0.064, 0.468, 1.0, 1.0)
    n_glass.inputs["Roughness"].default_value = 0.85
    n_glass.inputs["IOR"].default_value = 100.0

    # Geometry node
    n_geom = nt.nodes.new("ShaderNodeNewGeometry")
    n_geom.location = (200, -300)

    # Link True Normal → Glass Normal
    nt.links.new(n_geom.outputs["True Normal"], n_glass.inputs["Normal"])

    n_mix_main = nt.nodes.new("ShaderNodeMixShader"); n_mix_main.location = (640, 20)

    if USE_DASH:
        # Dash mask: FRACT(t*repeat) < length
        n_mul   = nt.nodes.new("ShaderNodeMath"); n_mul.location   = (-820, -160); n_mul.operation = 'MULTIPLY'
        n_mul.inputs[1].default_value = DASH_REPEAT
        n_fract = nt.nodes.new("ShaderNodeMath"); n_fract.location = (-640, -160); n_fract.operation = 'FRACT'
        n_lt    = nt.nodes.new("ShaderNodeMath"); n_lt.location    = (-460, -160); n_lt.operation = 'LESS_THAN'
        n_lt.inputs[1].default_value = DASH_LENGTH
        n_soft  = nt.nodes.new("ShaderNodeMath"); n_soft.location  = (-280, -160); n_soft.operation = 'SMOOTH_MIN'
        n_soft.inputs[2].default_value = 0.05

        # Combine with fresnel
        n_mul_df = nt.nodes.new("ShaderNodeMath"); n_mul_df.location = (240, -40); n_mul_df.operation = 'MULTIPLY'

        nlink(n_attr, "Fac", n_mul, 0)
        nlink(n_mul,  0, n_fract, 0)
        nlink(n_fract, 0, n_lt, 0)
        nlink(n_lt, 0, n_soft, 0)

        nlink(n_fres, "Fac", n_fres_ramp, "Fac")
        nlink(n_soft, 0, n_mul_df, 0)
        nlink(n_fres_ramp, "Color", n_mul_df, 1)

        # Gradient → emission
        nlink(n_attr, "Fac", n_ramp, "Fac")
        nlink(n_ramp, "Color", n_emis, "Color")

        # Invert mask so "on" parts are emission (1 - mask)
        n_inv = nt.nodes.new("ShaderNodeMath"); n_inv.location = (420, -20); n_inv.operation = 'LOGARITHM'
        n_inv.inputs[0].default_value = 1.0
        nlink(n_mul_df, 0, n_inv, 1)

        nlink(n_glass, "BSDF", n_mix_main, 1)
        nlink(n_emis,  "Emission", n_mix_main, 2)
        nlink(n_inv,   0, n_mix_main, "Fac")
    else:
        # No dashes: mix emission with glass by fresnel
        n_mix_f = nt.nodes.new("ShaderNodeMixShader"); n_mix_f.location = (420, 20)
        nlink(n_attr, "Fac", n_ramp, "Fac")
        nlink(n_ramp, "Color", n_emis, "Color")
        nlink(n_fres, "Fac", n_fres_ramp, "Fac")
        nlink(n_glass, "BSDF", n_mix_f, 1)
        nlink(n_emis,  "Emission", n_mix_f, 2)
        nlink(n_fres_ramp, "Color", n_mix_f, "Fac")
        n_mix_main = n_mix_f

    nt.links.new(n_mix_main.outputs["Shader"], out.inputs["Surface"])

    # enable_bloom_version_safe()
    set_material_flags_version_safe(mat)
    return mat

# ==============================
# One-shot setup for a given OBJ path
# ==============================
def setup_line_from_obj(filepath):
    """Import a line OBJ, build material + geo-nodes, return list of imported objects."""
    imported = import_obj(filepath)
    if not imported:
        raise RuntimeError(f"Failed to import: {filepath}")
    mat = ensure_material("Line_FX")
    build_material(mat)
    # Apply to first imported object (typical for line OBJs)
    build_geo_nodes(imported[0], mat)
    return imported

# ==============================
# Batch render loop
# ==============================
def render_sequence(start_idx=0, end_idx=200):
    # Resolve base folder from the given path, build file pattern
    base_dir = os.path.dirname(bpy.path.abspath(LINE_OBJ_PATH))
    file_pattern = os.path.join(base_dir, "mapped", "mapped_loop_{:06d}.obj") \
        if os.path.basename(os.path.dirname(LINE_OBJ_PATH)) != "mapped" \
        else os.path.join(base_dir, "mapped_loop_{:06d}.obj")

    # Output folder
    out_dir = bpy.path.abspath(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    last_imported = []  # keep references to remove only these next round

    for i in range(start_idx, end_idx):
        # --- cleanup previous line meshes ONLY
        if last_imported:
            for obj in last_imported:
                try:
                    data = obj.data if hasattr(obj, "data") else None
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if data and hasattr(data, "users") and data.users == 0:
                        # remove orphaned mesh datablock
                        try:
                            bpy.data.meshes.remove(data)
                        except Exception:
                            pass
                except Exception:
                    pass
            last_imported = []

        # --- import next
        path = file_pattern.format(i)
        if not os.path.exists(path):
            print(f"⚠️  Skipping {i:06d}: file not found -> {path}")
            continue

        print(f"→ Loading {path}")
        last_imported = setup_line_from_obj(path)

        # --- render still
        bpy.context.scene.render.filepath = os.path.join(out_dir, f"frame_{i:06d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"✓ Rendered frame_{i:06d}.png")

    print("✅ Batch render complete.")

# ==============================
# Run the batch (0..83)
# ==============================
# ---- Call this once before render_sequence() ----
use_cycles(device_preference="GPU", samples=1024, preview_samples=64, denoise=True)
render_sequence(0, 2)
