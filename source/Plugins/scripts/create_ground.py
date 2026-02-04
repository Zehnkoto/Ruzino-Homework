from pxr import UsdGeom, Gf, Sdf, UsdShade

# Get the USD stage (global stage object is already available)
usd_stage = stage.get_pxr_stage()

# Clean up existing ground from previous runs
ground_path = Sdf.Path("/Ground")
existing_prim = usd_stage.GetPrimAtPath(ground_path)
if existing_prim:
    usd_stage.RemovePrim(ground_path)
    print("Cleaned up existing ground")

# Create materials scope if it doesn't exist
materials_path = Sdf.Path("/Materials")
materials_prim = usd_stage.GetPrimAtPath(materials_path)
if not materials_prim:
    UsdGeom.Scope.Define(usd_stage, materials_path)

# Create ground plane using a cube
# Sphere grid is 10x10 with spacing 1.5, centered at origin
# Grid extends from -6.75 to 6.75 in x and y
# Make ground 20x20 to cover all spheres with margin
ground = UsdGeom.Cube.Define(usd_stage, ground_path)

# Set cube size (default is 2x2x2, we'll scale it)
ground_prim = usd_stage.GetPrimAtPath(ground_path)
xformable = UsdGeom.Xformable(ground_prim)

# Position ground at z=0 (below lowest sphere point at z=0.5)
# Scale it to be 20x20 in x,y and 0.2 thick in z
translate_op = xformable.AddTranslateOp()
translate_op.Set(Gf.Vec3d(0, 0, 0))

scale_op = xformable.AddScaleOp()
scale_op.Set(Gf.Vec3d(10, 10, 0.1))  # 20x20x0.2 ground plane

# Create material for ground
material_path = "/Materials/GroundMaterial"
material = UsdShade.Material.Define(usd_stage, material_path)

# Create UsdPreviewSurface shader
shader = UsdShade.Shader.Define(usd_stage, f"{material_path}/PreviewSurface")
shader.CreateIdAttr("UsdPreviewSurface")

# Create diffuse color input (light gray)
diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
diffuse_input.Set(Gf.Vec3f(0.7, 0.7, 0.7))

# Set roughness
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)

# Create output
shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

# Connect material output
material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))

# Bind material to ground
UsdShade.MaterialBindingAPI(ground_prim).Bind(material)

print("Created ground plane (20x20) at z=0")
