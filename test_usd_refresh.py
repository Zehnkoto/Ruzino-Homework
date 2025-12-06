import os
import sys

# Setup USD environment
binary_dir = r"C:\Users\Pengfei\WorkSpace\Ruzino\Binaries\Release"
os.environ['PXR_USD_WINDOWS_DLL_PATH'] = binary_dir
sys.path.insert(0, binary_dir)
os.chdir(binary_dir)

from pxr import Usd, Sdf, UsdMtlx

# Test 1: 在同一 session 中修改 reference
print("=== Test 1: Modify referenced layer in same session ===")
stage = Usd.Stage.CreateInMemory()
mat = stage.DefinePrim('/material_test', 'Material')

# 创建一个临时 MaterialX 文件
mtlx_path = r'C:\Users\Pengfei\WorkSpace\Ruzino\Assets\temp_test.mtlx'
with open(mtlx_path, 'w') as f:
    f.write('''<?xml version="1.0"?>
<materialx version="1.39">
  <surfacematerial name="material_test" type="material"/>
</materialx>''')

# 添加 reference
mat.GetReferences().AddReference(mtlx_path, '/MaterialX/Materials/material_test')
print(f"After reference: children = {[c.GetName() for c in mat.GetChildren()]}")

# 修改 MaterialX 文件添加 shader
with open(mtlx_path, 'w') as f:
    f.write('''<?xml version="1.0"?>
<materialx version="1.39">
  <surfacematerial name="material_test" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="test_shader"/>
  </surfacematerial>
  <standard_surface name="test_shader" type="surfaceshader"/>
</materialx>''')

print("\nAfter modifying .mtlx file:")
print(f"  children (no action) = {[c.GetName() for c in mat.GetChildren()]}")

# 尝试各种刷新方法
layer = Sdf.Layer.Find(mtlx_path)
if layer:
    print(f"\nFound layer, reloading...")
    layer.Reload()
    print(f"  children (after Reload) = {[c.GetName() for c in mat.GetChildren()]}")
else:
    print("\nLayer not found in registry")

# 尝试 SetActive
mat.SetActive(False)
mat.SetActive(True)
print(f"  children (after SetActive) = {[c.GetName() for c in mat.GetChildren()]}")

# 清理
os.remove(mtlx_path)

print("\n=== Test 2: Open fresh stage with modified file ===")
# 创建带 shader 的文件
with open(mtlx_path, 'w') as f:
    f.write('''<?xml version="1.0"?>
<materialx version="1.39">
  <surfacematerial name="material_test" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="test_shader"/>
  </surfacematerial>
  <standard_surface name="test_shader" type="surfaceshader"/>
</materialx>''')

# 验证文件内容
print(f"\nMaterialX file content:")
with open(mtlx_path, 'r') as f:
    print(f.read())

stage2 = Usd.Stage.CreateInMemory()
mat2 = stage2.DefinePrim('/material_test2', 'Material')
mat2.GetReferences().AddReference(mtlx_path, '/MaterialX/Materials/material_test')

# 检查 composition
print(f"\nComposition arcs: {mat2.GetPrimStack()}")
print(f"Has composition: {mat2.HasAuthoredReferences()}")

# 尝试打开 MaterialX layer 验证路径
mtlx_layer = Sdf.Layer.FindOrOpen(mtlx_path)
if mtlx_layer:
    print(f"\n.mtlx opened as layer: {mtlx_layer.identifier}")
    temp_stage = Usd.Stage.Open(mtlx_layer)
    print(f"Prims in .mtlx stage:")
    for p in temp_stage.Traverse():
        print(f"  {p.GetPath()} ({p.GetTypeName()})")
        if p.GetPath() == '/MaterialX/Materials/material_test':
            children = list(p.GetChildren())
            print(f"    Children: {[c.GetName() for c in children]}")
            # 检查是否有 shader 属性
            for attr in p.GetAttributes():
                print(f"    Attr: {attr.GetName()} = {attr.Get()}")
else:
    print("\nERROR: Cannot open .mtlx as SdfLayer!")

print(f"\nFresh stage: children = {[c.GetName() for c in mat2.GetChildren()]}")

os.remove(mtlx_path)
