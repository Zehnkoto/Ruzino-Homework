"""
Test smaller tree grid (3x3x3) to debug the assertion issue
"""
import os
from ruzino_graph import RuzinoGraph
import stage_py
import geometry_py


def get_binary_dir():
    """Get the binary directory path"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    binary_dir = os.path.join(test_dir, '..', '..', '..', '..', 'Binaries', 'Debug')
    return os.path.abspath(binary_dir)


def test_tree_grid_3x3x3():
    """Generate a 3x3x3 grid of trees (27 trees total)"""
    print("\n" + "="*70)
    print("TEST: 3×3×3 Tree Grid Generation (Debug)")
    print("="*70)
    
    binary_dir = get_binary_dir()
    output_file = os.path.join(binary_dir, "tree_grid_3x3x3.usdc")
    
    g = RuzinoGraph("TreeGrid3x3x3")
    
    # Load configurations
    g.loadConfiguration(os.path.join(binary_dir, "geometry_nodes.json"))
    print(f"✓ Loaded geometry nodes configuration")
    
    g.loadConfiguration(os.path.join(binary_dir, "Plugins", "TreeGen_geometry_nodes.json"))
    print(f"✓ Loaded TreeGen configuration")
    
    # Create merge node to combine all trees
    merge_node = g.createNode("node_merge_geometry", name="merge_all_trees")
    print(f"✓ Created merge_geometry node")
    
    # Create write node
    write_node = g.createNode("write_usd", name="writer")
    print(f"✓ Created write_usd node")
    
    # Connect merge to write
    g.addEdge(merge_node, "Geometry", write_node, "Geometry")
    
    # Grid spacing
    spacing = 15.0
    
    # Parameter ranges (3 values each)
    growth_years_range = [2, 3, 4]  # X axis
    branch_angle_range = [25, 40, 55]  # Y axis (degrees)
    apical_control_range = [0.4, 0.6, 0.8]  # Z axis
    
    inputs = {}
    tree_count = 0
    
    print(f"\nGenerating 3×3×3 = 27 trees...")
    print(f"X axis: Growth Years {growth_years_range}")
    print(f"Y axis: Branch Angle {branch_angle_range}")
    print(f"Z axis: Apical Control {apical_control_range}\n")
    
    # Create 3x3x3 grid
    for x_idx, growth_years in enumerate(growth_years_range):
        for y_idx, branch_angle in enumerate(branch_angle_range):
            for z_idx, apical_control in enumerate(apical_control_range):
                tree_count += 1
                
                print(f"Creating tree {tree_count}/27: ({x_idx},{y_idx},{z_idx}) Years={growth_years}, Angle={branch_angle}, Apical={apical_control}")
                
                # Create tree generation node
                tree_gen = g.createNode("tree_generate", name=f"tree_{x_idx}_{y_idx}_{z_idx}")
                
                # Create mesh conversion node
                to_mesh = g.createNode("tree_to_mesh", name=f"mesh_{x_idx}_{y_idx}_{z_idx}")
                
                # Create transform node for positioning
                transform = g.createNode("transform_geom", name=f"transform_{x_idx}_{y_idx}_{z_idx}")
                
                # Connect: tree_generate -> tree_to_mesh -> transform -> merge
                g.addEdge(tree_gen, "Tree Branches", to_mesh, "Tree Branches")
                g.addEdge(to_mesh, "Mesh", transform, "Geometry")
                g.addEdge(transform, "Geometry", merge_node, "Geometries")
                
                # Calculate position
                pos_x = float(x_idx * spacing)
                pos_y = float(y_idx * spacing)
                pos_z = float(z_idx * spacing)
                
                # Set parameters - explicitly use float/int types
                inputs[(tree_gen, "Growth Years")] = int(growth_years)
                inputs[(tree_gen, "Branch Angle")] = float(branch_angle)
                inputs[(tree_gen, "Apical Control")] = float(apical_control)
                inputs[(tree_gen, "Internode Length")] = float(1.0)
                
                inputs[(to_mesh, "Radial Segments")] = int(6)
                
                inputs[(transform, "Translate X")] = pos_x
                inputs[(transform, "Translate Y")] = pos_y
                inputs[(transform, "Translate Z")] = pos_z
    
    print(f"\n✓ Created all {tree_count} tree nodes in graph")
    
    # Create Stage and convert to GeomPayload
    stage = stage_py.Stage(output_file)
    geom_payload = stage_py.create_payload_from_stage(stage, "/tree_grid")
    g.setGlobalParams(geom_payload)
    
    print(f"\nExecuting graph...")
    
    # Execute
    g.prepare_and_execute(inputs, required_node=write_node)
    print(f"✓ Executed graph")
    
    # Save the stage
    stage.save()
    
    # Check file size
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        file_size_kb = file_size / 1024
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS: 3×3×3 Tree Grid Generated!")
        print(f"{'='*70}")
        print(f"Total trees: {tree_count}")
        print(f"File size: {file_size:,} bytes ({file_size_kb:.2f} KB)")
        print(f"Output: {output_file}")
        print(f"{'='*70}")
        
        assert file_size > 50000, f"File unexpectedly small: {file_size} bytes"
        
    else:
        print(f"✗ USD file not found: {output_file}")
        assert False, f"File not created: {output_file}"


if __name__ == "__main__":
    test_tree_grid_3x3x3()
