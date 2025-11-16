"""
Test 3x3x3 tree grid with HIGHLY VISIBLE parameter variations (FAST version)
Based on Stava et al. 2014 paper - focusing on most influential parameters:
- X axis: Growth Years (tree age/maturity)
- Y axis: Lateral Buds (branching density)  
- Z axis: Apical Control (trunk dominance)
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


def test_tree_grid_3x3x3_visual():
    """Generate a 3x3x3 grid with maximum visual diversity - FAST"""
    print("\n" + "="*70)
    print("TEST: 3×3×3 Tree Grid - High Visual Variation (Fast)")
    print("="*70)
    
    binary_dir = get_binary_dir()
    output_file = os.path.join(binary_dir, "tree_grid_3x3x3_visual.usdc")
    
    g = RuzinoGraph("TreeGrid3x3x3Visual")
    
    # Load configurations
    g.loadConfiguration(os.path.join(binary_dir, "geometry_nodes.json"))
    print(f"✓ Loaded geometry nodes configuration")
    
    g.loadConfiguration(os.path.join(binary_dir, "Plugins", "TreeGen_geometry_nodes.json"))
    print(f"✓ Loaded TreeGen configuration")
    
    # Create merge node
    merge_node = g.createNode("node_merge_geometry", name="merge_all_trees")
    print(f"✓ Created merge_geometry node")
    
    # Create write node
    write_node = g.createNode("write_usd", name="writer")
    print(f"✓ Created write_usd node")
    
    # Connect merge to write
    g.addEdge(merge_node, "Geometry", write_node, "Geometry")
    
    # Grid spacing
    spacing = 15.0  # Larger spacing for better visualization
    
    # Parameter ranges - HIGHLY INFLUENTIAL but FAST (3x3x3 = 27 trees)
    growth_years_range = [2, 3, 4]  # X axis: young to mature
    lateral_buds_range = [2, 4, 6]  # Y axis: sparse to dense (BIG difference!)
    apical_control_range = [0.8, 2.0, 4.0]  # Z axis: weak trunk to strong trunk
    
    inputs = {}
    tree_count = 0
    
    print(f"\n{'='*70}")
    print(f"Generating 3×3×3 = 27 trees with HIGH visual variation...")
    print(f"X axis: Growth Years {growth_years_range} (maturity)")
    print(f"Y axis: Lateral Buds {lateral_buds_range} (branching density)")
    print(f"Z axis: Apical Control {apical_control_range} (trunk dominance)")
    print(f"{'='*70}\n")
    
    # Create 3x3x3 grid
    for x_idx, growth_years in enumerate(growth_years_range):
        for y_idx, lateral_buds in enumerate(lateral_buds_range):
            for z_idx, apical_control in enumerate(apical_control_range):
                tree_count += 1
                
                print(f"Creating tree {tree_count}/27: Years={growth_years}, Buds={lateral_buds}, Apical={apical_control:.1f}")
                
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
                inputs[(tree_gen, "Lateral Buds")] = int(lateral_buds)
                inputs[(tree_gen, "Apical Control")] = float(apical_control)
                
                # Keep other parameters at reasonable defaults
                inputs[(tree_gen, "Branch Angle")] = float(45.0)  # Default
                inputs[(tree_gen, "Growth Rate")] = float(2.5)  # Moderate
                inputs[(tree_gen, "Internode Length")] = float(0.8)
                
                inputs[(to_mesh, "Radial Segments")] = int(6)
                
                inputs[(transform, "Translate X")] = pos_x
                inputs[(transform, "Translate Y")] = pos_y
                inputs[(transform, "Translate Z")] = pos_z
    
    print(f"\n✓ Created all {tree_count} tree nodes in graph")
    
    # Create Stage and convert to GeomPayload
    stage = stage_py.Stage(output_file)
    geom_payload = stage_py.create_payload_from_stage(stage, "/tree_grid")
    g.setGlobalParams(geom_payload)
    
    print(f"\nExecuting graph (this should be fast)...")
    
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
        print(f"✅ SUCCESS: High Visual Variation Tree Grid Generated!")
        print(f"{'='*70}")
        print(f"Total trees: {tree_count}")
        print(f"File size: {file_size:,} bytes ({file_size_kb:.1f} KB)")
        print(f"Output: {output_file}")
        print(f"\nParameters varied (EXTREME ranges for visibility):")
        print(f"  - X: Growth Years 2→4 (young to mature)")
        print(f"  - Y: Lateral Buds 2→6 (sparse to VERY dense)")
        print(f"  - Z: Apical Control 0.8→4.0 (bushy to conifer-like)")
        print(f"\nExpected visual effects:")
        print(f"  - Low X, Low Y, Low Z: Small bushy tree")
        print(f"  - High X, High Y, Low Z: Large bushy tree")
        print(f"  - Low X, Low Y, High Z: Small pine-like tree")
        print(f"  - High X, High Y, High Z: Large dense conifer")
        print(f"{'='*70}")
        
        assert file_size > 50000, f"File unexpectedly small: {file_size} bytes"
        print(f"\n✓ File size validation passed")
        
    else:
        print(f"✗ USD file not found: {output_file}")
        assert False, f"File not created: {output_file}"


if __name__ == "__main__":
    test_tree_grid_3x3x3_visual()
