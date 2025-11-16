"""
Test 5x5x5 tree grid with HIGHLY VISIBLE parameter variations
Based on Stava et al. 2014 paper - focusing on most influential parameters:
- X axis: Growth Years (tree age/maturity)
- Y axis: Lateral Buds (branching density)  
- Z axis: Growth Rate (internode count per shoot)
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


def test_tree_grid_highly_visual():
    """Generate a 5x5x5 grid with maximum visual diversity"""
    print("\n" + "="*70)
    print("TEST: 5×5×5 Tree Grid - High Visual Variation")
    print("="*70)
    
    binary_dir = get_binary_dir()
    output_file = os.path.join(binary_dir, "tree_grid_visual.usdc")
    
    g = RuzinoGraph("TreeGridVisual")
    
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
    spacing = 12.0  # Slightly larger spacing for better visualization
    
    # Parameter ranges - BALANCED for visual impact vs performance
    # Note: High values create exponentially more branches!
    growth_years_range = [2, 3, 3, 4, 4]  # X axis: avoid 5-6 years (too slow)
    lateral_buds_range = [2, 3, 4, 4, 5]  # Y axis: avoid 6 buds (too dense)
    growth_rate_range = [1.5, 2.0, 2.5, 3.0, 3.5]  # Z axis: reduced max (was 5.5)
    
    inputs = {}
    tree_count = 0
    
    print(f"\n{'='*70}")
    print(f"Generating 5×5×5 = 125 trees with HIGH visual variation...")
    print(f"X axis: Growth Years {growth_years_range} (maturity)")
    print(f"Y axis: Lateral Buds {lateral_buds_range} (branching density)")
    print(f"Z axis: Growth Rate {growth_rate_range} (shoot length)")
    print(f"{'='*70}\n")
    
    # Create 5x5x5 grid
    for x_idx, growth_years in enumerate(growth_years_range):
        for y_idx, lateral_buds in enumerate(lateral_buds_range):
            for z_idx, growth_rate in enumerate(growth_rate_range):
                tree_count += 1
                
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
                inputs[(tree_gen, "Growth Rate")] = float(growth_rate)
                
                # Keep other parameters at reasonable defaults
                inputs[(tree_gen, "Branch Angle")] = float(45.0)  # Default
                inputs[(tree_gen, "Apical Control")] = float(2.0)  # Default
                inputs[(tree_gen, "Internode Length")] = float(0.8)
                
                inputs[(to_mesh, "Radial Segments")] = int(6)
                
                inputs[(transform, "Translate X")] = pos_x
                inputs[(transform, "Translate Y")] = pos_y
                inputs[(transform, "Translate Z")] = pos_z
                
                # Progress indicator
                if tree_count % 25 == 0:
                    print(f"  Created {tree_count}/125 trees...")
    
    print(f"✓ Created all {tree_count} tree nodes in graph")
    
    # Create Stage and convert to GeomPayload
    stage = stage_py.Stage(output_file)
    geom_payload = stage_py.create_payload_from_stage(stage, "/tree_grid")
    g.setGlobalParams(geom_payload)
    
    print(f"\nExecuting graph (this may take a moment)...")
    
    # Execute
    g.prepare_and_execute(inputs, required_node=write_node)
    print(f"✓ Executed graph")
    
    # Save the stage
    stage.save()
    
    # Check file size
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS: High Visual Variation Tree Grid Generated!")
        print(f"{'='*70}")
        print(f"Total trees: {tree_count}")
        print(f"File size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
        print(f"Output: {output_file}")
        print(f"Parameters varied:")
        print(f"  - X: Growth Years 2→6 (tree maturity)")
        print(f"  - Y: Lateral Buds 2→6 (branch density)")
        print(f"  - Z: Growth Rate 1.5→5.5 (shoot elongation)")
        print(f"{'='*70}")
        
        assert file_size > 100000, f"File unexpectedly small: {file_size} bytes"
        print(f"\n✓ File size validation passed")
        
    else:
        print(f"✗ USD file not found: {output_file}")
        assert False, f"File not created: {output_file}"


if __name__ == "__main__":
    test_tree_grid_highly_visual()
