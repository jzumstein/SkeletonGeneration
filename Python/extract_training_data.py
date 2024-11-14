import maya.cmds as cmds
import json
import os
import random

#TODO: Improve accuracy to ensure correct joint skeletal placement.

def get_geometry_data(geometry):
    """
    Collects the total vertex count and bounding box information for the geometry.
    """
    vertices = cmds.ls(f"{geometry}.vtx[*]", flatten=True)
    vertex_count = len(vertices)
    
    # Initialize bounds with extreme values
    bounds = {
        "up": -float("inf"),
        "down": float("inf"),
        "left": float("inf"),
        "right": -float("inf"),
        "forward": -float("inf"),
        "back": float("inf")
    }
    
    # Calculate bounds by iterating over each vertex's world position
    for vertex in vertices:
        pos = cmds.pointPosition(vertex, world=True)
        x, y, z = pos
        
        # Update bounds based on current vertex position
        bounds["up"] = max(bounds["up"], y)
        bounds["down"] = min(bounds["down"], y)
        bounds["left"] = min(bounds["left"], x)
        bounds["right"] = max(bounds["right"], x)
        bounds["forward"] = max(bounds["forward"], z)
        bounds["back"] = min(bounds["back"], z)

    return {
        "vertex_count": vertex_count,
        "bounds": bounds
    }

def get_joint_data(joint, root_pos):
    """
    Retrieves the joint's world and relative position data.
    """
    joint_data = {}
    joint_data["name"] = joint
    joint_pos = cmds.xform(joint, query=True, translation=True, worldSpace=True)
    joint_data["relative_position"] = [joint_pos[i] - root_pos[i] for i in range(3)]
    joint_data["world_position"] = joint_pos
    joint_data["children"] = []

    return joint_data

def export_joint_hierarchy(root_joint, geometry, output_file):
    """
    Export joint hierarchy with only joint positions relative to the root joint and bounding box data.
    """
    geometry_data = get_geometry_data(geometry)

    # Get root joint position for relative positioning
    root_pos = cmds.xform(root_joint, query=True, translation=True, worldSpace=True)

    # Initialize the data structure for the hierarchy
    def process_joint_hierarchy(joint):
        joint_data = get_joint_data(joint, root_pos)
        
        # Process children recursively
        for child in cmds.listRelatives(joint, children=True, type="joint") or []:
            joint_data["children"].append(process_joint_hierarchy(child))
        
        return joint_data
    
    hierarchy_data = process_joint_hierarchy(root_joint)
    
    output_data = {
        "geometry": geometry_data,
        "joint_hierarchy": hierarchy_data
    }
    
    # Write data to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Joint data exported to {output_file}")

# Specify root joint, geometry, and output path
root_joint = "Hips"  # Replace with the actual root joint of your rig

selection = cmds.ls(sl=True, l=True)
if selection:
    geometry = selection[0]  # Replace with the actual mesh name bound to skinCluster1
    name = "joint_position_data_" + str(random.randint(0, 100000)) + ".json"
    output_file = os.path.join("J:/dev/AI/SkeletonGeneration/JSON_DATA", name)
    export_joint_hierarchy(root_joint, geometry, output_file)
else: 
    print("No object selected")
