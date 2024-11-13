import maya.cmds as cmds
import json

def get_mesh_data(mesh_name):
    """Extracts vertex positions and bounding box information for the given mesh."""
    # Get the bounding box for mesh scaling information
    bounding_box = cmds.exactWorldBoundingBox(mesh_name)
    bounds = {
        "up": bounding_box[4],
        "down": bounding_box[1],
        "left": bounding_box[0],
        "right": bounding_box[3],
        "forward": bounding_box[2],
        "back": bounding_box[5]
    }

    # Get positions of all vertices in the mesh
    vertices = cmds.ls(f"{mesh_name}.vtx[*]", flatten=True)
    vertex_positions = [cmds.pointPosition(vertex, world=True) for vertex in vertices]

    # Structure data similarly to the training format
    mesh_data = {
        "geometry": {
            "vertex_count": len(vertex_positions),
            "bounds": bounds,
            "vertex_positions": vertex_positions
        }
    }

    return mesh_data

def export_mesh_data(mesh_name, output_file="character_mesh_data.json"):
    """Exports the mesh data to a JSON file."""
    mesh_data = get_mesh_data(mesh_name)
    with open(output_file, 'w') as f:
        json.dump(mesh_data, f, indent=4)
    print(f"Mesh data exported to {output_file}")

# Usage example: Select the mesh and run this script
selected_mesh = cmds.ls(selection=True)[0]  # Assumes you selected the character mesh
export_mesh_data(selected_mesh, "J:/dev/AI/SkeletonGeneration/MeshData/character_mesh_data.json")
