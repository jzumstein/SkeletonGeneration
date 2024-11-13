import maya.cmds as cmds
import json

def apply_joint_positions(joint_positions_file):
    with open(joint_positions_file, 'r') as f:
        joint_data = json.load(f)
    
    # Create each joint at its specified position in Maya
    for joint in joint_data:
        cmds.joint(name=joint["name"], position=joint["position"])

# Example usage:
# apply_joint_positions("predicted_joint_positions.json")
