import torch
import json
import numpy as np
from skelTransformer import JointTransformerModel  # Import your model class

# Model and input parameters (these should match training setup)
input_dim = 3  # Ensure this is consistent with training
d_model = 256
nhead = 8
num_encoder_layers = 4
dim_feedforward = 512
num_joints = 67  # Set this to match sequence_length (65)
output_dim = 3
sequence_length = 67  # Set this to 65, same as training

# Joint hierarchy for Mixamo skeleton
joint_hierarchy = {
    "Hips": None, "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Spine3": "Spine2", "Neck": "Spine3",
    "Head": "Neck", "LeftShoulder": "Spine3", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "LeftHandThumb1": "LeftHand", "LeftHandThumb2": "LeftHandThumb1", "LeftHandThumb3": "LeftHandThumb2",
    "LeftHandIndex1": "LeftHand", "LeftHandIndex2": "LeftHandIndex1", "LeftHandIndex3": "LeftHandIndex2",
    "LeftHandMiddle1": "LeftHand", "LeftHandMiddle2": "LeftHandMiddle1", "LeftHandMiddle3": "LeftHandMiddle2",
    "LeftHandRing1": "LeftHand", "LeftHandRing2": "LeftHandRing1", "LeftHandRing3": "LeftHandRing2",
    "LeftHandPinky1": "LeftHand", "LeftHandPinky2": "LeftHandPinky1", "LeftHandPinky3": "LeftHandPinky2",
    "RightShoulder": "Spine3", "RightArm": "RightShoulder", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "RightHandThumb1": "RightHand", "RightHandThumb2": "RightHandThumb1", "RightHandThumb3": "RightHandThumb2",
    "RightHandIndex1": "RightHand", "RightHandIndex2": "RightHandIndex1", "RightHandIndex3": "RightHandIndex2",
    "RightHandMiddle1": "RightHand", "RightHandMiddle2": "RightHandMiddle1", "RightHandMiddle3": "RightHandMiddle2",
    "RightHandRing1": "RightHand", "RightHandRing2": "RightHandRing1", "RightHandRing3": "RightHandRing2",
    "RightHandPinky1": "RightHand", "RightHandPinky2": "RightHandPinky1", "RightHandPinky3": "RightHandPinky2",
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg", "LeftToeBase": "LeftFoot",
    "LeftToe_End": "LeftToeBase", "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot", "RightToe_End": "RightToeBase", "LeftUpLegRoll": "LeftUpLeg", "LeftLegRoll": "LeftLeg",
    "LeftFootRoll": "LeftFoot", "RightUpLegRoll": "RightUpLeg", "RightLegRoll": "RightLeg", "RightFootRoll": "RightFoot",
    "Neck1": "Neck", "Neck2": "Neck1", "Neck3": "Neck2", "LeftEye": "Head", "RightEye": "Head", "Jaw": "Head"
}

def load_trained_model():
    model = JointTransformerModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, num_joints, output_dim)
    model.load_state_dict(torch.load("J:/dev/AI/SkeletonGeneration/TrainedModel/skeleton_model.pth", weights_only=True))
    model.eval()  # Set model to evaluation mode for inference
    return model

def prepare_mesh_data(mesh_data, sequence_length, input_dim):
    vertex_positions = mesh_data["geometry"]["vertex_positions"]
    bounds = mesh_data["geometry"]["bounds"]

    # Normalize vertex positions relative to bounding box
    vertex_features = []
    for pos in vertex_positions:
        normalized_pos = [
            (pos[0] - bounds['left']) / (bounds['right'] - bounds['left']),
            (pos[1] - bounds['down']) / (bounds['up'] - bounds['down']),
            (pos[2] - bounds['forward']) / (bounds['back'] - bounds['forward'])
        ]
        vertex_features.append(normalized_pos)
    
    # Pad or truncate to match `sequence_length`
    vertex_features = vertex_features[:sequence_length] + [[0] * input_dim] * max(0, sequence_length - len(vertex_features))
    vertex_features = np.array(vertex_features)

    return torch.tensor(vertex_features, dtype=torch.float32).unsqueeze(0)  # Shape (1, sequence_length, input_dim)

def predict_joint_positions(mesh_json_file, output_json_file="J:/dev/AI/SkeletonGeneration/PredictedSkeleton/predicted_joint_positions.json"):
    # Load the mesh data (new character's JSON data)
    with open(mesh_json_file, 'r') as f:
        mesh_data = json.load(f)
    
    # Prepare the mesh data for prediction
    input_tensor = prepare_mesh_data(mesh_data, sequence_length, input_dim)
    
    # Load the trained model
    model = load_trained_model()
    
    # Get predictions
    with torch.no_grad():
        predicted_positions = model(input_tensor).squeeze(0)  # Shape (num_joints, 3)

    # Calculate bounding box center and scale
    bounds = mesh_data["geometry"]["bounds"]
    bounding_center = [
        (bounds['right'] + bounds['left']) / 2,
        (bounds['up'] + bounds['down']) / 2,
        (bounds['forward'] + bounds['back']) / 2
    ]
    scale_y = bounds['up'] - bounds['down']
    scale_x = bounds['right'] - bounds['left']
    scale_z = bounds['back'] - bounds['forward']

    # Denormalize positions
    denormalized_positions = []
    for pos in predicted_positions.tolist():
        denormalized_pos = [
            pos[0] * scale_x + bounding_center[0],
            pos[1] * scale_y + bounding_center[1],
            pos[2] * scale_z + bounding_center[2]
        ]
        denormalized_positions.append(denormalized_pos)

    # Sanity checks
    joint_indices = {name: i for i, name in enumerate(joint_hierarchy)}
    toe_lowest_y = min(denormalized_positions[joint_indices["LeftToeBase"]][1], 
                       denormalized_positions[joint_indices["RightToeBase"]][1])
    hand_furthest_x = max(denormalized_positions[joint_indices["LeftHandMiddle3"]][0], 
                          denormalized_positions[joint_indices["RightHandMiddle3"]][0])
    head_position_y = denormalized_positions[joint_indices["Head"]][1]

    if toe_lowest_y > bounds['down']:
        print(f"Warning: Toe bases are not at the lowest Y. Expected {bounds['down']}, found {toe_lowest_y}")
    if hand_furthest_x < bounds['right'] and -hand_furthest_x > bounds['left']:
        print(f"Warning: Hands are not at the farthest X positions.")
    if head_position_y < (bounds['down'] + 0.7 * scale_y):
        print(f"Warning: Head is not at least 70% of the character's height. Expected >= {(bounds['down'] + 0.7 * scale_y)}, found {head_position_y}")

    # Save output JSON
    joints_data = []
    for i, joint_name in enumerate(joint_hierarchy):
        if i < len(denormalized_positions):
            joint_data = {
                "name": joint_name,
                "position": denormalized_positions[i],
                "parent": joint_hierarchy[joint_name]
            }
            joints_data.append(joint_data)

    with open(output_json_file, 'w') as f:
        json.dump(joints_data, f, indent=4)

    print(f"Predicted joint positions saved to {output_json_file}")

# Usage
predict_joint_positions("J:/dev/AI/SkeletonGeneration/MeshData/character_mesh_data.json")
