import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import math
from glob import glob
import os

# Data loading and preparation functions
def load_and_prepare_data(json_path, sequence_length, input_dim):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    joint_features = []
    
    def process_joint(joint_data):
        avg_vertex_position = np.array(joint_data.get('avg_vertex_position', [0, 0, 0]))
        flattened_vectors = np.array(joint_data.get('flattened_vertex_vectors', []))
        joint_feature = np.concatenate((avg_vertex_position, flattened_vectors))
        return joint_feature

    def recursive_process(joint_data):
        joint_features.append(process_joint(joint_data))
        for child in joint_data.get("children", []):
            recursive_process(child)
    
    recursive_process(data['joint_hierarchy'])
    joint_features = np.pad(joint_features, ((0, sequence_length - len(joint_features)), (0, 0)), 'constant')
    joint_features = joint_features[:sequence_length]
    
    return torch.tensor(joint_features, dtype=torch.float32)

def load_all_data(json_folder, sequence_length, input_dim):
    all_tensors = []
    for json_file in glob(os.path.join(json_folder, '*.json')):
        tensor = load_and_prepare_data(json_file, sequence_length, input_dim)
        all_tensors.append(tensor)
    return torch.stack(all_tensors, dim=0)

# Transformer model definition
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.encoding[:x.size(0), :]

class JointTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, num_joints, output_dim):
        super(JointTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Make sure input_dim here matches the feature size
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True),  # batch_first=True
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, output_dim * num_joints)
    
    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        transformer_output = self.transformer_encoder(src)
        output = self.fc(transformer_output[:, -1, :])  # Use batch_first indexing
        return output.view(-1, num_joints, 3)

# Hyperparameters
input_dim = 3
d_model = 256
nhead = 8
num_encoder_layers = 4
dim_feedforward = 512
num_joints = 67
output_dim = 3

# Initialize model, loss, and optimizer
model = JointTransformerModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, num_joints, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load data and train the model
json_folder = '../JSON_DATA'
sequence_length = 67
batch_inputs = load_all_data(json_folder, sequence_length, input_dim)

num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(batch_inputs)
    target_positions = torch.zeros(predictions.shape)  # Placeholder for actual targets
    loss = criterion(predictions, target_positions)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "J:/dev/AI/SkeletonGeneration/TrainedModel/skeleton_model.pth")