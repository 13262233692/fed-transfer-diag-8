import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_length: int = 1000, latent_dim: int = 128):
        super(CNNFeatureExtractor, self).__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        conv_output_length = self._calculate_conv_output_length(input_length)
        
        self.fc1 = nn.Linear(256 * conv_output_length, latent_dim)
        self.fc_bn = nn.BatchNorm1d(latent_dim)
    
    def _calculate_conv_output_length(self, input_length: int) -> int:
        def conv_output_len(length, kernel_size, stride, padding):
            return (length + 2 * padding - kernel_size) // stride + 1
        
        length = input_length
        length = conv_output_len(length, 15, 2, 7)
        length = length // 2
        length = conv_output_len(length, 11, 2, 5)
        length = length // 2
        length = conv_output_len(length, 7, 2, 3)
        length = length // 2
        length = conv_output_len(length, 5, 2, 2)
        return length
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_bn(self.fc1(x)))
        
        return x
    
    def get_parameters_dict(self) -> dict:
        return {name: param.data.clone() for name, param in self.state_dict().items()}
    
    def load_parameters_dict(self, params: dict):
        self.load_state_dict(params)


class Classifier(nn.Module):
    def __init__(self, latent_dim: int = 128, num_classes: int = 5):
        super(Classifier, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(latent_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
    
    def get_parameters_dict(self) -> dict:
        return {name: param.data.clone() for name, param in self.state_dict().items()}
    
    def load_parameters_dict(self, params: dict):
        self.load_state_dict(params)


class FaultDiagnosisModel(nn.Module):
    def __init__(self, input_length: int = 1000, latent_dim: int = 128, num_classes: int = 5):
        super(FaultDiagnosisModel, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(input_length, latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return features, logits
    
    def get_feature_extractor_params(self) -> dict:
        return self.feature_extractor.get_parameters_dict()
    
    def load_feature_extractor_params(self, params: dict):
        self.feature_extractor.load_parameters_dict(params)
