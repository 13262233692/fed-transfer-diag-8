import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import copy


class LocalAdapter(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        adapter_dim: int = 64,
        num_classes: int = 5,
        num_bottleneck_layers: int = 2,
        dropout_rate: float = 0.3
    ):
        super(LocalAdapter, self).__init__()
        self.latent_dim = latent_dim
        self.adapter_dim = adapter_dim
        self.num_classes = num_classes
        
        self.down_proj = nn.Linear(latent_dim, adapter_dim)
        self.bn_down = nn.BatchNorm1d(adapter_dim)
        
        bottleneck_layers = []
        in_dim = adapter_dim
        for i in range(num_bottleneck_layers):
            out_dim = max(adapter_dim // (2 ** i), 16)
            bottleneck_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_dim = out_dim
        
        self.bottleneck = nn.Sequential(*bottleneck_layers)
        self.bottleneck_out_dim = in_dim
        
        self.up_proj = nn.Linear(self.bottleneck_out_dim, latent_dim)
        self.bn_up = nn.BatchNorm1d(latent_dim)
        
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        self.gate = nn.Linear(latent_dim, latent_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        nn.init.uniform_(self.gate.weight, -0.01, 0.01)
        nn.init.constant_(self.gate.bias, -2.0)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = features
        
        adapted = F.relu(self.bn_down(self.down_proj(features)))
        adapted = self.bottleneck(adapted)
        adapted = F.relu(self.bn_up(self.up_proj(adapted)))
        
        gate_weight = torch.sigmoid(self.gate(residual))
        adapted_features = gate_weight * adapted + (1 - gate_weight) * residual
        
        logits = self.classifier(adapted_features)
        
        return adapted_features, logits


class AdaptedFaultDiagnosisModel(nn.Module):
    def __init__(
        self,
        base_model,
        adapter: Optional[LocalAdapter] = None,
        adapter_dim: int = 64,
        num_classes: int = 5
    ):
        super(AdaptedFaultDiagnosisModel, self).__init__()
        self.feature_extractor = base_model.feature_extractor
        self.base_classifier = base_model.classifier
        
        latent_dim = base_model.feature_extractor.latent_dim
        
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = LocalAdapter(
                latent_dim=latent_dim,
                adapter_dim=adapter_dim,
                num_classes=num_classes
            )
        
        self._adapter_active = True
    
    def set_adapter_active(self, active: bool):
        self._adapter_active = active
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        base_logits = self.base_classifier(features)
        
        if self._adapter_active and self.training:
            adapted_features, adapted_logits = self.adapter(features)
            return {
                'features': features,
                'base_logits': base_logits,
                'adapted_features': adapted_features,
                'adapted_logits': adapted_logits
            }
        elif self._adapter_active and not self.training:
            adapted_features, adapted_logits = self.adapter(features)
            return {
                'features': features,
                'base_logits': base_logits,
                'adapted_features': adapted_features,
                'adapted_logits': adapted_logits
            }
        else:
            return {
                'features': features,
                'base_logits': base_logits,
                'adapted_features': features,
                'adapted_logits': base_logits
            }
    
    def get_feature_extractor_params(self) -> dict:
        return self.feature_extractor.get_parameters_dict()
    
    def load_feature_extractor_params(self, params: dict):
        self.feature_extractor.load_parameters_dict(params)
    
    def get_adapter_params(self) -> dict:
        return {name: param.data.clone() for name, param in self.adapter.state_dict().items()}
    
    def load_adapter_params(self, params: dict):
        self.adapter.load_state_dict(params)
    
    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    
    def freeze_base_classifier(self):
        for param in self.base_classifier.parameters():
            param.requires_grad = False
    
    def unfreeze_base_classifier(self):
        for param in self.base_classifier.parameters():
            param.requires_grad = True
