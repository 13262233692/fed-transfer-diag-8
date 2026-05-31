import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
import math

from .cnn_feature_extractor import CNNFeatureExtractor, Classifier


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def gradient_reversal(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes: int, smoothing: float = 0.1, dim: int = -1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim: int = 128, num_domains: int = 4, dropout_rate: float = 0.5):
        super(DomainDiscriminator, self).__init__()
        self.latent_dim = latent_dim
        self.num_domains = num_domains
        
        self.fc1 = nn.Linear(latent_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, num_domains)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        x = gradient_reversal(x, alpha)
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class DANNModule(nn.Module):
    def __init__(
        self,
        input_length: int = 1000,
        latent_dim: int = 128,
        num_classes: int = 5,
        num_domains: int = 4,
        dropout_rate: float = 0.5
    ):
        super(DANNModule, self).__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_domains = num_domains
        
        self.feature_extractor = CNNFeatureExtractor(input_length, latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(latent_dim, num_domains, dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
        return_domain: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features = self.feature_extractor(x)
        class_logits = self.classifier(features)
        
        if return_domain:
            domain_logits = self.domain_discriminator(features, alpha)
            return features, class_logits, domain_logits
        else:
            return features, class_logits, None
    
    def get_feature_extractor_params(self) -> Dict[str, torch.Tensor]:
        return self.feature_extractor.get_parameters_dict()
    
    def load_feature_extractor_params(self, params: Dict[str, torch.Tensor]):
        self.feature_extractor.load_parameters_dict(params)


class DANNTrainer:
    def __init__(
        self,
        model: DANNModule,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        lambda_class: float = 1.0,
        lambda_domain: float = 0.5,
        max_grad_norm: float = 5.0,
        label_smoothing: float = 0.1,
        use_gradient_penalty: bool = True,
        gp_lambda: float = 10.0,
        temperature: float = 1.0,
        early_stopping_patience: int = 5
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.lambda_class = lambda_class
        self.lambda_domain = lambda_domain
        self.max_grad_norm = max_grad_norm
        self.use_gradient_penalty = use_gradient_penalty
        self.gp_lambda = gp_lambda
        self.temperature = temperature
        self.early_stopping_patience = early_stopping_patience
        
        self.class_criterion = LabelSmoothingLoss(
            classes=model.num_classes, 
            smoothing=label_smoothing
        )
        self.domain_criterion = LabelSmoothingLoss(
            classes=model.num_domains, 
            smoothing=label_smoothing
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            min_lr=1e-6
        )
        
        self.history = {
            'total_loss': [],
            'class_loss': [],
            'domain_loss': [],
            'class_accuracy': [],
            'domain_accuracy': [],
            'learning_rate': []
        }
        
        self.best_state = None
        self.best_class_loss = float('inf')
        self.patience_counter = 0
        self.consecutive_nan_count = 0
        self.max_consecutive_nan = 3
    
    def _compute_gradient_penalty(
        self, 
        batch_X: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_gradient_penalty:
            return torch.tensor(0.0, device=self.device)
        
        alpha = torch.rand(batch_X.size(0), 1, device=self.device)
        
        permuted_idx = torch.randperm(batch_X.size(0))
        interpolates = alpha * batch_X + (1 - alpha) * batch_X[permuted_idx]
        interpolates = interpolates.requires_grad_(True)
        
        _, _, domain_logits = self.model(interpolates, alpha=1.0)
        
        gradients = torch.autograd.grad(
            outputs=domain_logits,
            inputs=interpolates,
            grad_outputs=torch.ones_like(domain_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty * self.gp_lambda
    
    def compute_alpha(
        self, 
        current_epoch: int, 
        max_epochs: int,
        warmup_epochs: int = 3,
        max_alpha: float = 0.5
    ) -> float:
        if current_epoch < warmup_epochs:
            p = float(current_epoch) / warmup_epochs
            alpha = max_alpha * 0.1 * (1 - p) + max_alpha * 0.1 * p
        else:
            p = float(current_epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
            alpha = max_alpha * (2. / (1. + np.exp(-5 * p)) - 1)
        
        return max(0.001, min(alpha, max_alpha))
    
    def _clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=self.max_grad_norm
        )
    
    def _check_nan(self, loss: torch.Tensor, class_loss: torch.Tensor, domain_loss: torch.Tensor) -> bool:
        losses = [loss, class_loss, domain_loss]
        return any(torch.isnan(l).any() or torch.isinf(l).any() for l in losses)
    
    def _restore_best_state(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state['model_state'])
            self.optimizer.load_state_dict(self.best_state['optimizer_state'])
            print('Restored best model state due to instability')
    
    def prepare_domain_data(
        self,
        domain_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 32
    ) -> DataLoader:
        all_X = []
        all_y = []
        all_domain_labels = []
        
        for domain_idx, (domain_id, (X, y)) in enumerate(domain_datasets.items()):
            all_X.append(X)
            all_y.append(y)
            domain_labels = np.full(len(X), domain_idx)
            all_domain_labels.append(domain_labels)
        
        all_X = np.concatenate(all_X, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        all_domain_labels = np.concatenate(all_domain_labels, axis=0)
        
        indices = np.random.permutation(len(all_X))
        all_X = all_X[indices]
        all_y = all_y[indices]
        all_domain_labels = all_domain_labels[indices]
        
        X_tensor = torch.tensor(all_X, dtype=torch.float32)
        y_tensor = torch.tensor(all_y, dtype=torch.long)
        domain_tensor = torch.tensor(all_domain_labels, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor, domain_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        return dataloader
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        max_epochs: int
    ) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        class_correct = 0
        domain_correct = 0
        total = 0
        valid_batches = 0
        
        alpha = self.compute_alpha(epoch, max_epochs)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        for batch_idx, (batch_X, batch_y, batch_domain) in enumerate(dataloader):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_domain = batch_domain.to(self.device)
            
            if batch_X.size(0) < 2:
                continue
            
            self.optimizer.zero_grad()
            
            features, class_logits, domain_logits = self.model(batch_X, alpha=alpha)
            
            class_logits = class_logits / self.temperature
            domain_logits = domain_logits / self.temperature
            
            class_loss = self.class_criterion(class_logits, batch_y)
            domain_loss = self.domain_criterion(domain_logits, batch_domain)
            
            gp_loss = self._compute_gradient_penalty(batch_X)
            
            loss = (self.lambda_class * class_loss + 
                    self.lambda_domain * domain_loss + 
                    gp_loss)
            
            if self._check_nan(loss, class_loss, domain_loss):
                self.consecutive_nan_count += 1
                print(f'Warning: NaN detected in batch {batch_idx}, skipping...')
                if self.consecutive_nan_count >= self.max_consecutive_nan:
                    print('Too many consecutive NaNs, restoring best state...')
                    self._restore_best_state()
                    self.consecutive_nan_count = 0
                continue
            
            self.consecutive_nan_count = 0
            
            loss.backward()
            self._clip_gradients()
            
            self.optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
            total_class_loss += class_loss.item() * batch_X.size(0)
            total_domain_loss += domain_loss.item() * batch_X.size(0)
            
            _, class_pred = torch.max(class_logits.data, 1)
            _, domain_pred = torch.max(domain_logits.data, 1)
            
            total += batch_y.size(0)
            class_correct += (class_pred == batch_y).sum().item()
            domain_correct += (domain_pred == batch_domain).sum().item()
            valid_batches += 1
        
        if total == 0:
            print('Warning: No valid batches in this epoch')
            return {
                'total_loss': float('inf'),
                'class_loss': float('inf'),
                'domain_loss': float('inf'),
                'class_accuracy': 0.0,
                'domain_accuracy': 0.0,
                'learning_rate': current_lr
            }
        
        avg_loss = total_loss / total
        avg_class_loss = total_class_loss / total
        avg_domain_loss = total_domain_loss / total
        class_acc = class_correct / total
        domain_acc = domain_correct / total
        
        if avg_class_loss < self.best_class_loss:
            self.best_class_loss = avg_class_loss
            self.best_state = {
                'model_state': copy.deepcopy(self.model.state_dict()),
                'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
                'epoch': epoch,
                'class_loss': avg_class_loss
            }
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        self.scheduler.step(avg_class_loss)
        
        return {
            'total_loss': avg_loss,
            'class_loss': avg_class_loss,
            'domain_loss': avg_domain_loss,
            'class_accuracy': class_acc,
            'domain_accuracy': domain_acc,
            'learning_rate': current_lr
        }
    
    def train(
        self,
        domain_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        epochs: int = 20,
        batch_size: int = 32,
        verbose: bool = True,
        validate_every: int = 1
    ) -> Dict[str, List[float]]:
        dataloader = self.prepare_domain_data(domain_datasets, batch_size)
        
        self.best_state = None
        self.best_class_loss = float('inf')
        self.patience_counter = 0
        self.consecutive_nan_count = 0
        
        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader, epoch, epochs)
            
            for key, value in metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            
            if verbose:
                lr_str = f'LR: {metrics["learning_rate"]:.6f}'
                print(f'DANN Epoch {epoch+1}/{epochs} - '
                      f'Total Loss: {metrics["total_loss"]:.4f} - '
                      f'Class Loss: {metrics["class_loss"]:.4f} - '
                      f'Domain Loss: {metrics["domain_loss"]:.4f} - '
                      f'Class Acc: {metrics["class_accuracy"]:.4f} - '
                      f'Domain Acc: {metrics["domain_accuracy"]:.4f} - '
                      f'{lr_str}')
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state['model_state'])
            print(f'Restored best model from epoch {self.best_state["epoch"] + 1}')
        
        return self.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        domain_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        self.model.eval()
        
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            _, class_logits, domain_logits = self.model(X_tensor, alpha=0.0)
            
            class_loss = self.class_criterion(class_logits, y_tensor)
            _, class_pred = torch.max(class_logits.data, 1)
            class_acc = (class_pred == y_tensor).float().mean().item()
            
            results = {
                'class_loss': class_loss.item(),
                'class_accuracy': class_acc
            }
            
            if domain_labels is not None:
                domain_tensor = torch.tensor(domain_labels, dtype=torch.long).to(self.device)
                domain_loss = self.domain_criterion(domain_logits, domain_tensor)
                _, domain_pred = torch.max(domain_logits.data, 1)
                domain_acc = (domain_pred == domain_tensor).float().mean().item()
                
                results['domain_loss'] = domain_loss.item()
                results['domain_accuracy'] = domain_acc
        
        return results
    
    def get_aligned_feature_extractor(self) -> CNNFeatureExtractor:
        return copy.deepcopy(self.model.feature_extractor)
