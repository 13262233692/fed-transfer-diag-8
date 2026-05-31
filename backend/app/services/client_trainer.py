import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

from ..models.cnn_feature_extractor import FaultDiagnosisModel
from ..models.local_adapter import LocalAdapter, AdaptedFaultDiagnosisModel
from ..core.data_simulator import normalize_signal


class ClientTrainer:
    def __init__(
        self,
        client_id: str,
        model: FaultDiagnosisModel,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        batch_size: int = 32,
        adapter_dim: int = 64
    ):
        self.client_id = client_id
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        
        self.adapted_model: Optional[AdaptedFaultDiagnosisModel] = None
        self.adapter_dim = adapter_dim
        
        self.before_adapter_accuracy: Optional[float] = None
        self.after_adapter_accuracy: Optional[float] = None
        self.adapter_history: Optional[Dict] = None
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        X = normalize_signal(X)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            _, logits = self.model(batch_X)
            loss = self.criterion(logits, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
            
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                _, logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict:
        train_loader, val_loader = self.prepare_data(X, y)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if verbose:
                print(f'[{self.client_id}] Epoch {epoch+1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return history
    
    def get_feature_extractor_params(self) -> Dict[str, torch.Tensor]:
        return self.model.get_feature_extractor_params()
    
    def load_feature_extractor_params(self, params: Dict[str, torch.Tensor]):
        self.model.load_feature_extractor_params(params)
        if self.adapted_model is not None:
            self.adapted_model.load_feature_extractor_params(params)
    
    def get_classifier_params(self) -> Dict[str, torch.Tensor]:
        return self.model.classifier.get_parameters_dict()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        X_test = normalize_signal(X_test)
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            _, logits = self.model(X_tensor)
            loss = self.criterion(logits, y_tensor)
            _, predicted = torch.max(logits.data, 1)
            accuracy = (predicted == y_tensor).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def evaluate_with_adapter(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        if self.adapted_model is None:
            return self.evaluate(X_test, y_test)
        
        X_test = normalize_signal(X_test)
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        self.adapted_model.eval()
        with torch.no_grad():
            base_logits = self.adapted_model.feature_extractor(X_tensor)
            base_logits = self.adapted_model.base_classifier(base_logits)
            _, base_predicted = torch.max(base_logits.data, 1)
            base_accuracy = (base_predicted == y_tensor).float().mean().item()
        
        if not self.adapted_model._adapter_active:
            return {
                'base_loss': 0.0,
                'base_accuracy': base_accuracy,
                'adapted_loss': 0.0,
                'adapted_accuracy': base_accuracy,
                'improvement': 0.0
            }
        
        self.adapted_model.set_adapter_active(True)
        with torch.no_grad():
            outputs = self.adapted_model(X_tensor)
            adapted_logits = outputs['adapted_logits']
            adapted_loss = self.criterion(adapted_logits, y_tensor)
            _, adapted_predicted = torch.max(adapted_logits.data, 1)
            adapted_accuracy = (adapted_predicted == y_tensor).float().mean().item()
        
        self.adapted_model.set_adapter_active(False)
        
        return {
            'base_loss': 0.0,
            'base_accuracy': base_accuracy,
            'adapted_loss': adapted_loss.item(),
            'adapted_accuracy': adapted_accuracy,
            'improvement': adapted_accuracy - base_accuracy
        }
    
    def init_adapter(self):
        self.adapted_model = AdaptedFaultDiagnosisModel(
            base_model=copy.deepcopy(self.model),
            adapter_dim=self.adapter_dim,
            num_classes=self.model.classifier.num_classes
        )
        self.adapted_model.to(self.device)
        self.adapted_model.set_adapter_active(True)
    
    def train_adapter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        learning_rate: float = 0.001,
        verbose: bool = True
    ) -> Dict:
        if self.adapted_model is None:
            self.init_adapter()
        
        X = normalize_signal(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.adapted_model.freeze_feature_extractor()
        self.adapted_model.freeze_base_classifier()
        
        adapter_params = list(self.adapted_model.adapter.parameters())
        optimizer = optim.Adam(adapter_params, lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )
        
        criterion = nn.CrossEntropyLoss()
        kl_div = nn.KLDivLoss(reduction='batchmean')
        consistency_weight = 0.3
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'base_acc': [],
            'adapted_acc': []
        }
        
        self.adapted_model.set_adapter_active(True)
        
        if verbose:
            print(f'\n[{self.client_id}] Adapter Training - Freezing feature extractor & base classifier')
            print(f'[{self.client_id}] Only adapter parameters will be updated ({sum(p.numel() for p in adapter_params)} params)')
        
        best_adapted_acc = 0.0
        best_state_dict = None
        patience_counter = 0
        patience_limit = 5
        
        for epoch in range(epochs):
            self.adapted_model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.adapted_model(batch_X)
                adapted_logits = outputs['adapted_logits']
                base_logits = outputs['base_logits'].detach()
                
                cls_loss = criterion(adapted_logits, batch_y)
                
                adapted_log_probs = F.log_softmax(adapted_logits / 2.0, dim=1)
                base_probs = F.softmax(base_logits / 2.0, dim=1)
                consistency_loss = kl_div(adapted_log_probs, base_probs)
                
                loss = cls_loss + consistency_weight * consistency_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=5.0)
                optimizer.step()
                
                total_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(adapted_logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            train_loss = total_loss / total
            train_acc = correct / total
            
            self.adapted_model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0
            base_correct = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.adapted_model(batch_X)
                    
                    adapted_logits = outputs['adapted_logits']
                    a_loss = criterion(adapted_logits, batch_y)
                    val_loss_total += a_loss.item() * batch_X.size(0)
                    
                    _, adapted_pred = torch.max(adapted_logits.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (adapted_pred == batch_y).sum().item()
                    
                    base_logits = outputs['base_logits']
                    _, base_pred = torch.max(base_logits.data, 1)
                    base_correct += (base_pred == batch_y).sum().item()
            
            val_loss = val_loss_total / val_total
            val_acc = val_correct / val_total
            base_acc = base_correct / val_total
            
            scheduler.step(val_acc)
            
            if val_acc > best_adapted_acc:
                best_adapted_acc = val_acc
                best_state_dict = {k: v.clone() for k, v in self.adapted_model.adapter.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['base_acc'].append(base_acc)
            history['adapted_acc'].append(val_acc)
            
            if verbose:
                print(f'[{self.client_id}] Adapter Epoch {epoch+1}/{epochs} - '
                      f'Loss: {train_loss:.4f} - '
                      f'Base Acc: {base_acc:.4f} -> Adapted Acc: {val_acc:.4f} '
                      f'(+{val_acc - base_acc:+.4f})')
            
            if patience_counter >= patience_limit:
                if verbose:
                    print(f'[{self.client_id}] Early stopping at epoch {epoch+1}')
                break
        
        if best_state_dict is not None:
            self.adapted_model.adapter.load_state_dict(best_state_dict)
            if verbose:
                print(f'[{self.client_id}] Restored best adapter state (val_acc={best_adapted_acc:.4f})')
        
        self.adapted_model.unfreeze_feature_extractor()
        self.adapted_model.unfreeze_base_classifier()
        
        self.adapter_history = history
        
        eval_result = self.evaluate_with_adapter(X, y)
        self.before_adapter_accuracy = eval_result['base_accuracy']
        self.after_adapter_accuracy = eval_result['adapted_accuracy']
        
        if self.after_adapter_accuracy < self.before_adapter_accuracy:
            if verbose:
                print(f'[{self.client_id}] Adapter degrades performance, disabling adapter (falling back to base model)')
            self.adapted_model.set_adapter_active(False)
            self.after_adapter_accuracy = self.before_adapter_accuracy
        
        if verbose:
            print(f'[{self.client_id}] Adapter training complete: '
                  f'{self.before_adapter_accuracy:.4f} -> {self.after_adapter_accuracy:.4f} '
                  f'(improvement: {self.after_adapter_accuracy - self.before_adapter_accuracy:+.4f})')
        
        return history
    
    def get_adapter_params(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.adapted_model is None:
            return None
        return self.adapted_model.get_adapter_params()
    
    def load_adapter_params(self, params: Dict[str, torch.Tensor]):
        if self.adapted_model is not None:
            self.adapted_model.load_adapter_params(params)
    
    def get_model_state(self) -> Dict:
        state = {
            'client_id': self.client_id,
            'feature_extractor': self.get_feature_extractor_params(),
            'classifier': self.get_classifier_params(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'before_adapter_accuracy': self.before_adapter_accuracy,
            'after_adapter_accuracy': self.after_adapter_accuracy
        }
        if self.adapted_model is not None:
            state['adapter_params'] = self.get_adapter_params()
        return state


class FederatedClientManager:
    def __init__(self, device: str = 'cpu'):
        self.clients: Dict[str, ClientTrainer] = {}
        self.device = device
    
    def create_client(
        self,
        client_id: str,
        model: FaultDiagnosisModel,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        adapter_dim: int = 64
    ) -> ClientTrainer:
        client = ClientTrainer(
            client_id=client_id,
            model=copy.deepcopy(model),
            device=self.device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            adapter_dim=adapter_dim
        )
        self.clients[client_id] = client
        return client
    
    def get_client(self, client_id: str) -> Optional[ClientTrainer]:
        return self.clients.get(client_id)
    
    def remove_client(self, client_id: str):
        if client_id in self.clients:
            del self.clients[client_id]
    
    def train_all_clients(
        self,
        client_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        epochs: int = 10
    ) -> Dict[str, Dict]:
        all_histories = {}
        
        for client_id, (X, y) in client_datasets.items():
            if client_id in self.clients:
                print(f'\nTraining client {client_id}...')
                history = self.clients[client_id].train(X, y, epochs=epochs)
                all_histories[client_id] = history
        
        return all_histories
    
    def train_all_adapters(
        self,
        client_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        adapter_epochs: int = 10,
        adapter_lr: float = 0.001
    ) -> Dict[str, Dict]:
        all_histories = {}
        
        print('\n=== Adapter Personalization Phase ===')
        
        for client_id, (X, y) in client_datasets.items():
            if client_id in self.clients:
                history = self.clients[client_id].train_adapter(
                    X, y, epochs=adapter_epochs, learning_rate=adapter_lr
                )
                all_histories[client_id] = history
        
        return all_histories
    
    def collect_feature_extractor_params(self) -> Dict[str, Dict]:
        params = {}
        for client_id, client in self.clients.items():
            params[client_id] = client.get_feature_extractor_params()
        return params
    
    def broadcast_feature_extractor_params(self, global_params: Dict[str, torch.Tensor]):
        for client in self.clients.values():
            client.load_feature_extractor_params(global_params)
    
    def get_adapter_comparison(self) -> Dict[str, Dict]:
        comparison = {}
        for client_id, client in self.clients.items():
            comparison[client_id] = {
                'before_adapter_accuracy': client.before_adapter_accuracy,
                'after_adapter_accuracy': client.after_adapter_accuracy,
                'improvement': (client.after_adapter_accuracy - client.before_adapter_accuracy)
                    if client.before_adapter_accuracy is not None and client.after_adapter_accuracy is not None
                    else None,
                'has_adapter': client.adapted_model is not None
            }
        return comparison
