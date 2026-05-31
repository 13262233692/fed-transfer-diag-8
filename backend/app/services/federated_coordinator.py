import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import copy
from datetime import datetime

from ..models.cnn_feature_extractor import FaultDiagnosisModel, CNNFeatureExtractor
from ..models.dann_module import DANNModule, DANNTrainer
from .client_trainer import FederatedClientManager
from ..core.data_simulator import normalize_signal


class FederatedAverager:
    @staticmethod
    def fedavg(
        client_params: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        if not client_params:
            return {}
        
        client_ids = list(client_params.keys())
        first_client_params = client_params[client_ids[0]]
        
        if client_weights is None:
            client_weights = {cid: 1.0 / len(client_ids) for cid in client_ids}
        
        total_weight = sum(client_weights.values())
        normalized_weights = {
            cid: w / total_weight for cid, w in client_weights.items()
        }
        
        global_params = {}
        for param_name in first_client_params.keys():
            weighted_sum = None
            for client_id in client_ids:
                param = client_params[client_id][param_name]
                weight = normalized_weights[client_id]
                
                if weighted_sum is None:
                    weighted_sum = param.clone() * weight
                else:
                    weighted_sum += param * weight
            
            global_params[param_name] = weighted_sum
        
        return global_params
    
    @staticmethod
    def compute_client_weights(
        client_data_sizes: Dict[str, int]
    ) -> Dict[str, float]:
        total_samples = sum(client_data_sizes.values())
        return {
            cid: size / total_samples for cid, size in client_data_sizes.items()
        }


class FederatedCoordinator:
    def __init__(
        self,
        global_model: FaultDiagnosisModel,
        dann_module: Optional[DANNModule] = None,
        device: str = 'cpu',
        adapter_dim: int = 64
    ):
        self.device = torch.device(device)
        self.global_model = global_model.to(self.device)
        self.dann_module = dann_module
        if dann_module is not None:
            self.dann_module = dann_module.to(self.device)
        
        self.client_manager = FederatedClientManager(device=device)
        self.adapter_dim = adapter_dim
        
        self.round_history = {
            'round': [],
            'avg_train_loss': [],
            'avg_train_acc': [],
            'global_model_accuracy': []
        }
        
        self.adapter_comparison: Dict[str, Dict] = {}
        
        self.current_round = 0
    
    def register_client(
        self,
        client_id: str,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        self.client_manager.create_client(
            client_id=client_id,
            model=self.global_model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            adapter_dim=self.adapter_dim
        )
        print(f'Client {client_id} registered successfully')
    
    def register_clients(
        self,
        client_ids: List[str],
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        for client_id in client_ids:
            self.register_client(client_id, learning_rate, batch_size)
    
    def train_clients_one_round(
        self,
        client_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        local_epochs: int = 5
    ) -> Dict[str, Dict]:
        print(f'\n=== Federated Training Round {self.current_round + 1} ===')
        
        histories = self.client_manager.train_all_clients(
            client_datasets, epochs=local_epochs
        )
        
        return histories
    
    def aggregate_parameters(
        self,
        client_data_sizes: Optional[Dict[str, int]] = None
    ) -> Dict[str, torch.Tensor]:
        client_params = self.client_manager.collect_feature_extractor_params()
        
        if client_data_sizes is not None:
            client_weights = FederatedAverager.compute_client_weights(client_data_sizes)
            global_params = FederatedAverager.fedavg(client_params, client_weights)
        else:
            global_params = FederatedAverager.fedavg(client_params)
        
        self.global_model.load_feature_extractor_params(global_params)
        
        return global_params
    
    def broadcast_global_model(self):
        global_params = self.global_model.get_feature_extractor_params()
        self.client_manager.broadcast_feature_extractor_params(global_params)
        print('Global model parameters broadcasted to all clients')
    
    def dann_alignment(
        self,
        domain_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        dann_epochs: int = 15,
        batch_size: int = 32,
        max_alpha: float = 0.3,
        lambda_domain: float = 0.3
    ) -> Dict[str, List[float]]:
        if self.dann_module is None:
            print('DANN module not initialized, skipping domain alignment')
            return {}
        
        print('\n=== DANN Domain Alignment ===')
        
        original_params = copy.deepcopy(self.global_model.get_feature_extractor_params())
        
        self.dann_module.load_feature_extractor_params(
            self.global_model.get_feature_extractor_params()
        )
        
        dann_trainer = DANNTrainer(
            model=self.dann_module,
            device=str(self.device),
            learning_rate=0.0005,
            lambda_class=1.0,
            lambda_domain=lambda_domain,
            max_grad_norm=5.0,
            label_smoothing=0.1,
            use_gradient_penalty=True,
            gp_lambda=10.0,
            temperature=1.0,
            early_stopping_patience=5
        )
        
        normalized_datasets = {}
        for domain_id, (X, y) in domain_datasets.items():
            X_normalized = normalize_signal(X)
            normalized_datasets[domain_id] = (X_normalized, y)
        
        history = dann_trainer.train(
            normalized_datasets,
            epochs=dann_epochs,
            batch_size=batch_size
        )
        
        aligned_params = dann_trainer.get_aligned_feature_extractor().get_parameters_dict()
        
        if history and 'class_accuracy' in history and len(history['class_accuracy']) > 0:
            final_class_acc = history['class_accuracy'][-1]
            if final_class_acc < 0.3:
                print(f'Warning: DANN alignment resulted in low accuracy ({final_class_acc:.4f}). '
                      f'Restoring original parameters...')
                self.global_model.load_feature_extractor_params(original_params)
                return {'warning': 'DANN alignment skipped due to instability', 'original_acc': 'preserved'}
        
        self.global_model.load_feature_extractor_params(aligned_params)
        
        return history
    
    def personalize_with_adapters(
        self,
        client_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        adapter_epochs: int = 10,
        adapter_lr: float = 0.001
    ) -> Dict[str, Dict]:
        print('\n=== Personalization Phase: Training Local Adapters ===')
        print(f'Adapter dimension: {self.adapter_dim}')
        print(f'Adapter training epochs: {adapter_epochs}')
        print('Adapter parameters are NOT shared in federated aggregation')
        
        adapter_histories = self.client_manager.train_all_adapters(
            client_datasets,
            adapter_epochs=adapter_epochs,
            adapter_lr=adapter_lr
        )
        
        self.adapter_comparison = self.client_manager.get_adapter_comparison()
        
        print('\n=== Adapter Personalization Results ===')
        for client_id, comp in self.adapter_comparison.items():
            if comp['has_adapter'] and comp['before_adapter_accuracy'] is not None:
                print(f'{client_id}: {comp["before_adapter_accuracy"]:.4f} -> {comp["after_adapter_accuracy"]:.4f} '
                      f'(improvement: +{comp["improvement"]:.4f})')
        
        return adapter_histories
    
    def evaluate_global_model(
        self,
        test_dataset: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        X_test, y_test = test_dataset
        X_test = normalize_signal(X_test)
        
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        self.global_model.eval()
        with torch.no_grad():
            _, logits = self.global_model(X_tensor)
            loss = torch.nn.CrossEntropyLoss()(logits, y_tensor)
            _, predicted = torch.max(logits.data, 1)
            accuracy = (predicted == y_tensor).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def get_adapter_comparison(self) -> Dict[str, Dict]:
        return self.adapter_comparison
    
    def run_federated_training(
        self,
        client_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        test_dataset: Tuple[np.ndarray, np.ndarray],
        num_rounds: int = 10,
        local_epochs: int = 5,
        use_dann: bool = True,
        dann_interval: int = 3,
        dann_epochs: int = 10,
        use_adapter: bool = True,
        adapter_epochs: int = 10,
        adapter_lr: float = 0.001
    ) -> Dict:
        client_ids = list(client_datasets.keys())
        self.register_clients(client_ids)
        
        client_data_sizes = {
            cid: len(dataset[0]) for cid, dataset in client_datasets.items()
        }
        
        full_history = {
            'rounds': [],
            'client_histories': {},
            'global_evaluation': [],
            'dann_history': [],
            'adapter_history': {},
            'adapter_comparison': {}
        }
        
        for round_idx in range(num_rounds):
            self.current_round = round_idx
            
            client_histories = self.train_clients_one_round(
                client_datasets, local_epochs=local_epochs
            )
            full_history['client_histories'][round_idx] = client_histories
            
            self.aggregate_parameters(client_data_sizes)
            self.broadcast_global_model()
            
            if use_dann and (round_idx + 1) % dann_interval == 0:
                dann_history = self.dann_alignment(
                    client_datasets, dann_epochs=dann_epochs
                )
                full_history['dann_history'].append({
                    'round': round_idx,
                    'history': dann_history
                })
                self.broadcast_global_model()
            
            global_metrics = self.evaluate_global_model(test_dataset)
            full_history['global_evaluation'].append(global_metrics)
            
            print(f'Round {round_idx + 1} Complete - Global Model Accuracy: {global_metrics["accuracy"]:.4f}')
            full_history['rounds'].append({
                'round': round_idx + 1,
                'global_metrics': global_metrics
            })
        
        if use_adapter:
            adapter_histories = self.personalize_with_adapters(
                client_datasets,
                adapter_epochs=adapter_epochs,
                adapter_lr=adapter_lr
            )
            full_history['adapter_history'] = adapter_histories
            full_history['adapter_comparison'] = self.get_adapter_comparison()
        
        return full_history
    
    def get_global_model_state(self) -> Dict:
        return {
            'feature_extractor': self.global_model.get_feature_extractor_params(),
            'classifier': self.global_model.classifier.get_parameters_dict(),
            'current_round': self.current_round
        }
    
    def save_checkpoint(self, filepath: str):
        checkpoint = {
            'global_model_state': self.global_model.state_dict(),
            'current_round': self.current_round,
            'round_history': self.round_history,
            'adapter_comparison': self.adapter_comparison,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved to {filepath}')
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath)
        self.global_model.load_state_dict(checkpoint['global_model_state'])
        self.current_round = checkpoint['current_round']
        self.round_history = checkpoint['round_history']
        if 'adapter_comparison' in checkpoint:
            self.adapter_comparison = checkpoint['adapter_comparison']
        print(f'Checkpoint loaded from {filepath}')
