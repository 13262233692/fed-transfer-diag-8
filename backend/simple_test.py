import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from app.core.data_simulator import FaultDiagnosticsDataSimulator, normalize_signal
from app.models.dann_module import DANNModule, DANNTrainer


class ExtremeDataSimulator(FaultDiagnosticsDataSimulator):
    def __init__(self, sample_rate: int = 1000, duration: float = 1.0):
        super().__init__(sample_rate, duration)
        self.working_conditions = {
            'factory_slow': {'speed': 0.1, 'load': 2.0, 'noise_factor': 0.5},
            'factory_medium': {'speed': 1.0, 'load': 1.0, 'noise_factor': 1.0},
            'factory_fast': {'speed': 10.0, 'load': 0.5, 'noise_factor': 2.0},
            'factory_extreme': {'speed': 20.0, 'load': 0.1, 'noise_factor': 5.0}
        }


def simple_test():
    print('Testing DANN module...')
    
    device = 'cpu'
    
    print('1. Generating data...')
    simulator = ExtremeDataSimulator()
    client_datasets = simulator.generate_federated_datasets(n_samples_per_class=20)
    
    normalized_datasets = {}
    for domain_id, (X, y) in client_datasets.items():
        X_normalized = normalize_signal(X)
        normalized_datasets[domain_id] = (X_normalized, y)
        print(f'   {domain_id}: {X_normalized.shape}')
    
    print('2. Creating DANN model...')
    dann_module = DANNModule(
        input_length=1000, 
        latent_dim=128, 
        num_classes=5, 
        num_domains=4,
        dropout_rate=0.5
    )
    
    print('3. Testing forward pass...')
    x_test = torch.randn(4, 1000)
    features, class_logits, domain_logits = dann_module(x_test, alpha=0.5)
    print(f'   Features shape: {features.shape}')
    print(f'   Class logits shape: {class_logits.shape}')
    print(f'   Domain logits shape: {domain_logits.shape}')
    
    print('4. Creating trainer...')
    trainer = DANNTrainer(
        model=dann_module,
        device=device,
        learning_rate=0.0005,
        lambda_class=1.0,
        lambda_domain=0.3,
        max_grad_norm=5.0,
        use_gradient_penalty=False
    )
    
    print('5. Running one epoch...')
    dataloader = trainer.prepare_domain_data(normalized_datasets, batch_size=8)
    
    trainer.model.train()
    for batch_X, batch_y, batch_domain in dataloader:
        print(f'   Batch X shape: {batch_X.shape}')
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_domain = batch_domain.to(device)
        
        trainer.optimizer.zero_grad()
        features, class_logits, domain_logits = trainer.model(batch_X, alpha=0.1)
        print(f'   Features after model: {features.shape}')
        
        class_loss = trainer.class_criterion(class_logits, batch_y)
        domain_loss = trainer.domain_criterion(domain_logits, batch_domain)
        
        loss = class_loss + domain_loss
        print(f'   Loss: {loss.item():.4f}')
        
        loss.backward()
        trainer._clip_gradients()
        trainer.optimizer.step()
        
        print('   Batch training successful!')
        break
    
    print('\nAll tests passed!')


if __name__ == '__main__':
    simple_test()
