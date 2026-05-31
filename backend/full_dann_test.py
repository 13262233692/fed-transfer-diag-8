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


def full_dann_test():
    print('=' * 70)
    print('Full DANN Stability Test with Gradient Penalty')
    print('=' * 70)
    
    device = 'cpu'
    
    print('\n1. Generating extreme domain shift data...')
    simulator = ExtremeDataSimulator()
    client_datasets = simulator.generate_federated_datasets(n_samples_per_class=30)
    
    for client_id, (X, y) in client_datasets.items():
        print(f'   {client_id}: {X.shape} samples, '
              f'mean={X.mean():.4f}, std={X.std():.4f}')
    
    print('\n2. Initializing DANN model...')
    dann_module = DANNModule(
        input_length=1000, 
        latent_dim=128, 
        num_classes=5, 
        num_domains=4,
        dropout_rate=0.5
    )
    
    dann_trainer = DANNTrainer(
        model=dann_module,
        device=device,
        learning_rate=0.0005,
        lambda_class=1.0,
        lambda_domain=0.3,
        max_grad_norm=5.0,
        label_smoothing=0.1,
        use_gradient_penalty=True,
        gp_lambda=1.0,
        temperature=1.0,
        early_stopping_patience=5
    )
    
    normalized_datasets = {}
    for domain_id, (X, y) in client_datasets.items():
        X_normalized = normalize_signal(X)
        normalized_datasets[domain_id] = (X_normalized, y)
    
    print('\n3. Running DANN training (5 epochs)...\n')
    
    history = dann_trainer.train(
        normalized_datasets,
        epochs=5,
        batch_size=16,
        verbose=True
    )
    
    print('\n' + '=' * 70)
    print('Training Complete!')
    print('=' * 70)
    
    has_nan = False
    for key in ['total_loss', 'class_loss', 'domain_loss']:
        if key in history:
            losses = np.array(history[key])
            if np.isnan(losses).any() or np.isinf(losses).any():
                has_nan = True
                print(f'WARNING: NaN/Inf found in {key}')
    
    if not has_nan:
        print('\nNumerical Stability: PASSED (No NaN/Inf detected)')
    else:
        print('\nNumerical Stability: FAILED')
    
    if 'class_accuracy' in history:
        final_acc = history['class_accuracy'][-1]
        print(f'Final Classification Accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)')
        if final_acc > 0.5:
            print('Performance: Good')
        elif final_acc > 0.3:
            print('Performance: Moderate')
        else:
            print('Performance: Low (may need more training)')
    
    print('\n' + '=' * 70)
    print('Test Complete!')
    print('=' * 70)


if __name__ == '__main__':
    try:
        full_dann_test()
        sys.exit(0)
    except Exception as e:
        print(f'\n\nError during testing: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
