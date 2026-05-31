import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from app.core.data_simulator import FaultDiagnosticsDataSimulator
from app.models.cnn_feature_extractor import FaultDiagnosisModel
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


def test_dann_stability():
    print('=' * 70)
    print('DANN Stability Test - Extreme Data Distribution (Speed 0.1x to 20x)')
    print('=' * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nUsing device: {device}')
    
    print('\n1. Generating extreme domain shift data...')
    simulator = ExtremeDataSimulator()
    client_datasets = simulator.generate_federated_datasets(n_samples_per_class=80)
    
    for client_id, (X, y) in client_datasets.items():
        print(f'   {client_id}: {X.shape} samples, '
              f'mean={X.mean():.4f}, std={X.std():.4f}')
    
    print('\n2. Initializing DANN model with stability improvements...')
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
        gp_lambda=10.0,
        temperature=1.0,
        early_stopping_patience=5
    )
    print('   DANN model initialized with stability enhancements')
    
    print('\n3. Running DANN training on extreme domain data...')
    print('   (This tests for gradient vanishing, loss NaN, and numerical instability)\n')
    
    from app.core.data_simulator import normalize_signal
    normalized_datasets = {}
    for domain_id, (X, y) in client_datasets.items():
        X_normalized = normalize_signal(X)
        normalized_datasets[domain_id] = (X_normalized, y)
    
    history = dann_trainer.train(
        normalized_datasets,
        epochs=15,
        batch_size=32,
        verbose=True
    )
    
    print('\n' + '=' * 70)
    print('Training Complete - Stability Analysis')
    print('=' * 70)
    
    has_nan = any(
        np.isnan(loss).any() or np.isinf(loss).any()
        for loss in [history['total_loss'], history['class_loss'], history['domain_loss']]
    )
    
    final_class_acc = history['class_accuracy'][-1]
    final_domain_acc = history['domain_accuracy'][-1]
    
    print(f'\nNumerical Stability Check:')
    print(f'   Loss contains NaN/Inf: {"YES (FAILED)" if has_nan else "NO (PASSED)"}')
    
    print(f'\nFinal Performance:')
    print(f'   Classification Accuracy: {final_class_acc:.4f} ({final_class_acc * 100:.2f}%)')
    print(f'   Domain Discrimination Accuracy: {final_domain_acc:.4f} ({final_domain_acc * 100:.2f}%)')
    
    print(f'\nLoss Trends:')
    print(f'   Initial Total Loss: {history["total_loss"][0]:.6f}')
    print(f'   Final Total Loss:   {history["total_loss"][-1]:.6f}')
    print(f'   Initial Class Loss: {history["class_loss"][0]:.6f}')
    print(f'   Final Class Loss:   {history["class_loss"][-1]:.6f}')
    
    acc_trend = 'improving' if final_class_acc > history['class_accuracy'][0] else 'deteriorating'
    print(f'\nClassification accuracy is {acc_trend}')
    
    if has_nan:
        print('\n' + '!' * 70)
        print('WARNING: NaN/Inf detected in losses!')
        print('!' * 70)
        return False
    elif final_class_acc < 0.4:
        print('\n' + '!' * 70)
        print('WARNING: Classification accuracy is very low!')
        print('!' * 70)
        return False
    else:
        print('\n' + '=' * 70)
        print('STABILITY TEST PASSED!')
        print('DANN module handles extreme domain shift without numerical issues.')
        print('=' * 70)
        return True


if __name__ == '__main__':
    try:
        success = test_dann_stability()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'\n\nError during testing: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
