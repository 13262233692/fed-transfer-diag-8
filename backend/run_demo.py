import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from app.core.data_simulator import FaultDiagnosticsDataSimulator
from app.models.cnn_feature_extractor import FaultDiagnosisModel
from app.models.dann_module import DANNModule
from app.services.federated_coordinator import FederatedCoordinator


def main():
    print('=' * 60)
    print('Federated Transfer Learning - Fault Diagnosis Demo')
    print('=' * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nUsing device: {device}')
    
    print('\n1. Generating simulated vibration data for 4 factories...')
    simulator = FaultDiagnosticsDataSimulator()
    client_datasets = simulator.generate_federated_datasets(n_samples_per_class=50)
    
    for client_id, (X, y) in client_datasets.items():
        print(f'   {client_id}: {X.shape} samples, {len(np.unique(y))} classes')
    
    X_test, y_test = simulator.generate_dataset('factory_A', n_samples_per_class=20)
    test_dataset = (X_test, y_test)
    print(f'   Test set: {X_test.shape} samples')
    
    print('\n2. Initializing models...')
    global_model = FaultDiagnosisModel(input_length=1000, latent_dim=128, num_classes=5)
    dann_module = DANNModule(input_length=1000, latent_dim=128, num_classes=5, num_domains=4)
    
    coordinator = FederatedCoordinator(global_model, dann_module, device=device)
    print('   Models initialized successfully')
    
    print('\n3. Starting federated training with DANN domain alignment...')
    print('   (This will take a few minutes...)')
    print()
    
    history = coordinator.run_federated_training(
        client_datasets=client_datasets,
        test_dataset=test_dataset,
        num_rounds=5,
        local_epochs=3,
        use_dann=True,
        dann_interval=2,
        dann_epochs=5
    )
    
    print('\n' + '=' * 60)
    print('Training Complete!')
    print('=' * 60)
    
    final_acc = history['global_evaluation'][-1]['accuracy']
    final_loss = history['global_evaluation'][-1]['loss']
    print(f'\nFinal Global Model Performance:')
    print(f'   Accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)')
    print(f'   Loss: {final_loss:.4f}')
    
    print(f'\nTotal Rounds: {len(history["rounds"])}')
    print(f'DANN Alignment: Performed {len(history["dann_history"])} times')
    
    print('\n' + '=' * 60)
    print('Per-Client Training Results (Last Round)')
    print('=' * 60)
    
    last_round = max(history['client_histories'].keys())
    for client_id, hist in history['client_histories'][last_round].items():
        print(f'\n{client_id}:')
        print(f'   Final Train Accuracy: {hist["train_acc"][-1]:.4f}')
        print(f'   Final Val Accuracy:   {hist["val_acc"][-1]:.4f}')
    
    print('\n' + '=' * 60)
    print('Demo completed successfully!')
    print('=' * 60)
    
    return history


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nDemo interrupted by user')
        sys.exit(0)
    except Exception as e:
        print(f'\n\nError during demo: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
