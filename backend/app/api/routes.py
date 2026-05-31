from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import torch
import numpy as np
from datetime import datetime
import asyncio

from ..schemas.training_schemas import (
    TrainingConfig, SystemStatus, ClientStatus,
    ModelInfo, PredictionRequest, PredictionResponse,
    AdapterComparisonItem, AdapterComparisonResponse
)
from ..core.data_simulator import FaultDiagnosticsDataSimulator, normalize_signal
from ..models.cnn_feature_extractor import FaultDiagnosisModel
from ..models.dann_module import DANNModule
from ..services.federated_coordinator import FederatedCoordinator
from ..services.model_repository import ModelRepository, TrainingLogger

router = APIRouter()

data_simulator = FaultDiagnosticsDataSimulator()

global_model = FaultDiagnosisModel(input_length=1000, latent_dim=128, num_classes=5)
dann_module = DANNModule(input_length=1000, latent_dim=128, num_classes=5, num_domains=4)

coordinator = FederatedCoordinator(global_model, dann_module, device='cpu', adapter_dim=64)
model_repo = ModelRepository(storage_dir='./models')
training_logger = TrainingLogger(log_dir='./logs')

training_state = {
    'is_training': False,
    'current_config': None,
    'training_history': None
}

client_datasets = {}


@router.on_event("startup")
async def startup_event():
    global client_datasets
    datasets = data_simulator.generate_federated_datasets(n_samples_per_class=50)
    client_datasets = datasets
    print(f"Generated datasets for {len(datasets)} clients")


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    global coordinator, training_state, model_repo
    
    clients = []
    adapter_comp = coordinator.get_adapter_comparison()
    
    for client_id, trainer in coordinator.client_manager.clients.items():
        data_size = len(client_datasets.get(client_id, [[]])[0]) if client_id in client_datasets else 0
        comp = adapter_comp.get(client_id, {})
        clients.append(ClientStatus(
            client_id=client_id,
            status='registered',
            data_size=data_size,
            last_training_time=None,
            has_adapter=comp.get('has_adapter', False),
            before_adapter_accuracy=comp.get('before_adapter_accuracy'),
            after_adapter_accuracy=comp.get('after_adapter_accuracy')
        ))
    
    latest_model = model_repo.get_latest_model()
    
    adapter_done = any(
        comp.get('has_adapter', False) for comp in adapter_comp.values()
    )
    
    return SystemStatus(
        status='running' if not training_state['is_training'] else 'training',
        num_clients=len(clients),
        current_round=coordinator.current_round,
        is_training=training_state['is_training'],
        latest_model_version=latest_model['version_id'] if latest_model else None,
        active_clients=clients,
        adapter_personalization_done=adapter_done
    )


@router.get("/clients")
async def get_clients():
    global client_datasets, coordinator
    adapter_comp = coordinator.get_adapter_comparison()
    
    return {
        'clients': [
            {
                'client_id': client_id,
                'data_size': len(X),
                'num_classes': len(np.unique(y)),
                'has_adapter': adapter_comp.get(client_id, {}).get('has_adapter', False),
                'before_adapter_accuracy': adapter_comp.get(client_id, {}).get('before_adapter_accuracy'),
                'after_adapter_accuracy': adapter_comp.get(client_id, {}).get('after_adapter_accuracy')
            }
            for client_id, (X, y) in client_datasets.items()
        ]
    }


@router.get("/datasets/{client_id}")
async def get_dataset_info(client_id: str):
    global client_datasets
    if client_id not in client_datasets:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    
    X, y = client_datasets[client_id]
    return {
        'client_id': client_id,
        'num_samples': len(X),
        'signal_length': X.shape[1] if len(X) > 0 else 0,
        'class_distribution': {
            int(cls): int(count) for cls, count in zip(*np.unique(y, return_counts=True))
        }
    }


@router.get("/adapter/comparison", response_model=AdapterComparisonResponse)
async def get_adapter_comparison():
    global coordinator
    
    comp = coordinator.get_adapter_comparison()
    
    items = []
    before_accs = []
    after_accs = []
    improvements = []
    
    for client_id, data in comp.items():
        item = AdapterComparisonItem(
            client_id=client_id,
            before_adapter_accuracy=data.get('before_adapter_accuracy'),
            after_adapter_accuracy=data.get('after_adapter_accuracy'),
            improvement=data.get('improvement'),
            has_adapter=data.get('has_adapter', False)
        )
        items.append(item)
        
        if data.get('before_adapter_accuracy') is not None:
            before_accs.append(data['before_adapter_accuracy'])
        if data.get('after_adapter_accuracy') is not None:
            after_accs.append(data['after_adapter_accuracy'])
        if data.get('improvement') is not None:
            improvements.append(data['improvement'])
    
    return AdapterComparisonResponse(
        clients=items,
        avg_before_accuracy=np.mean(before_accs) if before_accs else None,
        avg_after_accuracy=np.mean(after_accs) if after_accs else None,
        avg_improvement=np.mean(improvements) if improvements else None
    )


@router.post("/adapter/train")
async def train_adapters(
    adapter_epochs: int = 10,
    adapter_lr: float = 0.001,
    background_tasks: BackgroundTasks = None
):
    global coordinator, client_datasets, training_state
    
    if training_state['is_training']:
        raise HTTPException(status_code=400, detail="Training is already in progress")
    
    if not coordinator.client_manager.clients:
        client_ids = list(client_datasets.keys())
        coordinator.register_clients(client_ids)
    
    training_state['is_training'] = True
    
    if background_tasks:
        background_tasks.add_task(run_adapter_training, adapter_epochs, adapter_lr)
    else:
        await run_adapter_training(adapter_epochs, adapter_lr)
    
    return {
        'message': 'Adapter personalization started',
        'adapter_epochs': adapter_epochs,
        'adapter_lr': adapter_lr
    }


async def run_adapter_training(adapter_epochs: int = 10, adapter_lr: float = 0.001):
    global coordinator, client_datasets, training_state
    
    try:
        coordinator.personalize_with_adapters(
            client_datasets,
            adapter_epochs=adapter_epochs,
            adapter_lr=adapter_lr
        )
    except Exception as e:
        print(f"Adapter training error: {e}")
    finally:
        training_state['is_training'] = False


@router.post("/training/start")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks
):
    global training_state, coordinator, client_datasets, model_repo, training_logger
    
    if training_state['is_training']:
        raise HTTPException(status_code=400, detail="Training is already in progress")
    
    training_state['is_training'] = True
    training_state['current_config'] = config.dict()
    
    coordinator.adapter_dim = config.adapter_dim
    
    background_tasks.add_task(run_federated_training, config)
    
    return {
        'message': 'Training started',
        'config': config.dict()
    }


async def run_federated_training(config: TrainingConfig):
    global training_state, coordinator, client_datasets, model_repo, training_logger
    
    try:
        X_test, y_test = data_simulator.generate_dataset('factory_A', n_samples_per_class=20)
        test_dataset = (X_test, y_test)
        
        session_id = training_logger.start_session(
            'federated_training',
            config.dict()
        )
        
        client_ids = list(client_datasets.keys())
        coordinator.register_clients(client_ids)
        
        client_data_sizes = {
            cid: len(dataset[0]) for cid, dataset in client_datasets.items()
        }
        
        for round_idx in range(config.num_rounds):
            coordinator.current_round = round_idx
            
            print(f"\n=== Round {round_idx + 1}/{config.num_rounds} ===")
            
            histories = coordinator.train_clients_one_round(
                client_datasets, local_epochs=config.local_epochs
            )
            
            coordinator.aggregate_parameters(client_data_sizes)
            coordinator.broadcast_global_model()
            
            if config.use_dann and (round_idx + 1) % config.dann_interval == 0:
                dann_history = coordinator.dann_alignment(
                    client_datasets, dann_epochs=config.dann_epochs
                )
                training_logger.log_dann_metrics(round_idx + 1, dann_history)
                coordinator.broadcast_global_model()
            
            global_metrics = coordinator.evaluate_global_model(test_dataset)
            
            round_client_metrics = {}
            for client_id, hist in histories.items():
                round_client_metrics[client_id] = {
                    'final_train_loss': hist['train_loss'][-1],
                    'final_train_acc': hist['train_acc'][-1],
                    'final_val_loss': hist['val_loss'][-1],
                    'final_val_acc': hist['val_acc'][-1]
                }
            
            training_logger.log_round(round_idx + 1, global_metrics, round_client_metrics)
            
            print(f"Round {round_idx + 1} - Global Accuracy: {global_metrics['accuracy']:.4f}")
        
        if config.use_adapter:
            coordinator.personalize_with_adapters(
                client_datasets,
                adapter_epochs=config.adapter_epochs,
                adapter_lr=config.adapter_lr
            )
        
        version_id = model_repo.save_model(
            coordinator.global_model,
            'global_model',
            f'Federated training with {config.num_rounds} rounds',
            global_metrics
        )
        
        training_logger.end_session()
        
        adapter_comp = coordinator.get_adapter_comparison()
        
        training_state['training_history'] = {
            'session_id': session_id,
            'model_version': version_id,
            'final_accuracy': global_metrics['accuracy'],
            'adapter_comparison': {
                cid: {
                    'before': comp.get('before_adapter_accuracy'),
                    'after': comp.get('after_adapter_accuracy'),
                    'improvement': comp.get('improvement')
                }
                for cid, comp in adapter_comp.items()
            }
        }
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        training_state['is_training'] = False


@router.get("/training/status")
async def get_training_status():
    global training_state, coordinator
    
    return {
        'is_training': training_state['is_training'],
        'current_round': coordinator.current_round,
        'config': training_state['current_config'],
        'history': training_state['training_history']
    }


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    global model_repo
    models = model_repo.list_models()
    return [
        ModelInfo(
            version_id=m['version_id'],
            version_name=m['version_name'],
            timestamp=m['timestamp'],
            description=m['description'],
            metrics=m['metrics']
        )
        for m in models
    ]


@router.get("/models/{version_id}")
async def get_model_info(version_id: str):
    global model_repo
    model_info = model_repo.get_model_info(version_id)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    return model_info


@router.delete("/models/{version_id}")
async def delete_model(version_id: str):
    global model_repo
    success = model_repo.delete_model(version_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {'message': f'Model {version_id} deleted'}


@router.get("/training/sessions")
async def list_training_sessions():
    global training_logger
    sessions = training_logger.list_sessions()
    return {'sessions': sessions}


@router.get("/training/sessions/{session_id}")
async def get_training_session(session_id: str):
    global training_logger
    try:
        session_data = training_logger.load_session(session_id)
        return session_data
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global coordinator, model_repo
    
    signal = np.array(request.signal_data, dtype=np.float32)
    if len(signal) != 1000:
        raise HTTPException(status_code=400, detail="Signal must be 1000 samples long")
    
    signal = normalize_signal(signal.reshape(1, -1))
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    
    coordinator.global_model.eval()
    with torch.no_grad():
        _, logits = coordinator.global_model(signal_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=probabilities[0].tolist(),
        prediction_time=datetime.now().isoformat()
    )


@router.get("/simulate/generate/{client_id}")
async def generate_simulated_data(client_id: str, n_samples: int = 100):
    global data_simulator, client_datasets
    
    X, y = data_simulator.generate_dataset(client_id, n_samples_per_class=n_samples)
    client_datasets[client_id] = (X, y)
    
    return {
        'message': f'Dataset generated for {client_id}',
        'num_samples': len(X),
        'num_classes': len(np.unique(y))
    }


@router.get("/fault-types")
async def get_fault_types():
    return {
        'fault_types': ['normal', 'bearing', 'gear', 'unbalance', 'misalignment']
    }
