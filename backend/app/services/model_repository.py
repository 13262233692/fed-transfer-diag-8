import torch
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import copy

from ..models.cnn_feature_extractor import FaultDiagnosisModel


class ModelRepository:
    def __init__(self, storage_dir: str = './models'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.versions_dir = os.path.join(storage_dir, 'versions')
        os.makedirs(self.versions_dir, exist_ok=True)
        
        self.metadata_file = os.path.join(storage_dir, 'model_metadata.json')
        self._load_metadata()
    
    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'models': {},
                'latest_version': None
            }
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(
        self,
        model: FaultDiagnosisModel,
        version_name: str,
        description: str = '',
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        timestamp = datetime.now().isoformat()
        
        version_id = f"{version_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = os.path.join(self.versions_dir, f'{version_id}.pt')
        torch.save(model.state_dict(), model_path)
        
        model_info = {
            'version_id': version_id,
            'version_name': version_name,
            'timestamp': timestamp,
            'description': description,
            'metrics': metrics or {},
            'path': model_path
        }
        
        self.metadata['models'][version_id] = model_info
        self.metadata['latest_version'] = version_id
        self._save_metadata()
        
        print(f'Model saved: {version_id}')
        return version_id
    
    def load_model(
        self,
        version_id: Optional[str] = None,
        model_class: type = FaultDiagnosisModel,
        **model_kwargs
    ) -> FaultDiagnosisModel:
        if version_id is None:
            version_id = self.metadata.get('latest_version')
            if version_id is None:
                raise ValueError('No models available in repository')
        
        if version_id not in self.metadata['models']:
            raise ValueError(f'Model version {version_id} not found')
        
        model_path = self.metadata['models'][version_id]['path']
        
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(model_path))
        
        return model
    
    def get_model_info(self, version_id: str) -> Optional[Dict]:
        return self.metadata['models'].get(version_id)
    
    def list_models(self) -> List[Dict]:
        return list(self.metadata['models'].values())
    
    def get_latest_model(self) -> Optional[Dict]:
        latest_version = self.metadata.get('latest_version')
        if latest_version:
            return self.get_model_info(latest_version)
        return None
    
    def delete_model(self, version_id: str) -> bool:
        if version_id not in self.metadata['models']:
            return False
        
        model_path = self.metadata['models'][version_id]['path']
        if os.path.exists(model_path):
            os.remove(model_path)
        
        del self.metadata['models'][version_id]
        
        if self.metadata.get('latest_version') == version_id:
            remaining_models = list(self.metadata['models'].keys())
            self.metadata['latest_version'] = remaining_models[-1] if remaining_models else None
        
        self._save_metadata()
        print(f'Model deleted: {version_id}')
        return True


class TrainingLogger:
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.current_session_id = None
        self.session_data = None
    
    def start_session(
        self,
        session_name: str,
        config: Dict[str, Any]
    ) -> str:
        self.current_session_id = f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_data = {
            'session_id': self.current_session_id,
            'session_name': session_name,
            'start_time': datetime.now().isoformat(),
            'config': config,
            'rounds': [],
            'client_metrics': {},
            'dann_metrics': []
        }
        
        return self.current_session_id
    
    def log_round(
        self,
        round_num: int,
        global_metrics: Dict[str, float],
        client_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ):
        if self.session_data is None:
            return
        
        round_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'global_metrics': global_metrics
        }
        self.session_data['rounds'].append(round_data)
        
        if client_metrics:
            for client_id, metrics in client_metrics.items():
                if client_id not in self.session_data['client_metrics']:
                    self.session_data['client_metrics'][client_id] = []
                self.session_data['client_metrics'][client_id].append({
                    'round': round_num,
                    **metrics
                })
    
    def log_dann_metrics(
        self,
        round_num: int,
        dann_metrics: Dict[str, List[float]]
    ):
        if self.session_data is None:
            return
        
        self.session_data['dann_metrics'].append({
            'round': round_num,
            'metrics': dann_metrics
        })
    
    def end_session(self):
        if self.session_data is None:
            return
        
        self.session_data['end_time'] = datetime.now().isoformat()
        
        session_file = os.path.join(self.log_dir, f'{self.current_session_id}.json')
        with open(session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        print(f'Training session saved: {self.current_session_id}')
        
        self.current_session_id = None
        self.session_data = None
    
    def list_sessions(self) -> List[str]:
        return sorted([f for f in os.listdir(self.log_dir) if f.endswith('.json')])
    
    def load_session(self, session_id: str) -> Dict:
        session_file = os.path.join(self.log_dir, f'{session_id}.json')
        if not os.path.exists(session_file):
            raise ValueError(f'Session {session_id} not found')
        
        with open(session_file, 'r') as f:
            return json.load(f)
