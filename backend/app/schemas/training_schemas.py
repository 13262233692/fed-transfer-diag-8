from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class ClientMetrics(BaseModel):
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]


class GlobalMetrics(BaseModel):
    loss: float
    accuracy: float


class RoundData(BaseModel):
    round: int
    global_metrics: GlobalMetrics


class TrainingSession(BaseModel):
    session_id: str
    session_name: str
    start_time: str
    end_time: Optional[str] = None
    config: Dict[str, Any]
    rounds: List[RoundData]
    client_metrics: Dict[str, List[Dict[str, Any]]]
    dann_metrics: List[Dict[str, Any]]


class ModelInfo(BaseModel):
    version_id: str
    version_name: str
    timestamp: str
    description: str
    metrics: Dict[str, float]


class TrainingConfig(BaseModel):
    num_rounds: int = Field(default=10, ge=1, le=100)
    local_epochs: int = Field(default=5, ge=1, le=20)
    use_dann: bool = True
    dann_interval: int = Field(default=3, ge=1, le=10)
    dann_epochs: int = Field(default=10, ge=1, le=30)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.01)
    batch_size: int = Field(default=32, ge=8, le=128)
    use_adapter: bool = True
    adapter_epochs: int = Field(default=10, ge=1, le=30)
    adapter_lr: float = Field(default=0.001, ge=0.0001, le=0.01)
    adapter_dim: int = Field(default=64, ge=16, le=256)


class ClientStatus(BaseModel):
    client_id: str
    status: str
    data_size: int
    last_training_time: Optional[str] = None
    has_adapter: bool = False
    before_adapter_accuracy: Optional[float] = None
    after_adapter_accuracy: Optional[float] = None


class AdapterComparisonItem(BaseModel):
    client_id: str
    before_adapter_accuracy: Optional[float] = None
    after_adapter_accuracy: Optional[float] = None
    improvement: Optional[float] = None
    has_adapter: bool


class AdapterComparisonResponse(BaseModel):
    clients: List[AdapterComparisonItem]
    avg_before_accuracy: Optional[float] = None
    avg_after_accuracy: Optional[float] = None
    avg_improvement: Optional[float] = None


class SystemStatus(BaseModel):
    status: str
    num_clients: int
    current_round: int
    is_training: bool
    latest_model_version: Optional[str] = None
    active_clients: List[ClientStatus]
    adapter_personalization_done: bool = False


class PredictionRequest(BaseModel):
    signal_data: List[float]


class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float
    class_probabilities: List[float]
    prediction_time: str
