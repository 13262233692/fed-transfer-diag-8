import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface SystemStatus {
  status: string
  num_clients: number
  current_round: number
  is_training: boolean
  latest_model_version: string | null
  active_clients: ClientStatus[]
  adapter_personalization_done: boolean
}

export interface ClientStatus {
  client_id: string
  status: string
  data_size: number
  last_training_time: string | null
  has_adapter: boolean
  before_adapter_accuracy: number | null
  after_adapter_accuracy: number | null
}

export interface TrainingConfig {
  num_rounds: number
  local_epochs: number
  use_dann: boolean
  dann_interval: number
  dann_epochs: number
  learning_rate: number
  batch_size: number
  use_adapter: boolean
  adapter_epochs: number
  adapter_lr: number
  adapter_dim: number
}

export interface ModelInfo {
  version_id: string
  version_name: string
  timestamp: string
  description: string
  metrics: {
    accuracy?: number
    loss?: number
  }
}

export interface PredictionRequest {
  signal_data: number[]
}

export interface PredictionResponse {
  predicted_class: number
  confidence: number
  class_probabilities: number[]
  prediction_time: string
}

export interface AdapterComparisonItem {
  client_id: string
  before_adapter_accuracy: number | null
  after_adapter_accuracy: number | null
  improvement: number | null
  has_adapter: boolean
}

export interface AdapterComparisonResponse {
  clients: AdapterComparisonItem[]
  avg_before_accuracy: number | null
  avg_after_accuracy: number | null
  avg_improvement: number | null
}

export const systemApi = {
  getStatus: () => api.get<SystemStatus>('/status'),
  getClients: () => api.get('/clients'),
  getDatasetInfo: (clientId: string) => api.get(`/datasets/${clientId}`),
}

export const trainingApi = {
  startTraining: (config: TrainingConfig) =>
    api.post('/training/start', config),
  getTrainingStatus: () => api.get('/training/status'),
  getSessions: () => api.get('/training/sessions'),
  getSession: (sessionId: string) => api.get(`/training/sessions/${sessionId}`),
}

export const adapterApi = {
  getComparison: () => api.get<AdapterComparisonResponse>('/adapter/comparison'),
  trainAdapters: (adapterEpochs: number = 10, adapterLr: number = 0.001) =>
    api.post('/adapter/train', null, { params: { adapter_epochs: adapterEpochs, adapter_lr: adapterLr } }),
}

export const modelApi = {
  listModels: () => api.get<ModelInfo[]>('/models'),
  getModelInfo: (versionId: string) => api.get(`/models/${versionId}`),
  deleteModel: (versionId: string) => api.delete(`/models/${versionId}`),
}

export const predictionApi = {
  predict: (data: PredictionRequest) =>
    api.post<PredictionResponse>('/predict', data),
  getFaultTypes: () => api.get('/fault-types'),
}

export default api
