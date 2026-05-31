import { useState, useEffect } from 'react'
import { trainingApi, TrainingConfig as TrainingConfigType } from '../services/api'
import { Play, RefreshCw, Settings, Clock } from 'lucide-react'

const defaultConfig: TrainingConfigType = {
  num_rounds: 10,
  local_epochs: 5,
  use_dann: true,
  dann_interval: 3,
  dann_epochs: 10,
  learning_rate: 0.001,
  batch_size: 32,
  use_adapter: true,
  adapter_epochs: 10,
  adapter_lr: 0.001,
  adapter_dim: 64,
}

export default function Training() {
  const [config, setConfig] = useState<TrainingConfigType>(defaultConfig)
  const [trainingStatus, setTrainingStatus] = useState<any>(null)
  const [sessions, setSessions] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchTrainingStatus()
    fetchSessions()
    const interval = setInterval(fetchTrainingStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  const fetchTrainingStatus = async () => {
    try {
      const response = await trainingApi.getTrainingStatus()
      setTrainingStatus(response.data)
    } catch (error) {
      console.error('Failed to fetch training status:', error)
    }
  }

  const fetchSessions = async () => {
    try {
      const response = await trainingApi.getSessions()
      setSessions(response.data.sessions)
    } catch (error) {
      console.error('Failed to fetch sessions:', error)
    }
  }

  const handleStartTraining = async () => {
    try {
      setLoading(true)
      await trainingApi.startTraining(config)
      fetchTrainingStatus()
    } catch (error) {
      console.error('Failed to start training:', error)
      alert('Failed to start training. Training may already be in progress.')
    } finally {
      setLoading(false)
    }
  }

  const handleConfigChange = (key: keyof TrainingConfigType, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Training</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-gray-800">Training Status</h2>
                  <p className="text-sm text-gray-500">Current federated training progress</p>
                </div>
                <div className={`flex items-center px-4 py-2 rounded-full ${
                  trainingStatus?.is_training
                    ? 'bg-yellow-100 text-yellow-700'
                    : 'bg-green-100 text-green-700'
                }`}>
                  {trainingStatus?.is_training ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Training
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Ready
                    </>
                  )}
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-500">Current Round</p>
                  <p className="text-3xl font-bold text-gray-800">
                    {trainingStatus?.current_round || 0}
                    <span className="text-lg font-normal text-gray-400">
                      / {trainingStatus?.config?.num_rounds || config.num_rounds}
                    </span>
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-500">Final Accuracy</p>
                  <p className="text-3xl font-bold text-green-600">
                    {trainingStatus?.history?.final_accuracy
                      ? `${(trainingStatus.history.final_accuracy * 100).toFixed(1)}%`
                      : '-'
                    }
                  </p>
                </div>
              </div>

              {trainingStatus?.config && (
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-blue-800 mb-2">Current Configuration</h3>
                  <div className="grid grid-cols-2 gap-2 text-sm text-blue-700">
                    <p>Rounds: {trainingStatus.config.num_rounds}</p>
                    <p>Local Epochs: {trainingStatus.config.local_epochs}</p>
                    <p>DANN: {trainingStatus.config.use_dann ? 'Enabled' : 'Disabled'}</p>
                    <p>Batch Size: {trainingStatus.config.batch_size}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b">
              <h2 className="text-lg font-semibold text-gray-800">Training Sessions</h2>
              <p className="text-sm text-gray-500">Recent training sessions</p>
            </div>
            <div className="p-6">
              {sessions.length > 0 ? (
                <div className="space-y-2">
                  {sessions.slice(-5).reverse().map((session) => (
                    <div
                      key={session}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="flex items-center">
                        <Clock className="w-5 h-5 mr-3 text-gray-400" />
                        <span className="text-sm font-mono text-gray-600">{session}</span>
                      </div>
                      <span className="text-xs text-gray-400">Completed</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Clock className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                  <p>No training sessions yet</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b">
            <div className="flex items-center">
              <Settings className="w-5 h-5 mr-2 text-gray-600" />
              <h2 className="text-lg font-semibold text-gray-800">Configuration</h2>
            </div>
          </div>
          <div className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of Rounds
              </label>
              <input
                type="number"
                value={config.num_rounds}
                onChange={(e) => handleConfigChange('num_rounds', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="1"
                max="100"
                disabled={trainingStatus?.is_training}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Local Epochs per Round
              </label>
              <input
                type="number"
                value={config.local_epochs}
                onChange={(e) => handleConfigChange('local_epochs', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="1"
                max="20"
                disabled={trainingStatus?.is_training}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Batch Size
              </label>
              <input
                type="number"
                value={config.batch_size}
                onChange={(e) => handleConfigChange('batch_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="8"
                max="128"
                disabled={trainingStatus?.is_training}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                value={config.learning_rate}
                onChange={(e) => handleConfigChange('learning_rate', parseFloat(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                step="0.0001"
                min="0.0001"
                max="0.01"
                disabled={trainingStatus?.is_training}
              />
            </div>

            <div className="border-t pt-4">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-medium text-gray-700">Enable DANN</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.use_dann}
                    onChange={(e) => handleConfigChange('use_dann', e.target.checked)}
                    className="sr-only peer"
                    disabled={trainingStatus?.is_training}
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>

              {config.use_dann && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      DANN Interval
                    </label>
                    <input
                      type="number"
                      value={config.dann_interval}
                      onChange={(e) => handleConfigChange('dann_interval', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      min="1"
                      max="10"
                      disabled={trainingStatus?.is_training}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      DANN Epochs
                    </label>
                    <input
                      type="number"
                      value={config.dann_epochs}
                      onChange={(e) => handleConfigChange('dann_epochs', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      min="1"
                      max="30"
                      disabled={trainingStatus?.is_training}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="border-t pt-4">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-medium text-gray-700">Enable Local Adapter</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.use_adapter}
                    onChange={(e) => handleConfigChange('use_adapter', e.target.checked)}
                    className="sr-only peer"
                    disabled={trainingStatus?.is_training}
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-green-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                </label>
              </div>

              {config.use_adapter && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Adapter Dimension
                    </label>
                    <input
                      type="number"
                      value={config.adapter_dim}
                      onChange={(e) => handleConfigChange('adapter_dim', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                      min="16"
                      max="256"
                      disabled={trainingStatus?.is_training}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Adapter Epochs
                    </label>
                    <input
                      type="number"
                      value={config.adapter_epochs}
                      onChange={(e) => handleConfigChange('adapter_epochs', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                      min="1"
                      max="30"
                      disabled={trainingStatus?.is_training}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Adapter Learning Rate
                    </label>
                    <input
                      type="number"
                      value={config.adapter_lr}
                      onChange={(e) => handleConfigChange('adapter_lr', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                      step="0.0001"
                      min="0.0001"
                      max="0.01"
                      disabled={trainingStatus?.is_training}
                    />
                  </div>
                  <p className="text-xs text-gray-400">
                    Adapter parameters are kept local and not shared during federated aggregation.
                  </p>
                </div>
              )}
            </div>

            <button
              onClick={handleStartTraining}
              disabled={trainingStatus?.is_training || loading}
              className={`w-full py-3 px-4 rounded-lg font-medium flex items-center justify-center ${
                trainingStatus?.is_training || loading
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {loading ? (
                <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
              ) : (
                <Play className="w-5 h-5 mr-2" />
              )}
              {trainingStatus?.is_training ? 'Training in Progress...' : 'Start Training'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
