import { useState, useEffect } from 'react'
import { modelApi, ModelInfo } from '../services/api'
import { Database, Trash2, Download, Clock } from 'lucide-react'

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchModels()
  }, [])

  const fetchModels = async () => {
    try {
      const response = await modelApi.listModels()
      setModels(response.data)
    } catch (error) {
      console.error('Failed to fetch models:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteModel = async (versionId: string) => {
    if (confirm(`Are you sure you want to delete model ${versionId}?`)) {
      try {
        await modelApi.deleteModel(versionId)
        fetchModels()
      } catch (error) {
        console.error('Failed to delete model:', error)
      }
    }
  }

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Model Repository</h1>
        <span className="text-sm text-gray-500">{models.length} models stored</span>
      </div>

      {models.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models.map((model) => (
            <div
              key={model.version_id}
              className="bg-white rounded-lg shadow hover:shadow-md transition-shadow"
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                    <Database className="w-6 h-6 text-purple-600" />
                  </div>
                  <div className="flex space-x-2">
                    <button
                      className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded transition-colors"
                      title="Download"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDeleteModel(model.version_id)}
                      className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <h3 className="font-semibold text-gray-800 mb-1">{model.version_name}</h3>
                <p className="text-xs text-gray-400 font-mono mb-3">{model.version_id}</p>

                {model.description && (
                  <p className="text-sm text-gray-600 mb-4">{model.description}</p>
                )}

                <div className="border-t pt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-500">Accuracy</span>
                    <span className="font-semibold text-green-600">
                      {model.metrics?.accuracy
                        ? `${(model.metrics.accuracy * 100).toFixed(1)}%`
                        : '-'
                      }
                    </span>
                  </div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-gray-500">Loss</span>
                    <span className="font-medium text-gray-700">
                      {model.metrics?.loss?.toFixed(4) || '-'}
                    </span>
                  </div>
                  <div className="flex items-center text-xs text-gray-400">
                    <Clock className="w-3 h-3 mr-1" />
                    {formatDate(model.timestamp)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <Database className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <h3 className="text-lg font-medium text-gray-600 mb-2">No Models Yet</h3>
          <p className="text-gray-500 mb-6">
            Start a training session to generate your first model.
          </p>
          <button
            onClick={() => (window.location.href = '/training')}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Go to Training
          </button>
        </div>
      )}
    </div>
  )
}
