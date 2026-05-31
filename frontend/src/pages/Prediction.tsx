import { useState, useEffect } from 'react'
import { predictionApi, PredictionResponse } from '../services/api'
import { Activity, Play, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react'

const faultTypes = ['Normal', 'Bearing Fault', 'Gear Fault', 'Unbalance', 'Misalignment']

export default function Prediction() {
  const [signalData, setSignalData] = useState<number[]>([])
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [faultTypesList, setFaultTypesList] = useState<string[]>([])

  useEffect(() => {
    fetchFaultTypes()
  }, [])

  const fetchFaultTypes = async () => {
    try {
      const response = await predictionApi.getFaultTypes()
      setFaultTypesList(response.data.fault_types)
    } catch (error) {
      console.error('Failed to fetch fault types:', error)
    }
  }

  const generateSampleSignal = () => {
    const sampleRate = 1000
    const signal: number[] = []
    const baseFreq = 50 + Math.random() * 50

    for (let i = 0; i < sampleRate; i++) {
      const t = i / sampleRate
      let value = Math.sin(2 * Math.PI * baseFreq * t)
      value += 0.5 * Math.sin(2 * Math.PI * baseFreq * 2 * t)
      value += 0.25 * Math.sin(2 * Math.PI * baseFreq * 3 * t)
      value += (Math.random() - 0.5) * 0.3
      signal.push(value)
    }

    setSignalData(signal)
    setPrediction(null)
  }

  const handlePredict = async () => {
    if (signalData.length !== 1000) {
      alert('Please generate a signal first (1000 samples required)')
      return
    }

    try {
      setLoading(true)
      const response = await predictionApi.predict({ signal_data: signalData })
      setPrediction(response.data)
    } catch (error) {
      console.error('Prediction failed:', error)
      alert('Prediction failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (className: number) => {
    if (className === 0) return 'text-green-600 bg-green-100'
    return 'text-red-600 bg-red-100'
  }

  const getStatusIcon = (className: number) => {
    if (className === 0) return <CheckCircle className="w-5 h-5" />
    return <AlertCircle className="w-5 h-5" />
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Fault Diagnosis</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Activity className="w-5 h-5 mr-2 text-gray-600" />
                <h2 className="text-lg font-semibold text-gray-800">Signal Input</h2>
              </div>
              <button
                onClick={generateSampleSignal}
                className="px-4 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              >
                Generate Sample
              </button>
            </div>
          </div>
          <div className="p-6">
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Vibration Signal ({signalData.length} samples)
              </label>
              <div className="h-40 bg-gray-50 rounded-lg border border-gray-200 overflow-hidden">
                {signalData.length > 0 ? (
                  <svg viewBox="0 0 1000 160" className="w-full h-full">
                    <polyline
                      fill="none"
                      stroke="#3B82F6"
                      strokeWidth="1"
                      points={signalData
                        .map((val, i) => {
                          const x = i
                          const y = 80 + val * 30
                          return `${x},${y}`
                        })
                        .join(' ')}
                    />
                  </svg>
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-400">
                    Click "Generate Sample" to create a test signal
                  </div>
                )}
              </div>
            </div>

            <div className="bg-blue-50 p-4 rounded-lg mb-6">
              <h3 className="font-semibold text-blue-800 mb-2">Fault Types</h3>
              <div className="grid grid-cols-2 gap-2 text-sm text-blue-700">
                {faultTypesList.map((type, idx) => (
                  <div key={type} className="flex items-center">
                    <span className="w-5 h-5 bg-blue-200 rounded mr-2 flex items-center justify-center text-xs font-medium">
                      {idx}
                    </span>
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={signalData.length !== 1000 || loading}
              className={`w-full py-3 px-4 rounded-lg font-medium flex items-center justify-center ${
                signalData.length !== 1000 || loading
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              {loading ? (
                <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
              ) : (
                <Play className="w-5 h-5 mr-2" />
              )}
              Run Diagnosis
            </button>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b">
            <h2 className="text-lg font-semibold text-gray-800">Diagnosis Result</h2>
          </div>
          <div className="p-6">
            {prediction ? (
              <div className="space-y-6">
                <div className={`flex items-center p-4 rounded-lg ${getStatusColor(prediction.predicted_class)}`}>
                  {getStatusIcon(prediction.predicted_class)}
                  <div className="ml-3">
                    <p className="text-sm font-medium">
                      {prediction.predicted_class === 0 ? 'Normal Operation' : 'Fault Detected'}
                    </p>
                    <p className="text-lg font-bold">
                      {faultTypes[prediction.predicted_class]}
                    </p>
                  </div>
                  <div className="ml-auto text-right">
                    <p className="text-sm opacity-75">Confidence</p>
                    <p className="text-lg font-bold">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div>
                  <h3 className="font-medium text-gray-700 mb-4">Class Probabilities</h3>
                  <div className="space-y-3">
                    {prediction.class_probabilities.map((prob, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className={idx === prediction.predicted_class ? 'font-semibold text-gray-800' : 'text-gray-600'}>
                            {faultTypes[idx]}
                          </span>
                          <span className={idx === prediction.predicted_class ? 'font-semibold text-gray-800' : 'text-gray-600'}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full transition-all ${
                              idx === 0
                                ? idx === prediction.predicted_class
                                  ? 'bg-green-500'
                                  : 'bg-green-300'
                                : idx === prediction.predicted_class
                                ? 'bg-red-500'
                                : 'bg-red-300'
                            }`}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="text-xs text-gray-400 text-right">
                  Prediction time: {new Date(prediction.prediction_time).toLocaleString()}
                </div>
              </div>
            ) : (
              <div className="text-center py-16 text-gray-500">
                <Activity className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                <p className="mb-2">No diagnosis yet</p>
                <p className="text-sm">Generate a signal and run diagnosis to see results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
