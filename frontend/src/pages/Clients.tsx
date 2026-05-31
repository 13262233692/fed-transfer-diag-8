import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { systemApi, adapterApi, AdapterComparisonItem } from '../services/api'
import { Server, HardDrive, BarChart3, Zap, TrendingUp, ArrowUpRight } from 'lucide-react'

interface ClientData {
  client_id: string
  data_size: number
  num_classes: number
  has_adapter: boolean
  before_adapter_accuracy: number | null
  after_adapter_accuracy: number | null
}

export default function Clients() {
  const [clients, setClients] = useState<ClientData[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedClient, setSelectedClient] = useState<string | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<any>(null)
  const [adapterData, setAdapterData] = useState<AdapterComparisonItem[]>([])
  const [avgBefore, setAvgBefore] = useState<number | null>(null)
  const [avgAfter, setAvgAfter] = useState<number | null>(null)
  const [avgImprovement, setAvgImprovement] = useState<number | null>(null)

  useEffect(() => {
    fetchClients()
    fetchAdapterComparison()
    const interval = setInterval(() => {
      fetchAdapterComparison()
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchClients = async () => {
    try {
      const response = await systemApi.getClients()
      setClients(response.data.clients)
    } catch (error) {
      console.error('Failed to fetch clients:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchAdapterComparison = async () => {
    try {
      const response = await adapterApi.getComparison()
      setAdapterData(response.data.clients)
      setAvgBefore(response.data.avg_before_accuracy)
      setAvgAfter(response.data.avg_after_accuracy)
      setAvgImprovement(response.data.avg_improvement)
    } catch (error) {
      console.error('Failed to fetch adapter comparison:', error)
    }
  }

  const fetchDatasetInfo = async (clientId: string) => {
    try {
      const response = await systemApi.getDatasetInfo(clientId)
      setDatasetInfo(response.data)
      setSelectedClient(clientId)
    } catch (error) {
      console.error('Failed to fetch dataset info:', error)
    }
  }

  const chartData = adapterData.map(item => ({
    name: item.client_id.replace('factory_', ''),
    'Before Adapter': item.before_adapter_accuracy !== null ? Number((item.before_adapter_accuracy * 100).toFixed(1)) : 0,
    'After Adapter': item.after_adapter_accuracy !== null ? Number((item.after_adapter_accuracy * 100).toFixed(1)) : 0,
    improvement: item.improvement !== null ? Number((item.improvement * 100).toFixed(1)) : 0,
    has_adapter: item.has_adapter
  }))

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  const hasAnyAdapter = adapterData.some(item => item.has_adapter)

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Clients & Personalization</h1>

      {hasAnyAdapter && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-semibold text-gray-800 flex items-center">
                <Zap className="w-5 h-5 mr-2 text-yellow-500" />
                Adapter Personalization Results
              </h2>
              <p className="text-sm text-gray-500 mt-1">
                Before/After accuracy comparison for each client's local adapter
              </p>
            </div>
            <div className="flex space-x-4">
              <div className="text-center px-4">
                <p className="text-xs text-gray-500">Avg Before</p>
                <p className="text-lg font-bold text-gray-500">
                  {avgBefore !== null ? `${(avgBefore * 100).toFixed(1)}%` : '-'}
                </p>
              </div>
              <div className="text-center px-4 border-l border-r">
                <p className="text-xs text-gray-500">Avg After</p>
                <p className="text-lg font-bold text-green-600">
                  {avgAfter !== null ? `${(avgAfter * 100).toFixed(1)}%` : '-'}
                </p>
              </div>
              <div className="text-center px-4">
                <p className="text-xs text-gray-500">Avg Gain</p>
                <p className="text-lg font-bold text-blue-600">
                  {avgImprovement !== null ? `+${(avgImprovement * 100).toFixed(1)}%` : '-'}
                </p>
              </div>
            </div>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} barGap={4}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                <Tooltip formatter={(value: number) => `${value}%`} />
                <Legend />
                <Bar dataKey="Before Adapter" fill="#94a3b8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="After Adapter" fill="#22c55e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
            {adapterData.map((item) => (
              <div
                key={item.client_id}
                className={`p-3 rounded-lg border-2 ${
                  item.has_adapter ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">
                    {item.client_id.replace('factory_', '')}
                  </span>
                  {item.improvement !== null && item.improvement > 0 && (
                    <span className="flex items-center text-xs font-bold text-green-600 bg-green-100 px-2 py-0.5 rounded-full">
                      <ArrowUpRight className="w-3 h-3 mr-0.5" />
                      +{(item.improvement * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                <div className="flex items-center text-xs">
                  <span className="text-gray-500 w-14">Before:</span>
                  <span className="font-semibold text-gray-500">
                    {item.before_adapter_accuracy !== null ? `${(item.before_adapter_accuracy * 100).toFixed(1)}%` : '-'}
                  </span>
                </div>
                <div className="flex items-center text-xs mt-1">
                  <span className="text-gray-500 w-14">After:</span>
                  <span className="font-semibold text-green-600">
                    {item.after_adapter_accuracy !== null ? `${(item.after_adapter_accuracy * 100).toFixed(1)}%` : '-'}
                  </span>
                </div>
                {!item.has_adapter && (
                  <p className="text-xs text-gray-400 mt-1">No adapter trained</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b">
            <h2 className="text-lg font-semibold text-gray-800">Connected Clients</h2>
            <p className="text-sm text-gray-500">Simulated factory devices</p>
          </div>
          <div className="p-6 space-y-4">
            {clients.map((client) => (
              <div
                key={client.client_id}
                onClick={() => fetchDatasetInfo(client.client_id)}
                className={`p-4 rounded-lg cursor-pointer transition-colors ${
                  selectedClient === client.client_id
                    ? 'bg-blue-50 border-2 border-blue-500'
                    : 'bg-gray-50 hover:bg-gray-100 border-2 border-transparent'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className={`w-12 h-12 rounded-lg flex items-center justify-center mr-4 ${
                      client.has_adapter ? 'bg-green-100' : 'bg-blue-100'
                    }`}>
                      <Server className={`w-6 h-6 ${client.has_adapter ? 'text-green-600' : 'text-blue-600'}`} />
                    </div>
                    <div>
                      <p className="font-semibold text-gray-800">{client.client_id}</p>
                      <p className="text-sm text-gray-500">Factory Device</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-500">Samples</p>
                    <p className="font-semibold text-gray-800">{client.data_size}</p>
                  </div>
                </div>
                <div className="mt-3 flex items-center justify-between text-sm">
                  <div className="flex items-center text-gray-500">
                    <HardDrive className="w-4 h-4 mr-1" />
                    {client.num_classes} fault types
                  </div>
                  <div className="flex items-center space-x-3">
                    {client.has_adapter && client.before_adapter_accuracy !== null && client.after_adapter_accuracy !== null && (
                      <span className="flex items-center text-green-600 text-xs">
                        <TrendingUp className="w-3 h-3 mr-1" />
                        {(client.before_adapter_accuracy * 100).toFixed(0)}% → {(client.after_adapter_accuracy * 100).toFixed(0)}%
                      </span>
                    )}
                    <span className={`flex items-center ${client.has_adapter ? 'text-green-600' : 'text-gray-400'}`}>
                      <div className={`w-2 h-2 rounded-full mr-2 ${client.has_adapter ? 'bg-green-500' : 'bg-gray-300'}`} />
                      {client.has_adapter ? 'Adapted' : 'Connected'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b">
            <h2 className="text-lg font-semibold text-gray-800">Dataset Details</h2>
            <p className="text-sm text-gray-500">
              {selectedClient ? selectedClient : 'Select a client to view details'}
            </p>
          </div>
          <div className="p-6">
            {datasetInfo ? (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500">Total Samples</p>
                    <p className="text-2xl font-bold text-gray-800">{datasetInfo.num_samples}</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500">Signal Length</p>
                    <p className="text-2xl font-bold text-gray-800">{datasetInfo.signal_length}</p>
                  </div>
                </div>

                {selectedClient && (() => {
                  const adapterInfo = adapterData.find(a => a.client_id === selectedClient)
                  return adapterInfo?.has_adapter && adapterInfo.before_adapter_accuracy !== null ? (
                    <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                      <h3 className="font-semibold text-green-800 mb-3 flex items-center">
                        <Zap className="w-4 h-4 mr-2" />
                        Local Adapter Performance
                      </h3>
                      <div className="grid grid-cols-3 gap-3">
                        <div className="text-center">
                          <p className="text-xs text-gray-500">Before</p>
                          <p className="text-xl font-bold text-gray-600">
                            {(adapterInfo.before_adapter_accuracy * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-gray-500">After</p>
                          <p className="text-xl font-bold text-green-600">
                            {(adapterInfo.after_adapter_accuracy! * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-gray-500">Gain</p>
                          <p className="text-xl font-bold text-blue-600">
                            +{(adapterInfo.improvement! * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <p className="text-xs text-green-600 mt-2">
                        Adapter fine-tunes the global model for local working conditions.
                        Parameters are kept private and not shared in federated aggregation.
                      </p>
                    </div>
                  ) : null
                })()}

                <div>
                  <div className="flex items-center mb-4">
                    <BarChart3 className="w-5 h-5 mr-2 text-gray-600" />
                    <h3 className="font-semibold text-gray-800">Class Distribution</h3>
                  </div>
                  <div className="space-y-3">
                    {Object.entries(datasetInfo.class_distribution).map(([cls, count]: [string, any]) => {
                      const faultNames = ['Normal', 'Bearing', 'Gear', 'Unbalance', 'Misalignment']
                      const percentage = (count / datasetInfo.num_samples) * 100
                      return (
                        <div key={cls}>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-gray-600">{faultNames[parseInt(cls)] || cls}</span>
                            <span className="font-medium">{count} ({percentage.toFixed(1)}%)</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all"
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-blue-800 mb-2">Working Condition</h3>
                  <p className="text-sm text-blue-600">
                    Each client simulates different factory conditions with variations in:
                  </p>
                  <ul className="list-disc list-inside text-sm text-blue-600 mt-2">
                    <li>Rotational speed</li>
                    <li>Load conditions</li>
                    <li>Noise levels</li>
                  </ul>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Server className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                <p>Select a client from the list</p>
                <p className="text-sm">to view dataset details</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
