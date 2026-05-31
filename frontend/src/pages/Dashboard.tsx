import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { systemApi, SystemStatus } from '../services/api'
import { Server, Cpu, Database, Activity } from 'lucide-react'

export default function Dashboard() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 3000)
    return () => clearInterval(interval)
  }, [])

  const fetchStatus = async () => {
    try {
      const response = await systemApi.getStatus()
      setStatus(response.data)
    } catch (error) {
      console.error('Failed to fetch status:', error)
    } finally {
      setLoading(false)
    }
  }

  const mockChartData = [
    { round: 1, accuracy: 0.65 },
    { round: 2, accuracy: 0.72 },
    { round: 3, accuracy: 0.78 },
    { round: 4, accuracy: 0.82 },
    { round: 5, accuracy: 0.85 },
    { round: 6, accuracy: 0.88 },
    { round: 7, accuracy: 0.90 },
    { round: 8, accuracy: 0.91 },
    { round: 9, accuracy: 0.93 },
    { round: 10, accuracy: 0.94 },
  ]

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500 text-sm">Active Clients</p>
              <p className="text-3xl font-bold text-gray-800">{status?.num_clients || 4}</p>
            </div>
            <div className="bg-blue-100 p-3 rounded-full">
              <Server className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500 text-sm">Current Round</p>
              <p className="text-3xl font-bold text-gray-800">{status?.current_round || 0}</p>
            </div>
            <div className="bg-green-100 p-3 rounded-full">
              <Cpu className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500 text-sm">Training Status</p>
              <p className={`text-xl font-bold ${status?.is_training ? 'text-yellow-600' : 'text-green-600'}`}>
                {status?.is_training ? 'Training...' : 'Ready'}
              </p>
            </div>
            <div className={`p-3 rounded-full ${status?.is_training ? 'bg-yellow-100' : 'bg-green-100'}`}>
              <Activity className={`w-6 h-6 ${status?.is_training ? 'text-yellow-600' : 'text-green-600'}`} />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-500 text-sm">Model Version</p>
              <p className="text-lg font-bold text-gray-800 truncate max-w-32">
                {status?.latest_model_version || 'None'}
              </p>
            </div>
            <div className="bg-purple-100 p-3 rounded-full">
              <Database className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Global Model Accuracy</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Active Clients</h2>
          <div className="space-y-4">
            {status?.active_clients.map((client) => (
              <div
                key={client.client_id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center mr-4">
                    <Server className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-800">{client.client_id}</p>
                    <p className="text-sm text-gray-500">{client.data_size} samples</p>
                  </div>
                </div>
                <span className={`px-3 py-1 text-sm rounded-full ${
                  client.status === 'registered'
                    ? 'bg-green-100 text-green-700'
                    : 'bg-gray-100 text-gray-700'
                }`}>
                  {client.status}
                </span>
              </div>
            ))}
            {(!status?.active_clients || status.active_clients.length === 0) && (
              <div className="text-center py-8 text-gray-500">
                No active clients. Clients will appear after training starts.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
