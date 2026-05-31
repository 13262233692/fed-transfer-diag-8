import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Clients from './pages/Clients'
import Models from './pages/Models'
import Training from './pages/Training'
import Prediction from './pages/Prediction'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/clients" element={<Clients />} />
        <Route path="/training" element={<Training />} />
        <Route path="/models" element={<Models />} />
        <Route path="/prediction" element={<Prediction />} />
      </Routes>
    </Layout>
  )
}

export default App
