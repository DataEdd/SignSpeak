import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json'
  }
})

export async function translateText(text) {
  try {
    const response = await api.post('/translate', { text })
    return {
      videoUrl: response.data.video_url,
      glosses: response.data.glosses || [],
      confidence: response.data.confidence,
    }
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.message || 'Translation failed')
    }
    throw new Error('Network error - please check if the backend is running')
  }
}

export async function checkHealth() {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    throw new Error('Backend not available')
  }
}

export default api
