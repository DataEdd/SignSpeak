import axios from 'axios'
import { mockTranslate } from './mockTranslate'

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
      isDemo: false,
    }
  } catch (error) {
    // Only throw if we got a real JSON error from our backend
    const data = error.response?.data
    if (error.response && typeof data === 'object' && data?.message) {
      throw new Error(data.message)
    }
    // Backend unreachable or not our API (e.g. GitHub Pages 404) — fall back to mock
    const result = mockTranslate(text)
    return { videoUrl: null, ...result }
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
