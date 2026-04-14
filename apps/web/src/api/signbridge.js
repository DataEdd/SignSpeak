import axios from 'axios'
import { mockTranslate } from './mockTranslate'
import { findExample } from '../data/examples'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' }
})

function bundledVideoUrl(slug) {
  // Vite resolves import.meta.env.BASE_URL to '/' in dev and '/SignSpeak/' in prod
  return `${import.meta.env.BASE_URL}videos/examples/${slug}.mp4`
}

export async function translateText(text) {
  // Preloaded-phrase path: ship a pre-rendered video, skip backend + CWASA entirely
  const example = findExample(text)
  if (example) {
    return {
      videoUrl: bundledVideoUrl(example.slug),
      glosses: example.glosses,
      confidence: 1.0,
      sigml: null,
      isDemo: false,
      isPreloaded: true
    }
  }

  try {
    const response = await api.post('/translate', { text })
    return {
      videoUrl: response.data.video_url,
      glosses: response.data.glosses || [],
      confidence: response.data.confidence,
      sigml: response.data.sigml || null,
      isDemo: false,
      isPreloaded: false
    }
  } catch (error) {
    const data = error.response?.data
    if (error.response && typeof data === 'object' && data?.message) {
      throw new Error(data.message)
    }
    // Backend unreachable (e.g. on GitHub Pages): fall back to client-side NLP
    const result = mockTranslate(text)
    return {
      videoUrl: null,
      sigml: result.sigml || null,
      isPreloaded: false,
      ...result
    }
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
