import axios from 'axios'
import { mockTranslate } from './mockTranslate'
import { SHOWCASE } from '../data/examples'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' }
})

function showcaseUrl() {
  return `${import.meta.env.BASE_URL}${SHOWCASE.videoPath}`
}

/** Resolve the showcase — a single pre-rendered 3D avatar clip. */
export function getShowcase() {
  return {
    videoUrl: showcaseUrl(),
    glosses: SHOWCASE.glosses,
    label: SHOWCASE.label,
    description: SHOWCASE.description,
    isShowcase: true
  }
}

/**
 * Translate arbitrary English text to ASL glosses. If a backend is reachable
 * it's used; otherwise the client-side grammar engine runs. There is no
 * avatar for arbitrary text — the UI shows the gloss sequence.
 */
export async function translateText(text) {
  try {
    const response = await api.post('/translate', { text })
    return {
      videoUrl: response.data.video_url || null,
      glosses: response.data.glosses || [],
      confidence: response.data.confidence,
      isShowcase: false
    }
  } catch (error) {
    const data = error.response?.data
    if (error.response && typeof data === 'object' && data?.message) {
      throw new Error(data.message)
    }
    const result = mockTranslate(text)
    return { videoUrl: null, isShowcase: false, ...result }
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
