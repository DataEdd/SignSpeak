import React, { useState, useCallback } from 'react'
import TextInput from './components/TextInput'
import GlossPanel from './components/GlossPanel'
import VideoPanel from './components/VideoPanel'
import { translateText, getShowcase } from './api/signbridge'
import './App.css'

function App() {
  const [inputText, setInputText] = useState('')
  const [glosses, setGlosses] = useState([])
  const [confidence, setConfidence] = useState(null)
  const [showcaseVideoUrl, setShowcaseVideoUrl] = useState(null)
  const [isTranslating, setIsTranslating] = useState(false)
  const [error, setError] = useState(null)

  const handleTranslate = useCallback(async (text) => {
    if (!text.trim()) return
    setIsTranslating(true)
    setError(null)
    setInputText(text)
    try {
      const result = await translateText(text)
      setGlosses(result.glosses || [])
      setConfidence(result.confidence ?? null)
    } catch (err) {
      setError(err.message || 'Translation failed')
      setGlosses([])
      setConfidence(null)
    } finally {
      setIsTranslating(false)
    }
  }, [])

  // Playing the showcase doesn't touch the grammar panel — that view is
  // reserved for typed input. Just surface the pre-rendered video.
  const handleShowcase = useCallback(() => {
    setError(null)
    setShowcaseVideoUrl(getShowcase().videoUrl)
  }, [])

  const handleCloseShowcase = useCallback(() => {
    setShowcaseVideoUrl(null)
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1 className="logo">
            <span className="logo-icon">&#x1F91F;</span>
            SignSpeak
          </h1>
          <p className="tagline">English → ASL grammar conversion + 3D avatar showcase</p>
        </div>
      </header>

      <main className="app-main">
        <div className="top-row">
          <div className="text-section">
            <TextInput
              onTranslate={handleTranslate}
              onShowcase={handleShowcase}
              isTranslating={isTranslating}
            />
            {error && <div className="error-message">{error}</div>}
          </div>

          <div className="gloss-section">
            <GlossPanel
              inputText={inputText}
              glosses={glosses}
              confidence={confidence}
              isTranslating={isTranslating}
            />
          </div>
        </div>

        {showcaseVideoUrl && (
          <div className="showcase-row">
            <VideoPanel
              videoUrl={showcaseVideoUrl}
              isShowcase={true}
              onClose={handleCloseShowcase}
            />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>SignSpeak — Making communication accessible through sign language</p>
      </footer>
    </div>
  )
}

export default App
