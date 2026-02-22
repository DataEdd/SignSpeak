import React, { useState, useCallback } from 'react'
import TextInput from './components/TextInput'
import VideoPanel from './components/VideoPanel'
import { translateText } from './api/signbridge'
import './App.css'

function App() {
  const [videoUrl, setVideoUrl] = useState(null)
  const [currentText, setCurrentText] = useState('')
  const [isTranslating, setIsTranslating] = useState(false)
  const [error, setError] = useState(null)
  const [glosses, setGlosses] = useState([])
  const [confidence, setConfidence] = useState(null)
  const [isDemo, setIsDemo] = useState(false)

  const handleTranslate = useCallback(async (text) => {
    if (!text.trim()) return

    setIsTranslating(true)
    setError(null)
    setCurrentText(text)

    try {
      const result = await translateText(text)
      setVideoUrl(result.videoUrl)
      setGlosses(result.glosses || [])
      setConfidence(result.confidence)
      setIsDemo(!!result.isDemo)
    } catch (err) {
      setError(err.message || 'Translation failed')
      setVideoUrl(null)
      setGlosses([])
      setConfidence(null)
      setIsDemo(false)
    } finally {
      setIsTranslating(false)
    }
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1 className="logo">
            <span className="logo-icon">&#x1F91F;</span>
            SignSpeak
          </h1>
          <p className="tagline">Text to Sign Language Video Translation</p>
        </div>
      </header>

      <main className="app-main">
        <div className="content-area">
          <div className="text-section">
            <TextInput
              onTranslate={handleTranslate}
              isTranslating={isTranslating}
            />
            {currentText && (
              <div className="translation-info">
                <h3>Input Text:</h3>
                <p className="input-text">{currentText}</p>
                {glosses.length > 0 && (
                  <>
                    <h3>ASL Glosses:</h3>
                    <div className="gloss-list">
                      {glosses.map((gloss, index) => (
                        <span key={index} className="gloss-tag">
                          {gloss}
                        </span>
                      ))}
                    </div>
                  </>
                )}
                {confidence !== null && (
                  <p className="confidence">
                    Confidence: {(confidence * 100).toFixed(0)}%
                  </p>
                )}
              </div>
            )}
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        <div className="video-area">
          <VideoPanel
            videoUrl={videoUrl}
            isTranslating={isTranslating}
            isDemo={isDemo}
            glosses={glosses}
            confidence={confidence}
          />
        </div>
      </main>

      <footer className="app-footer">
        <p>SignSpeak - Making communication accessible through sign language</p>
      </footer>
    </div>
  )
}

export default App
