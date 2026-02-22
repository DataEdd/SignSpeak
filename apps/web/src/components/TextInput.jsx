import React, { useState } from 'react'
import './TextInput.css'

const SAMPLE_PHRASES = [
  "Hello, how are you?",
  "Thank you very much",
  "What is your name?",
  "I want to learn sign language",
  "Good morning, nice to meet you",
  "Where is the school?"
]

function TextInput({ onTranslate, isTranslating }) {
  const [text, setText] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (text.trim() && !isTranslating) {
      onTranslate(text.trim())
    }
  }

  const handleSampleClick = (phrase) => {
    setText(phrase)
    onTranslate(phrase)
  }

  return (
    <div className="text-input">
      <div className="input-header">
        <h2>Text to ASL Translation</h2>
        <p>Enter any English text to see it translated to American Sign Language</p>
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <div className="textarea-wrapper">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type your text here..."
            rows={4}
            disabled={isTranslating}
          />
          <span className="char-count">{text.length} characters</span>
        </div>

        <button
          type="submit"
          className="translate-btn"
          disabled={!text.trim() || isTranslating}
        >
          {isTranslating ? (
            <>
              <span className="spinner"></span>
              Translating...
            </>
          ) : (
            'Translate to ASL'
          )}
        </button>
      </form>

      <div className="sample-phrases">
        <h3>Try these examples:</h3>
        <div className="phrase-list">
          {SAMPLE_PHRASES.map((phrase, index) => (
            <button
              key={index}
              className="phrase-btn"
              onClick={() => handleSampleClick(phrase)}
              disabled={isTranslating}
            >
              {phrase}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default TextInput
