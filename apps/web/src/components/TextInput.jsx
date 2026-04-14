import React, { useState } from 'react'
import { SHOWCASE } from '../data/examples'
import './TextInput.css'

function TextInput({ onTranslate, onShowcase, isTranslating }) {
  const [text, setText] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (text.trim() && !isTranslating) {
      onTranslate(text.trim())
    }
  }

  return (
    <div className="text-input">
      <div className="input-header">
        <h2>Text to ASL Translation</h2>
        <p>Enter any English text to see it converted to ASL gloss order by the NLP engine.</p>
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
        <h3>3D avatar showcase</h3>
        <p className="showcase-hint">{SHOWCASE.description}</p>
        <button
          type="button"
          className="phrase-btn showcase-btn"
          onClick={onShowcase}
          disabled={isTranslating}
        >
          ▶ {SHOWCASE.label}
        </button>
      </div>
    </div>
  )
}

export default TextInput
