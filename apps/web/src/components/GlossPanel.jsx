import React from 'react'
import GlossAnimation from './GlossAnimation'
import './GlossPanel.css'

function GlossPanel({ inputText, glosses, confidence, isTranslating }) {
  const empty = !inputText && glosses.length === 0

  return (
    <div className="gloss-panel">
      <div className="gloss-panel-header">
        <h2>NLP Grammar Engine</h2>
        <span className="gloss-panel-subtitle">English → ASL gloss order</span>
      </div>

      {empty && !isTranslating && (
        <div className="gloss-panel-empty">
          Type a sentence and press <strong>Translate</strong> to see ASL gloss order.
        </div>
      )}

      {isTranslating && (
        <div className="gloss-panel-empty">Translating…</div>
      )}

      {!empty && (
        <>
          <div className="gloss-panel-input">
            <span className="gloss-panel-label">Input:</span>
            <p>{inputText}</p>
          </div>
          <GlossAnimation glosses={glosses} confidence={confidence} />
        </>
      )}
    </div>
  )
}

export default GlossPanel
