import React from 'react'
import './Transcript.css'

function Transcript({ currentText, glosses }) {
  return (
    <div className="transcript">
      <div className="transcript-header">
        <h2>Live Transcript</h2>
        <span className="live-badge">LIVE</span>
      </div>

      <div className="transcript-content">
        {currentText ? (
          <>
            <div className="current-text">
              <p>{currentText}</p>
            </div>

            {glosses.length > 0 && (
              <div className="gloss-display">
                <h3>ASL Translation</h3>
                <div className="gloss-flow">
                  {glosses.map((gloss, index) => (
                    <span key={index} className="gloss-item">
                      {gloss}
                      {index < glosses.length - 1 && (
                        <span className="gloss-arrow">&rarr;</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="transcript-placeholder">
            <p>Waiting for transcript...</p>
            <p className="hint">Play the demo video to see live transcription</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default Transcript
