import React, { useState, useRef, useEffect } from 'react'
import './VideoPlayer.css'

// Demo transcript with timing (simulating real news)
const DEMO_TRANSCRIPT = [
  { start: 0, end: 3, text: "Welcome to the news." },
  { start: 3, end: 7, text: "Today we have important updates." },
  { start: 7, end: 11, text: "The government announced new policies." },
  { start: 11, end: 15, text: "This will help many people across the country." },
  { start: 15, end: 19, text: "Schools will now have sign language classes." },
  { start: 19, end: 23, text: "This is good news for the deaf community." },
  { start: 23, end: 27, text: "Thank you for watching." }
]

function VideoPlayer({ onTranscriptUpdate }) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(27)
  const [currentSegment, setCurrentSegment] = useState(null)
  const timerRef = useRef(null)

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setCurrentTime(prev => {
          if (prev >= duration) {
            setIsPlaying(false)
            return 0
          }
          return prev + 0.1
        })
      }, 100)
    } else {
      clearInterval(timerRef.current)
    }

    return () => clearInterval(timerRef.current)
  }, [isPlaying, duration])

  useEffect(() => {
    const segment = DEMO_TRANSCRIPT.find(
      seg => currentTime >= seg.start && currentTime < seg.end
    )

    if (segment && segment !== currentSegment) {
      setCurrentSegment(segment)
      onTranscriptUpdate(segment.text)
    }
  }, [currentTime, currentSegment, onTranscriptUpdate])

  const togglePlay = () => {
    setIsPlaying(!isPlaying)
  }

  const handleSeek = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const percent = (e.clientX - rect.left) / rect.width
    setCurrentTime(percent * duration)
  }

  const formatTime = (time) => {
    const mins = Math.floor(time / 60)
    const secs = Math.floor(time % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const restart = () => {
    setCurrentTime(0)
    setIsPlaying(true)
    setCurrentSegment(null)
  }

  return (
    <div className="video-player">
      <div className="video-display">
        <div className="video-placeholder">
          <div className="news-overlay">
            <div className="news-logo">NEWS LIVE</div>
            <div className="news-ticker">
              Breaking: Sign Language Support Now Available
            </div>
          </div>
          {currentSegment && (
            <div className="caption-overlay">
              {currentSegment.text}
            </div>
          )}
        </div>
      </div>

      <div className="video-controls">
        <button className="play-btn" onClick={togglePlay}>
          {isPlaying ? '\u23F8' : '\u25B6'}
        </button>

        <button className="restart-btn" onClick={restart}>
          \u21BB
        </button>

        <div className="progress-container" onClick={handleSeek}>
          <div
            className="progress-bar"
            style={{ width: `${(currentTime / duration) * 100}%` }}
          />
          {DEMO_TRANSCRIPT.map((seg, index) => (
            <div
              key={index}
              className="segment-marker"
              style={{ left: `${(seg.start / duration) * 100}%` }}
            />
          ))}
        </div>

        <span className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>

      <div className="transcript-segments">
        {DEMO_TRANSCRIPT.map((seg, index) => (
          <div
            key={index}
            className={`segment ${currentSegment === seg ? 'active' : ''} ${
              currentTime >= seg.end ? 'completed' : ''
            }`}
            onClick={() => setCurrentTime(seg.start)}
          >
            <span className="segment-time">{formatTime(seg.start)}</span>
            <span className="segment-text">{seg.text}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default VideoPlayer
