import { useState, useRef, useCallback } from 'react'

/**
 * Records a <canvas> element to a video blob via MediaRecorder + captureStream.
 * Returns a blob URL playable in a <video> element.
 */
export function useCanvasRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [videoUrl, setVideoUrl] = useState(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const urlRef = useRef(null)

  const startRecording = useCallback((canvas) => {
    // Clean up previous blob
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current)
      urlRef.current = null
      setVideoUrl(null)
    }

    if (!canvas?.captureStream || typeof MediaRecorder === 'undefined') {
      return false
    }

    try {
      const stream = canvas.captureStream(30)
      const types = [
        'video/webm;codecs=vp9',
        'video/webm;codecs=vp8',
        'video/webm',
        'video/mp4',
      ]
      const mimeType = types.find(t => {
        try { return MediaRecorder.isTypeSupported(t) } catch { return false }
      })
      if (!mimeType) return false

      const recorder = new MediaRecorder(stream, { mimeType })
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        if (chunksRef.current.length > 0) {
          const blob = new Blob(chunksRef.current, { type: mimeType })
          const url = URL.createObjectURL(blob)
          urlRef.current = url
          setVideoUrl(url)
        }
        setIsRecording(false)
      }

      recorderRef.current = recorder
      recorder.start(100)
      setIsRecording(true)
      return true
    } catch (err) {
      console.error('Canvas recording failed:', err)
      return false
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop()
    }
  }, [])

  const clearRecording = useCallback(() => {
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current)
      urlRef.current = null
    }
    setVideoUrl(null)
  }, [])

  return { isRecording, videoUrl, startRecording, stopRecording, clearRecording }
}
