// Preloaded showcase clip. The live demo is primarily the NLP grammar engine;
// this single entry is the "what the full system produces" reference video.
//
// Source: hackathon-v1/output/avatar_with_captions_synced.mp4 — a 12.8s
// SMPL-X 3D avatar signing the NWS winter-storm news broadcast with
// synchronised word-level captions.

export const SHOWCASE = {
  id: 'news-broadcast',
  label: 'Play 3D avatar showcase',
  description: 'Weather news broadcast signed by our SMPL-X 3D avatar (~13s)',
  videoPath: 'videos/showcase.mp4',
  glosses: [
    'THIS', 'BIG', 'ONE', 'WE', 'TAKE', 'SERIOUS', 'YES',
    'ABSOLUTE', 'THIS', 'BIG', 'ONE', 'MEAN', 'THINK',
    'WE', 'HAVE', 'WATCH', 'WARNING', 'NATIONAL', 'WEATHER', 'SERVICE',
    'STRETCH', 'TWO', 'THOUSAND', 'MILE', 'FROM', 'NEW', 'MEXICO',
    'ALL', 'WAY', 'NEW', 'ENGLAND', 'THIS', 'BIG', 'STORM'
  ]
}
