export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const SUPPORTED_PLATFORMS = {
  instagram: {
    name: 'Instagram',
    patterns: ['instagram.com/p/', 'instagram.com/reel/'],
    icon: '📸'
  },
  twitter: {
    name: 'Twitter/X',
    patterns: ['twitter.com/', 'x.com/'],
    icon: '🐦'
  },
  tiktok: {
    name: 'TikTok',
    patterns: ['tiktok.com/'],
    icon: '🎵'
  },
  facebook: {
    name: 'Facebook',
    patterns: ['facebook.com/', 'fb.com/'],
    icon: '👥'
  },
  youtube: {
    name: 'YouTube',
    patterns: ['youtube.com/watch', 'youtu.be/'],
    icon: '▶️'
  }
};

export const VERDICT_COLORS = {
  'AI-Generated': {
    bg: 'bg-red-500',
    text: 'text-red-400',
    border: 'border-red-500'
  },
  'Real': {
    bg: 'bg-green-500',
    text: 'text-green-400',
    border: 'border-green-500'
  },
  'Uncertain': {
    bg: 'bg-yellow-500',
    text: 'text-yellow-400',
    border: 'border-yellow-500'
  }
};