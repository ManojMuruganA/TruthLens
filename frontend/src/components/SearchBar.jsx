import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { FiSearch, FiLink, FiLoader } from 'react-icons/fi';
import { analyzeUrl } from '../utils/api';
import { API_BASE_URL } from '../utils/constants';
import toast from 'react-hot-toast';

const SearchBar = ({ onStart, onComplete, onError }) => {
  const [url, setUrl] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validateUrl = (url) => {
    const socialMediaPatterns = [
      /instagram\.com\/p\//,
      /instagram\.com\/reel\//,
      /twitter\.com\/.*\/status\//,
      /x\.com\/.*\/status\//,
      /facebook\.com\/.*\/posts\//,
      /tiktok\.com\/@.*\/video\//,
      /youtube\.com\/watch/,
      /youtu\.be\//,
    ];

    return socialMediaPatterns.some(pattern => pattern.test(url));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!url.trim()) {
      toast.error('Please enter a URL');
      return;
    }
    
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      toast.error('Please enter a valid URL starting with http:// or https://');
      return;
    }

    if (!validateUrl(url)) {
      toast.error('Please enter a supported social media URL', {
        duration: 5000,
      });
      return;
    }

    setIsSubmitting(true);
    onStart();

    try {
      const { task_id } = await analyzeUrl(url);
      
      const pollInterval = 2000;
      const maxAttempts = 60;
      let attempts = 0;

      const checkResult = async () => {
        try {
          const response = await fetch(`${API_BASE_URL}/api/result/${task_id}`);
          const data = await response.json();

          if (data.status === 'SUCCESS') {
            onComplete(data.result);
            setIsSubmitting(false);
          } else if (data.status === 'FAILED') {
            throw new Error(data.error || 'Analysis failed');
          } else if (data.status === 'PENDING' || data.status === 'PROCESSING') {
            attempts++;
            
            if (attempts < maxAttempts) {
              setTimeout(checkResult, pollInterval);
            } else {
              throw new Error('Analysis timed out. Please try again.');
            }
          }
        } catch (error) {
          onError(error.message);
          setIsSubmitting(false);
        }
      };

      setTimeout(checkResult, pollInterval);
      
    } catch (error) {
      onError(error.message);
      setIsSubmitting(false);
    }
  };

  const handlePaste = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text && text.startsWith('http')) {
        setUrl(text);
        toast.success('URL pasted from clipboard!');
      }
    } catch (err) {
      toast.error('Failed to paste from clipboard');
    }
  }, []);

    return (
    <form onSubmit={handleSubmit}>
      <div className="input-wrapper">
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Paste Instagram, Twitter/X, or TikTok URL..."
          disabled={isSubmitting}
        />
        <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
          {isSubmitting ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>
      <p style={{ fontSize: '12px', color: '#9ca3af', marginTop: '12px', textAlign: 'center' }}>
        Supports Instagram, Twitter/X, TikTok, and Facebook
      </p>
    </form>
  );
};

export default SearchBar;