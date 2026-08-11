import React from 'react';
import { motion } from 'framer-motion';
import { FiCheckCircle, FiAlertTriangle, FiClock, FiLink, FiBarChart2 } from 'react-icons/fi';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

const ResultCard = ({ result }) => {
  const {
    verdict,
    confidence_score,
    media_type,
    platform,
    processing_time,
    original_url
  } = result;

  const isAI = verdict === 'AI-Generated';
  const confidencePercent = Math.round(confidence_score * 100);

    return (
    <div className="result-card">
      <div className="result-header">
        <div className={`result-icon ${isAI ? 'ai' : 'real'}`}>
          {isAI ? '⚠' : '✓'}
        </div>
        <div>
          <div className="result-title" style={{ color: isAI ? '#dc2626' : '#16a34a' }}>
            {isAI ? 'AI-Generated' : 'Likely Real'}
          </div>
          <div className="result-subtitle">
            {isAI ? 'This image appears to be artificially generated' : 'This image appears to be authentic'}
          </div>
        </div>
      </div>

      <div className="confidence-section">
        <div className="confidence-bar">
          <div className={`confidence-fill ${isAI ? 'ai' : 'real'}`} style={{ width: `${confidencePercent}%` }}></div>
        </div>
        <div className="confidence-text">{confidencePercent}%</div>
      </div>

      <div className="detail-grid">
        <div className="detail-item">
          Platform
          <span>{platform || 'Unknown'}</span>
        </div>
        <div className="detail-item">
          Media Type
          <span>{media_type || 'Image'}</span>
        </div>
        <div className="detail-item">
          Processing Time
          <span>{processing_time ? `${processing_time.toFixed(2)}s` : 'N/A'}</span>
        </div>
        <div className="detail-item">
          Source
          <span style={{ fontSize: '11px', wordBreak: 'break-all' }}>
            <a href={original_url} target="_blank" rel="noopener noreferrer" style={{ color: '#2563eb' }}>View Post →</a>
          </span>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;