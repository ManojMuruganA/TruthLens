import React, { useState } from 'react';
import SearchBar from './components/SearchBar';
import ResultCard from './components/ResultCard';
import HistoryPanel from './components/HistoryPanel';
import LoadingAnimation from './components/LoadingAnimation';
import HowItWorks from './components/HowItWorks';
import { motion, AnimatePresence } from 'framer-motion';
import { FiShield, FiActivity } from 'react-icons/fi';
import toast from 'react-hot-toast';

function App() {
  const [currentResult, setCurrentResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('detect');

  const handleDetectionStart = () => {
    setIsLoading(true);
    setCurrentResult(null);
  };

  const handleDetectionComplete = (result) => {
    setCurrentResult(result);
    setIsLoading(false);
    toast.success('Analysis complete!', {
      icon: '🎯',
    });
  };

  const handleDetectionError = (error) => {
    setIsLoading(false);
    toast.error(error || 'Analysis failed. Please try again.');
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f8f9fa' }}>
      <div className="container">
        
        {/* Header */}
        <div className="header">
          <h1>TruthLens</h1>
          <p>AI-Generated Image Detection for Social Media</p>
        </div>

        {/* Tabs */}
        <div className="tabs">
          <button className={`tab ${activeTab === 'detect' ? 'active' : ''}`} onClick={() => setActiveTab('detect')}>Detect</button>
          <button className={`tab ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>History</button>
        </div>

        {/* Content */}
        {activeTab === 'detect' && (
          <div>
            <div className="card">
              <SearchBar 
                onStart={handleDetectionStart}
                onComplete={handleDetectionComplete}
                onError={handleDetectionError}
              />
            </div>

            {isLoading && (
              <div className="card" style={{ textAlign: 'center' }}>
                <div className="loading-spinner"></div>
                <p className="loading-text">Analyzing image...</p>
              </div>
            )}

            {currentResult && !isLoading && (
              <ResultCard result={currentResult} />
            )}
          </div>
        )}

        {activeTab === 'history' && <HistoryPanel />}

        {/* Footer */}
        <div className="footer">
          JBNU · Engineering Capstone Design · 2026
        </div>
      </div>
    </div>
  );
}

export default App;