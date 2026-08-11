import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { FiClock, FiAlertTriangle, FiCheckCircle } from 'react-icons/fi';
import axios from 'axios';
import { API_BASE_URL } from '../utils/constants';

const HistoryPanel = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/history?limit=20`);
      setHistory(response.data.results || []);
    } catch (error) {
      console.error('Failed to fetch history:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center text-white py-12">
        <div className="animate-spin-slow w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full mx-auto mb-4"></div>
        <p>Loading history...</p>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="text-center text-purple-200 py-12">
        <FiClock className="text-6xl mx-auto mb-4 opacity-50" />
        <p className="text-xl">No detection history yet</p>
        <p className="mt-2">Analyze some content to build your history</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="space-y-4">
        {history.map((item, index) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glass-card p-4 hover:bg-white hover:bg-opacity-15 transition-all duration-300 cursor-pointer"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                {item.verdict === 'AI-Generated' ? (
                  <FiAlertTriangle className="text-red-400 text-2xl" />
                ) : (
                  <FiCheckCircle className="text-green-400 text-2xl" />
                )}
                <div>
                  <p className="text-white font-semibold">
                    {item.verdict}
                    <span className="ml-2 text-sm text-purple-300">
                      ({Math.round(item.confidence_score * 100)}% confidence)
                    </span>
                  </p>
                  <p className="text-purple-200 text-sm truncate max-w-md">
                    {item.original_url}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-purple-300 text-sm capitalize">{item.platform}</p>
                <p className="text-gray-400 text-xs">
                  {new Date(item.created_at).toLocaleDateString()}
                </p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default HistoryPanel;