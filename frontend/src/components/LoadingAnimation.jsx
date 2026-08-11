import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { FiSearch, FiImage, FiCpu } from 'react-icons/fi';

const LoadingAnimation = () => {
  const [step, setStep] = useState(0);
  
  const steps = [
    { icon: FiSearch, text: 'Extracting media from URL...', color: 'text-blue-400' },
    { icon: FiImage, text: 'Downloading content...', color: 'text-purple-400' },
    { icon: FiCpu, text: 'Running AI detection...', color: 'text-pink-400' },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((prev) => (prev + 1) % steps.length);
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-card p-8"
    >
      <div className="flex flex-col items-center justify-center space-y-6">
        {/* Animated ring */}
        <div className="relative w-32 h-32">
          <div className="absolute inset-0 rounded-full border-4 border-purple-500 border-opacity-30"></div>
          <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-purple-500 animate-spin"></div>
          
          <div className="absolute inset-0 flex items-center justify-center">
            {React.createElement(steps[step].icon, {
              className: `text-4xl ${steps[step].color}`
            })}
          </div>
        </div>
        
        {/* Step indicator */}
        <div className="text-center">
          <p className="text-xl text-white font-semibold mb-2">
            {steps[step].text}
          </p>
          <div className="flex space-x-2 justify-center">
            {steps.map((_, index) => (
              <div
                key={index}
                className={`w-2 h-2 rounded-full transition-all duration-300 ${
                  index === step ? 'bg-purple-400 scale-125' : 'bg-gray-600'
                }`}
              />
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default LoadingAnimation;