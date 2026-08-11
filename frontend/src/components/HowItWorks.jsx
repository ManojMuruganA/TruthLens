import React from 'react';
import { motion } from 'framer-motion';
import { FiLink, FiDownload, FiCpu, FiBarChart2 } from 'react-icons/fi';

const steps = [
  {
    icon: FiLink,
    title: 'Paste URL',
    description: 'Simply paste any social media post URL from Instagram, Twitter, TikTok, or Facebook.',
    color: 'text-blue-400'
  },
  {
    icon: FiDownload,
    title: 'Extract Media',
    description: 'Our system automatically extracts the image or video from the post without downloading the entire page.',
    color: 'text-purple-400'
  },
  {
    icon: FiCpu,
    title: 'AI Analysis',
    description: 'Advanced deep learning models analyze the content for AI generation artifacts and patterns.',
    color: 'text-pink-400'
  },
  {
    icon: FiBarChart2,
    title: 'Get Results',
    description: 'Receive a clear verdict with confidence score indicating if the content is AI-generated or authentic.',
    color: 'text-green-400'
  }
];

const HowItWorks = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-center mb-12"
      >
        <h2 className="text-3xl font-bold text-white mb-4">How TruthLens Works</h2>
        <p className="text-purple-200">
          Our advanced AI detection pipeline in four simple steps
        </p>
      </motion.div>

      <div className="grid md:grid-cols-2 gap-6">
        {steps.map((step, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.2 }}
            className="glass-card p-6"
          >
            <div className="flex items-start space-x-4">
              <div className={`${step.color} text-3xl`}>
                <step.icon />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  {step.title}
                </h3>
                <p className="text-purple-200">
                  {step.description}
                </p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="glass-card p-6 mt-8"
      >
        <h3 className="text-xl font-semibold text-white mb-4">Our Technology</h3>
        <div className="space-y-3 text-purple-200">
          <p>• <strong className="text-white">Ensemble Learning:</strong> Multiple AI models work together for higher accuracy</p>
          <p>• <strong className="text-white">Deep Neural Networks:</strong> State-of-the-art computer vision models</p>
          <p>• <strong className="text-white">Real-time Processing:</strong> Results in seconds, not minutes</p>
          <p>• <strong className="text-white">Multi-platform Support:</strong> Works across all major social media platforms</p>
        </div>
      </motion.div>
    </div>
  );
};

export default HowItWorks;