import { useState } from 'react';
import { motion } from 'framer-motion';

const AnimatedFeatureCard = ({ 
  icon, 
  title, 
  description, 
  details,
  index = 0,
  className = "" 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <motion.div
      className={`relative group cursor-pointer ${className}`}
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      whileHover={{ y: -10 }}
      onClick={() => setIsExpanded(!isExpanded)}
    >
      {/* Background glow effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-red-500/10 rounded-xl blur-xl group-hover:blur-2xl transition-all duration-300 opacity-0 group-hover:opacity-100"></div>
      
      {/* Main card */}
      <motion.div
        className="relative bg-gray-800/50 backdrop-blur-enhanced border border-gray-700 rounded-xl p-6 hover:bg-gray-800/70 transition-all duration-300 floating-card"
        animate={{
          scale: isExpanded ? 1.05 : 1,
          height: isExpanded ? 'auto' : 'auto',
        }}
        transition={{ duration: 0.3 }}
      >
        {/* Icon with floating animation */}
        <motion.div 
          className="mb-4"
          animate={{ 
            y: [0, -5, 0],
          }}
          transition={{ 
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          {icon}
        </motion.div>
        
        {/* Title */}
        <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-blue-400 transition-colors duration-300">
          {title}
        </h3>
        
        {/* Description */}
        <p className="text-gray-400 group-hover:text-gray-300 transition-colors duration-300">
          {description}
        </p>
        
        {/* Expandable details */}
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ 
            height: isExpanded ? 'auto' : 0,
            opacity: isExpanded ? 1 : 0
          }}
          transition={{ duration: 0.3 }}
          className="overflow-hidden"
        >
          {details && (
            <div className="mt-4 pt-4 border-t border-gray-700">
              <div className="text-sm text-gray-300 space-y-2">
                {details}
              </div>
            </div>
          )}
        </motion.div>
        
        {/* Expand indicator */}
        <motion.div
          className="absolute top-4 right-4 text-gray-500"
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.3 }}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </motion.div>
      </motion.div>
      
      {/* Floating particles around card */}
      <div className="absolute inset-0 pointer-events-none">
        {[...Array(3)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-blue-400 rounded-full"
            style={{
              left: `${20 + i * 30}%`,
              top: `${10 + i * 20}%`,
            }}
            animate={{
              y: [0, -10, 0],
              opacity: [0.3, 1, 0.3],
            }}
            transition={{
              duration: 2 + i,
              repeat: Infinity,
              ease: "easeInOut",
              delay: i * 0.5,
            }}
          />
        ))}
      </div>
    </motion.div>
  );
};

export default AnimatedFeatureCard; 