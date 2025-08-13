import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const FloatingButton = ({ 
  children, 
  onClick, 
  className = "", 
  variant = "primary",
  size = "md",
  icon,
  expandedContent,
  ...props 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const baseClasses = "relative inline-flex items-center justify-center font-semibold rounded-full transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl glow-button";
  
  const variants = {
    primary: "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white",
    secondary: "bg-gray-800/80 backdrop-blur-sm border border-gray-600 hover:bg-gray-700/80 hover:border-gray-500 text-white",
    danger: "bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white",
    success: "bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white",
  };

  const sizes = {
    sm: "px-4 py-2 text-sm",
    md: "px-6 py-3 text-base",
    lg: "px-8 py-4 text-lg",
    xl: "px-12 py-6 text-xl",
  };

  const handleClick = (e) => {
    if (expandedContent) {
      setIsExpanded(!isExpanded);
    }
    if (onClick) {
      onClick(e);
    }
  };

  return (
    <div className="relative">
      <motion.button
        className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`}
        onClick={handleClick}
        whileHover={{ 
          scale: 1.05,
          y: -2,
          boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"
        }}
        whileTap={{ scale: 0.95 }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        {...props}
      >
        {icon && <span className="mr-2">{icon}</span>}
        {children}
      </motion.button>

      <AnimatePresence>
        {isExpanded && expandedContent && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: -10 }}
            transition={{ duration: 0.2 }}
            className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-4 z-50"
          >
            <div className="bg-gray-900/95 backdrop-blur-sm border border-gray-700 rounded-xl p-4 shadow-2xl min-w-[200px]">
              {expandedContent}
            </div>
            {/* Arrow */}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900/95"></div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default FloatingButton; 