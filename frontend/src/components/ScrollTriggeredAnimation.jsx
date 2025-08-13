import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';

const ScrollTriggeredAnimation = ({ 
  children, 
  animation = "fadeInUp",
  delay = 0,
  duration = 0.6,
  className = "",
  threshold = 0.1
}) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { threshold, once: true });

  const animations = {
    fadeInUp: {
      hidden: { opacity: 0, y: 50 },
      visible: { opacity: 1, y: 0 }
    },
    fadeInDown: {
      hidden: { opacity: 0, y: -50 },
      visible: { opacity: 1, y: 0 }
    },
    fadeInLeft: {
      hidden: { opacity: 0, x: -50 },
      visible: { opacity: 1, x: 0 }
    },
    fadeInRight: {
      hidden: { opacity: 0, x: 50 },
      visible: { opacity: 1, x: 0 }
    },
    scaleIn: {
      hidden: { opacity: 0, scale: 0.8 },
      visible: { opacity: 1, scale: 1 }
    },
    rotateIn: {
      hidden: { opacity: 0, rotate: -180, scale: 0.8 },
      visible: { opacity: 1, rotate: 0, scale: 1 }
    },
    slideInUp: {
      hidden: { opacity: 0, y: 100 },
      visible: { opacity: 1, y: 0 }
    },
    slideInDown: {
      hidden: { opacity: 0, y: -100 },
      visible: { opacity: 1, y: 0 }
    }
  };

  const selectedAnimation = animations[animation] || animations.fadeInUp;

  return (
    <motion.div
      ref={ref}
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      variants={selectedAnimation}
      transition={{ 
        duration, 
        delay,
        ease: "easeOut"
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
};

export default ScrollTriggeredAnimation; 