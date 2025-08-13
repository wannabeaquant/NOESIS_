import { motion } from 'framer-motion';

const DynamicBackground = () => {
  return (
    <div className="fixed inset-0 z-0 pointer-events-none">
      {/* Animated gradient background */}
      <motion.div
        className="absolute inset-0"
        animate={{
          background: [
            "radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%)",
            "radial-gradient(circle at 80% 80%, rgba(236, 72, 153, 0.1) 0%, transparent 50%), radial-gradient(circle at 20% 20%, rgba(245, 158, 11, 0.1) 0%, transparent 50%)",
            "radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 50%), radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.1) 0%, transparent 50%)",
            "radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%)"
          ]
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      {/* Floating orbs */}
      {[...Array(5)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-32 h-32 rounded-full opacity-5"
          style={{
            left: `${20 + i * 15}%`,
            top: `${10 + i * 20}%`,
            background: `radial-gradient(circle, rgba(${59 + i * 20}, ${130 - i * 10}, ${246 - i * 30}, 0.1), transparent)`
          }}
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.05, 0.1, 0.05],
            x: [0, 50, 0],
            y: [0, -30, 0],
          }}
          transition={{
            duration: 8 + i * 2,
            repeat: Infinity,
            ease: "easeInOut",
            delay: i * 1.5
          }}
        />
      ))}
      
      {/* Subtle grid pattern */}
      <div 
        className="absolute inset-0 opacity-5"
        style={{
          backgroundImage: `
            linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }}
      />
    </div>
  );
};

export default DynamicBackground; 