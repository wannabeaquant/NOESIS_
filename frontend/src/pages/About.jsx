import { Shield, Globe, Brain, Users, Target, Zap, AlertCircle, CheckCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import DynamicBackground from '../components/DynamicBackground';

const About = () => {
  const problems = [
    {
      icon: <AlertCircle className="h-8 w-8 text-red-400" />,
      title: "Information Overload",
      description: "Thousands of protest-related posts flood social media daily, making it impossible to manually track and verify incidents."
    },
    {
      icon: <AlertCircle className="h-8 w-8 text-orange-400" />,
      title: "Delayed Response",
      description: "Traditional monitoring methods often result in delayed awareness of critical situations, hindering effective response."
    },
    {
      icon: <AlertCircle className="h-8 w-8 text-yellow-400" />,
      title: "Lack of Verification",
      description: "Misinformation and unverified reports create confusion and lead to poor decision-making during crisis situations."
    }
  ];

  const solutions = [
    {
      icon: <Brain className="h-8 w-8 text-blue-400" />,
      title: "AI-Powered Processing",
      description: "Advanced machine learning algorithms process thousands of data points per minute, extracting relevant information automatically."
    },
    {
      icon: <Zap className="h-8 w-8 text-green-400" />,
      title: "Real-Time Analysis",
      description: "Instant processing and classification of incidents with immediate alerts for high-priority situations."
    },
    {
      icon: <CheckCircle className="h-8 w-8 text-purple-400" />,
      title: "Multi-Source Verification",
      description: "Cross-reference multiple sources to verify incident authenticity and provide confidence scores."
    }
  ];

  const capabilities = [
    "Natural Language Processing for sentiment analysis",
    "Geospatial clustering of related incidents",
    "Multi-language support for global coverage",
    "Real-time data ingestion from 50+ sources",
    "Automated threat level assessment",
    "Historical trend analysis and prediction"
  ];

  // Animation state for Problems/Solutions
  const [problemsOpen, setProblemsOpen] = useState(false);
  const [solutionsOpen, setSolutionsOpen] = useState(false);

  return (
    <div className="bg-slate-950 min-h-screen font-['Inter'] relative overflow-hidden">
      <DynamicBackground />
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/10 via-purple-900/10 to-gray-900/10"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex justify-center items-center space-x-4 mb-8">
              <Shield className="h-12 w-12 text-blue-400" />
              <Globe className="h-12 w-12 text-purple-400" />
              <Brain className="h-12 w-12 text-green-400" />
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold mb-2 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
              About Our Platform
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-4 max-w-4xl mx-auto">
              Transforming chaos into clarity through advanced OSINT and AI technology
            </p>
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Our Mission Card */}
            <motion.div
              className="rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 p-10 shadow-xl mb-8 lg:mb-0 backdrop-blur-md hover:shadow-2xl transition-all duration-300"
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              viewport={{ once: true }}
            >
              <h2 className="text-3xl md:text-4xl font-extrabold mb-6 text-center bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
                Our Mission
              </h2>
              <p className="text-lg text-gray-300 mb-6 text-center">
                In an era of global connectivity and rapid information flow, understanding 
                civil unrest and protest movements has become more critical than ever. 
                Our platform bridges the gap between raw data and actionable intelligence.
              </p>
              <p className="text-lg text-gray-300 mb-6 text-center">
                We believe that informed decision-making requires access to verified, 
                real-time information. By leveraging cutting-edge AI and OSINT techniques, 
                we transform the noise of social media and news into clear, actionable insights.
              </p>
              <div className="flex items-center justify-center space-x-4">
                <Target className="h-8 w-8 text-blue-400" />
                <span className="text-xl font-semibold text-white">
                  Empowering informed decisions through intelligence
                </span>
              </div>
            </motion.div>
            {/* Key Statistics Card */}
            <motion.div
              className="rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 p-10 shadow-xl glassmorphism text-center hover:shadow-2xl hover:scale-105 hover:border-blue-500/60 transition-all duration-300 group cursor-pointer"
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 }}
              viewport={{ once: true }}
              whileHover={{ scale: 1.05, boxShadow: '0 8px 32px 0 rgba(0, 184, 255, 0.25)' }}
            >
              <h3 className="text-2xl font-extrabold mb-6 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">Key Statistics</h3>
              <div className="grid grid-cols-2 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-400 mb-2 group-hover:glow-text">50+</div>
                  <div className="text-gray-300">Data Sources</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-400 mb-2 group-hover:glow-text">24/7</div>
                  <div className="text-gray-300">Monitoring</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-400 mb-2 group-hover:glow-text">195</div>
                  <div className="text-gray-300">Countries</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-yellow-400 mb-2 group-hover:glow-text">&lt;60s</div>
                  <div className="text-gray-300">Alert Time</div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Problem & Solution */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div className="text-center mb-16" initial={{ opacity: 0, y: 40 }} whileInView={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} viewport={{ once: true }}>
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              The Challenge We Solve
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Traditional monitoring methods fall short in today's fast-paced, 
              information-rich environment. We've built a solution that addresses 
              these critical gaps.
            </p>
          </motion.div>

          {/* Problems */}
          <motion.div
            className="mb-16 flex flex-col items-center"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <motion.div
              className="main-card-problems relative rounded-2xl bg-gradient-to-br from-red-900/40 to-slate-900/80 border border-red-700/40 p-8 shadow-xl mb-8 w-full max-w-2xl text-center cursor-pointer hover:shadow-2xl hover:scale-105 transition-all duration-300 group"
              onMouseEnter={() => setProblemsOpen(true)}
              onMouseLeave={() => setProblemsOpen(false)}
              whileHover={{ scale: 1.04, boxShadow: '0 8px 32px 0 rgba(255, 0, 80, 0.18)' }}
            >
              <h3 className="text-2xl font-bold text-white mb-2">Current Problems</h3>
              <p className="text-gray-300 mb-2">What makes protest monitoring so difficult?</p>
              {/* Animated glowing border */}
              <div className="absolute inset-0 rounded-2xl pointer-events-none group-hover:shadow-[0_0_40px_10px_rgba(255,0,80,0.15)] transition-all duration-300" />
            </motion.div>
            <AnimatePresence>
              {problemsOpen && (
                <motion.div
                  className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-4 w-full max-w-5xl"
                  initial="hidden"
                  animate="visible"
                  exit="hidden"
                  variants={{
                    hidden: { opacity: 0, scale: 0.95, y: 20 },
                    visible: { opacity: 1, scale: 1, y: 0, transition: { staggerChildren: 0.12 } }
                  }}
                >
                  {problems.map((problem, index) => (
                    <motion.div
                      key={index}
                      className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-red-700/40 rounded-2xl p-8 shadow-xl hover:shadow-2xl hover:scale-105 hover:border-red-500/60 transition-all duration-300 group cursor-pointer relative"
                      initial={{ opacity: 0, scale: 0.95, y: 20 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95, y: 20 }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                    >
                      <div className="mb-4 group-hover:glow-text">{problem.icon}</div>
                      <h4 className="text-xl font-semibold text-white mb-3">
                        {problem.title}
                      </h4>
                      <p className="text-gray-400">
                        {problem.description}
                      </p>
                      {/* Glowing border effect */}
                      <div className="absolute inset-0 rounded-2xl pointer-events-none group-hover:shadow-[0_0_40px_10px_rgba(255,0,80,0.12)] transition-all duration-300" />
                    </motion.div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Solutions */}
          <motion.div
            className="flex flex-col items-center"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <motion.div
              className="main-card-solutions relative rounded-2xl bg-gradient-to-br from-green-900/40 to-slate-900/80 border border-green-700/40 p-8 shadow-xl mb-8 w-full max-w-2xl text-center cursor-pointer hover:shadow-2xl hover:scale-105 transition-all duration-300 group"
              onMouseEnter={() => setSolutionsOpen(true)}
              onMouseLeave={() => setSolutionsOpen(false)}
              whileHover={{ scale: 1.04, boxShadow: '0 8px 32px 0 rgba(0, 255, 80, 0.18)' }}
            >
              <h3 className="text-2xl font-bold text-white mb-2">Our Solutions</h3>
              <p className="text-gray-300 mb-2">How do we solve these challenges?</p>
              {/* Animated glowing border */}
              <div className="absolute inset-0 rounded-2xl pointer-events-none group-hover:shadow-[0_0_40px_10px_rgba(0,255,80,0.15)] transition-all duration-300" />
            </motion.div>
            <AnimatePresence>
              {solutionsOpen && (
                <motion.div
                  className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-4 w-full max-w-5xl"
                  initial="hidden"
                  animate="visible"
                  exit="hidden"
                  variants={{
                    hidden: { opacity: 0, scale: 0.95, y: 20 },
                    visible: { opacity: 1, scale: 1, y: 0, transition: { staggerChildren: 0.12 } }
                  }}
                >
                  {solutions.map((solution, index) => (
                    <motion.div
                      key={index}
                      className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-green-700/40 rounded-2xl p-8 shadow-xl hover:shadow-2xl hover:scale-105 hover:border-green-500/60 transition-all duration-300 group cursor-pointer relative"
                      initial={{ opacity: 0, scale: 0.95, y: 20 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95, y: 20 }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                    >
                      <div className="mb-4 group-hover:glow-text">{solution.icon}</div>
                      <h4 className="text-xl font-semibold text-white mb-3">
                        {solution.title}
                      </h4>
                      <p className="text-gray-400">
                        {solution.description}
                      </p>
                      {/* Glowing border effect */}
                      <div className="absolute inset-0 rounded-2xl pointer-events-none group-hover:shadow-[0_0_40px_10px_rgba(0,255,80,0.12)] transition-all duration-300" />
                    </motion.div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </section>

      {/* Technology Section */}
      <section className="py-20 bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 rounded-xl p-8 border border-slate-700/50 shadow-xl"
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              viewport={{ once: true }}
            >
              <h3 className="text-2xl font-bold text-white mb-6">
                Advanced Capabilities
              </h3>
              <ul className="space-y-4">
                {capabilities.map((capability, index) => (
                  <li key={index} className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-400 flex-shrink-0" />
                    <span className="text-gray-300">{capability}</span>
                  </li>
                ))}
              </ul>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 }}
              viewport={{ once: true }}
            >
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
                How AI & OSINT Work Together
              </h2>
              <p className="text-lg text-gray-300 mb-6">
                Our platform combines Open Source Intelligence (OSINT) methodologies 
                with state-of-the-art artificial intelligence to create a comprehensive 
                monitoring solution.
              </p>
              <p className="text-lg text-gray-300 mb-6">
                Machine learning models trained on historical protest data can identify 
                patterns, predict escalation, and classify incidents with remarkable accuracy. 
                This allows for proactive rather than reactive responses.
              </p>
              <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-6">
                <h4 className="text-xl font-semibold text-blue-400 mb-3">
                  Why This Matters
                </h4>
                <p className="text-gray-300">
                  In crisis situations, every minute counts. Our platform provides 
                  the situational awareness needed for governments, NGOs, journalists, 
                  and researchers to make informed decisions and respond appropriately 
                  to developing situations.
                </p>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div className="text-center mb-16" initial={{ opacity: 0, y: 40 }} whileInView={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} viewport={{ once: true }}>
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Who Benefits
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Our platform serves diverse stakeholders who need reliable, 
              real-time intelligence about civil unrest and protest activities.
            </p>
          </motion.div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: <Shield className="h-8 w-8 text-blue-400" />,
                title: "Government Agencies",
                description: "Public safety and security planning"
              },
              {
                icon: <Users className="h-8 w-8 text-green-400" />,
                title: "NGOs & Activists",
                description: "Human rights monitoring and advocacy"
              },
              {
                icon: <Globe className="h-8 w-8 text-purple-400" />,
                title: "Journalists & Researchers",
                description: "Data-driven reporting and analysis"
              },
              {
                icon: <Target className="h-8 w-8 text-yellow-400" />,
                title: "Crisis Responders",
                description: "Rapid response and resource allocation"
              }
            ].map((useCase, index) => (
              <motion.div
                key={index}
                className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 rounded-2xl p-8 shadow-xl hover:shadow-2xl hover:scale-105 transition-all duration-300 text-center group cursor-pointer"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.07 }}
              >
                <div className="mb-4 group-hover:glow-text">{useCase.icon}</div>
                <h4 className="text-xl font-semibold text-white mb-3">
                  {useCase.title}
                </h4>
                <p className="text-gray-400">
                  {useCase.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;

