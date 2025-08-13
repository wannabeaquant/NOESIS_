import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight, Globe, Shield, Zap, Eye, Brain, AlertTriangle, MapPin, Clock, Users, TrendingUp, BarChart3, Activity, Satellite, Wifi, Database } from 'lucide-react';
import AnimatedMapBackground from '../components/AnimatedMapBackground';
import FloatingButton from '../components/FloatingButton';
import AnimatedFeatureCard from '../components/AnimatedFeatureCard';
import ParticleSystem from '../components/ParticleSystem';
import ScrollTriggeredAnimation from '../components/ScrollTriggeredAnimation';
import DynamicBackground from '../components/DynamicBackground';
import InteractiveMap from '../components/InteractiveMap';

const Home = () => {
  const features = [
    {
      icon: <Eye className="h-8 w-8 text-cyan-400" />,
      title: "Real-Time Monitoring",
      description: "24/7 surveillance of global protest activities using advanced OSINT techniques",
      details: [
        "• Social media sentiment analysis",
        "• News aggregation and verification",
        "• Satellite imagery processing",
        "• Crowd size estimation algorithms"
      ]
    },
    {
      icon: <Brain className="h-8 w-8 text-emerald-400" />,
      title: "AI-Powered Analysis",
      description: "Machine learning algorithms analyze sentiment, severity, and verification status",
      details: [
        "• Natural language processing",
        "• Image recognition and analysis",
        "• Predictive modeling",
        "• Automated fact-checking"
      ]
    },
    {
      icon: <Globe className="h-8 w-8 text-violet-400" />,
      title: "Global Coverage",
      description: "Comprehensive tracking across all continents and major urban centers",
      details: [
        "• 195+ countries monitored",
        "• 1000+ cities tracked",
        "• Multi-language support",
        "• Local context analysis"
      ]
    },
    {
      icon: <Zap className="h-8 w-8 text-amber-400" />,
      title: "Instant Alerts",
      description: "Immediate notifications for high-severity incidents and escalating situations",
      details: [
        "• Real-time push notifications",
        "• Email alerts with priority levels",
        "• SMS for critical events",
        "• Custom alert thresholds"
      ]
    }
  ];

  const steps = [
    {
      number: "01",
      title: "Data Collection",
      description: "Aggregate information from social media, news sources, and citizen reports",
      icon: <Database className="h-6 w-6" />
    },
    {
      number: "02",
      title: "NLP Processing",
      description: "Natural language processing extracts key details and sentiment analysis",
      icon: <Brain className="h-6 w-6" />
    },
    {
      number: "03",
      title: "Clustering & Verification",
      description: "Group related incidents and verify authenticity through multiple sources",
      icon: <Shield className="h-6 w-6" />
    },
    {
      number: "04",
      title: "Visualization & Alerts",
      description: "Display on interactive map with real-time alerts for critical situations",
      icon: <MapPin className="h-6 w-6" />
    }
  ];

  const stats = [
    { 
      label: "Active Incidents", 
      value: "1,247", 
      icon: <Activity className="h-6 w-6" />, 
      color: "text-red-400",
      bgColor: "bg-red-500/10",
      borderColor: "border-red-500/20"
    },
    { 
      label: "Countries Monitored", 
      value: "195+", 
      icon: <Globe className="h-6 w-6" />, 
      color: "text-blue-400",
      bgColor: "bg-blue-500/10",
      borderColor: "border-blue-500/20"
    },
    { 
      label: "Data Sources", 
      value: "500+", 
      icon: <Wifi className="h-6 w-6" />, 
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
      borderColor: "border-emerald-500/20"
    },
    { 
      label: "Response Time", 
      value: "<30s", 
      icon: <Clock className="h-6 w-6" />, 
      color: "text-amber-400",
      bgColor: "bg-amber-500/10",
      borderColor: "border-amber-500/20"
    }
  ];

  // Reduce floating icons to 3
  const floatingIcons = [
    { icon: "🌍", delay: 0 },
    { icon: "📡", delay: 0.5 },
    { icon: "⚠️", delay: 1 }
  ];

  return (
    <div className="bg-slate-950 relative overflow-hidden font-['Inter']">
      {/* Dynamic Background */}
      <DynamicBackground />
      
      {/* Particle System - reduced count */}
      <ParticleSystem count={10} />
      
      {/* Animated Map Background - keep only this as main animated background */}
      <AnimatedMapBackground />
      
      {/* Floating Icons - reduced */}
      <div className="fixed inset-0 pointer-events-none z-10">
        {floatingIcons.map((item, index) => (
          <motion.div
            key={index}
            className="absolute text-2xl opacity-15"
            style={{
              left: `${20 + (index * 30)}%`,
              top: `${20 + (index * 20)}%`,
            }}
            animate={{
              y: [0, -10, 0],
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: 5,
              repeat: Infinity,
              ease: "easeInOut",
              delay: item.delay
            }}
          >
            {item.icon}
          </motion.div>
        ))}
      </div>

      {/* Hero Section */}
      <section className="relative z-20 overflow-hidden min-h-screen flex items-center">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-32 w-full">
          <div className="text-center">
            {/* Floating Icons */}
            <motion.div 
              className="flex justify-center items-center space-x-6 mb-12"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1 }}
            >
              <motion.div
                animate={{ 
                  y: [0, -10, 0],
                  rotate: [0, 5, 0]
                }}
                transition={{ 
                  duration: 4,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="p-4 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl border border-blue-500/30"
              >
                <Globe className="h-16 w-16 text-cyan-400" />
              </motion.div>
              <motion.div
                animate={{ 
                  y: [0, 10, 0],
                  rotate: [0, -5, 0]
                }}
                transition={{ 
                  duration: 4,
                  repeat: Infinity,
                  ease: "easeInOut",
                  delay: 1
                }}
                className="p-4 bg-gradient-to-br from-red-500/20 to-pink-500/20 rounded-2xl border border-red-500/30"
              >
                <Shield className="h-16 w-16 text-red-400" />
              </motion.div>
            </motion.div>

            {/* Main Title */}
            <motion.h1 
              className="text-5xl md:text-7xl lg:text-8xl font-bold text-white mb-8 leading-tight"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, delay: 0.3 }}
            >
              From Noise to{' '}
              <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
                Noesis
              </span>
            </motion.h1>

            <motion.p 
              className="text-xl md:text-2xl text-slate-300 mb-8 max-w-4xl mx-auto font-light"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, delay: 0.6 }}
            >
              Decoding Global Disruption through Real-Time OSINT Intelligence
            </motion.p>

            <motion.p 
              className="text-lg text-slate-400 mb-16 max-w-3xl mx-auto leading-relaxed"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, delay: 0.9 }}
            >
              Advanced AI-powered monitoring system that transforms chaotic protest data 
              into actionable intelligence for governments, organizations, and researchers worldwide.
            </motion.p>

            {/* Floating Buttons */}
            <motion.div 
              className="flex flex-col sm:flex-row gap-6 justify-center"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, delay: 1.2 }}
            >
              <FloatingButton
                variant="primary"
                size="xl"
                icon={<ArrowRight className="h-6 w-6" />}
                expandedContent={
                  <div className="text-left">
                    <h4 className="font-semibold text-white mb-3">Live Dashboard Features:</h4>
                    <ul className="text-sm text-slate-300 space-y-2">
                      <li>• Real-time incident tracking</li>
                      <li>• Interactive global map</li>
                      <li>• Advanced filtering options</li>
                      <li>• Export capabilities</li>
                    </ul>
                  </div>
                }
              >
                <Link to="/dashboard" className="flex items-center">
                  Explore Live Dashboard
                </Link>
              </FloatingButton>

              <FloatingButton
                variant="secondary"
                size="xl"
                expandedContent={
                  <div className="text-left">
                    <h4 className="font-semibold text-white mb-3">Learn More About:</h4>
                    <ul className="text-sm text-slate-300 space-y-2">
                      <li>• Our technology stack</li>
                      <li>• Data sources & methodology</li>
                      <li>• Use cases & applications</li>
                      <li>• Success stories</li>
                    </ul>
                  </div>
                }
              >
                <Link to="/about" className="flex items-center">
                  Learn More
                </Link>
              </FloatingButton>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative z-20 py-20 bg-slate-900/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollTriggeredAnimation animation="fadeInUp" delay={0.2}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {stats.map((stat, index) => (
                <motion.div
                  key={index}
                  className={`relative group cursor-pointer ${stat.bgColor} ${stat.borderColor} border rounded-2xl p-6 backdrop-blur-sm hover:shadow-2xl transition-all duration-300`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -8, scale: 1.05 }}
                >
                  {/* Glow effect */}
                  <div className={`absolute inset-0 ${stat.bgColor} rounded-2xl blur-xl opacity-0 group-hover:opacity-50 transition-opacity duration-300`}></div>
                  
                  <div className="relative z-10 text-center">
                    <motion.div
                      className={`mx-auto mb-4 ${stat.color}`}
                      animate={{ 
                        y: [0, -8, 0],
                      }}
                      transition={{ 
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut",
                        delay: index * 0.2
                      }}
                    >
                      {stat.icon}
                    </motion.div>
                    <div className="text-4xl font-bold text-white mb-2">{stat.value}</div>
                    <div className="text-slate-400 text-sm font-medium">{stat.label}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </ScrollTriggeredAnimation>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative z-20 py-24 bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollTriggeredAnimation animation="fadeInUp" delay={0.2}>
            <div className="text-center mb-20">
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                Powerful Intelligence Capabilities
              </h2>
              <p className="text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed">
                Cutting-edge technology meets human insight to deliver unprecedented 
                situational awareness of global civil unrest.
              </p>
            </div>
          </ScrollTriggeredAnimation>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <AnimatedFeatureCard
                key={index}
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
                details={feature.details}
                index={index}
              />
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="relative z-20 py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollTriggeredAnimation animation="fadeInUp" delay={0.2}>
            <div className="text-center mb-20">
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                How It Works
              </h2>
              <p className="text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed">
                Our sophisticated pipeline transforms raw data into actionable intelligence 
                through a four-stage process.
              </p>
            </div>
          </ScrollTriggeredAnimation>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <motion.div 
                key={index} 
                className="relative group"
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                {/* Number Badge */}
                <motion.div 
                  className="absolute -top-4 left-1/2 transform -translate-x-1/2 z-20"
                  animate={{ 
                    scale: [1, 1.1, 1],
                  }}
                  transition={{ 
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: index * 0.5
                  }}
                >
                  <div className="bg-gradient-to-r from-cyan-500 to-purple-500 text-white font-bold text-lg px-4 py-2 rounded-full shadow-lg">
                    {step.number}
                  </div>
                </motion.div>

                <motion.div 
                  className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 rounded-2xl p-8 pt-12 hover:border-slate-600/50 transition-all duration-300 backdrop-blur-sm group-hover:shadow-2xl"
                  whileHover={{ y: -12, scale: 1.02 }}
                >
                  <div className="flex items-center mb-4">
                    <div className="text-cyan-400 mr-3 group-hover:scale-110 transition-transform duration-300">
                      {step.icon}
                    </div>
                    <h3 className="text-xl font-semibold text-white">
                      {step.title}
                    </h3>
                  </div>
                  
                  <p className="text-slate-400 leading-relaxed">
                    {step.description}
                  </p>
                </motion.div>
                
                {index < steps.length - 1 && (
                  <motion.div 
                    className="hidden lg:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10"
                    animate={{ x: [0, 8, 0] }}
                    transition={{ 
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut",
                      delay: index * 0.3
                    }}
                  >
                    <ArrowRight className="h-8 w-8 text-slate-600" />
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Live Dashboard Preview */}
      <section className="relative z-20 py-24 bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <ScrollTriggeredAnimation animation="fadeInUp" delay={0.2}>
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                Live Dashboard Preview
              </h2>
              <p className="text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed">
                Experience real-time monitoring with our interactive global map 
                showing current protest activities and their severity levels.
              </p>
            </div>
          </ScrollTriggeredAnimation>
          
          <motion.div 
            className="bg-slate-800/50 rounded-3xl p-8 border border-slate-700/50 backdrop-blur-sm"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <div className="aspect-video bg-gradient-to-br from-slate-700 to-slate-800 rounded-2xl flex items-center justify-center border border-slate-600/50 relative overflow-hidden">
              <InteractiveMap />
              
              {/* Animated Legend */}
              <div className="absolute bottom-4 right-4 bg-slate-900/80 backdrop-blur-sm rounded-xl p-4 border border-slate-700/50">
                <h4 className="text-white font-semibold mb-3 text-sm">Legend</h4>
                <div className="space-y-2 text-xs">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-slate-300">Safe</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-orange-500 rounded-full animate-pulse"></div>
                    <span className="text-slate-300">Unverified</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                    <span className="text-slate-300">Critical</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-20 py-24">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <ScrollTriggeredAnimation animation="fadeInUp" delay={0.2}>
            <motion.div 
              className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-3xl p-16 border border-slate-700/50 backdrop-blur-sm"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <motion.div
                animate={{ 
                  scale: [1, 1.1, 1],
                }}
                transition={{ 
                  duration: 3,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              >
                <AlertTriangle className="h-20 w-20 text-amber-400 mx-auto mb-8" />
              </motion.div>
              
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-8">
                Stay Ahead of Global Events
              </h2>
              <p className="text-xl text-slate-300 mb-12 leading-relaxed">
                Join organizations worldwide who rely on our intelligence platform 
                for critical decision-making and risk assessment.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-6 justify-center">
                <FloatingButton
                  variant="danger"
                  size="xl"
                  icon={<ArrowRight className="h-6 w-6" />}
                >
                  <Link to="/contact">
                    Get Started Today
                  </Link>
                </FloatingButton>
                
                <FloatingButton
                  variant="secondary"
                  size="xl"
                >
                  <Link to="/team">
                    Meet Our Team
                  </Link>
                </FloatingButton>
              </div>
            </motion.div>
          </ScrollTriggeredAnimation>
        </div>
      </section>
    </div>
  );
};

export default Home;

