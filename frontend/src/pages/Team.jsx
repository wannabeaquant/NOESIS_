import { useState } from 'react';
import { Users, ChevronDown, ChevronUp, Github, Linkedin, Mail, Award, Code, Database, Shield, Globe } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import DynamicBackground from '../components/DynamicBackground';

const Team = () => {
  const [openFaq, setOpenFaq] = useState(null);

  const teamMembers = [
    {
      name: "Sanidhya Shishodia",
      role: "Student at VIPS",
      bio: "CS undergrad with experience in AI agents, fintech, and Web3. Interned at LitAmor, worked with CapxAI (MNC), and built multiple award-winning projects.",
      icon: <Code className="h-8 w-8 text-blue-400" />,
      expertise: ["AI/ML", "OSINT", "Frontend"],
      social: {
        github: "https://github.com/dev-sanidhya",
        linkedin: "https://linkedin.com/in/sanidhya-shishodia",
        email: "shishodiasanidhya@gmail.com"
      }
    },
    {
      name: "Vidisha Deswal",
      role: "Student at IGDTUW",
      bio: "CS & AI undergrad at IGDTUW, engaged in tech-for-good and leadership. Active in IEEE IGDTUW’s WIEmpower and Enactus IGDTUW, contributing to STEM empowerment initiatives. Passionate about building real-world impact through tech.",
      icon: <Database className="h-8 w-8 text-green-400" />,
      expertise: ["Frontend", "OSINT", "AI/ML"],
      social: {
        github: "https://github.com/vidishadeswal",
        linkedin: "https://www.linkedin.com/in/vidisha-deswal-b2722332b/",
        email: "vidishadeswal@gmail.com"
      }
    },
    {
      name: "Atharva Singh",
      role: "Student at VIPS",
      bio: "CS undergrad with a background in cybersecurity and AI systems. Interned at Himitsu Labs, worked with CapxAI (MNC), and contributed to cutting-edge tech solutions.",
      icon: <Shield className="h-8 w-8 text-purple-400" />,
      expertise: ["OSINT", "AI/ML", "Backend"],
      social: {
        github: "https://github.com/wannabeaquant",
        linkedin: "https://www.linkedin.com/in/atharva-singh-/",
        email: "atharvasingh0405@gmail.com"
      }
    },
    
  ];

  const timeline = [
    {
      phase: "Data Ingestion",
      description: "Collect information from social media, news outlets, government sources, and citizen reports",
      details: "Our system monitors 50+ data sources in real-time, processing thousands of posts per minute"
    },
    {
      phase: "NLP Processing",
      description: "Extract key information using natural language processing and sentiment analysis",
      details: "Advanced algorithms identify location, participants, severity, and emotional tone of incidents"
    },
    {
      phase: "Verification & Clustering",
      description: "Cross-reference sources and group related incidents for accuracy",
      details: "Multi-source verification ensures reliability while clustering reduces duplicate reports"
    },
    {
      phase: "Intelligence Generation",
      description: "Generate actionable intelligence with threat assessment and predictions",
      details: "AI models predict escalation probability and provide confidence scores for each incident"
    },
    {
      phase: "Alert & Visualization",
      description: "Display on interactive map with real-time alerts for critical situations",
      details: "Color-coded severity levels and instant notifications keep users informed of developing situations"
    }
  ];

  const faqs = [
    {
      question: "How accurate is the incident verification?",
      answer: "Our multi-source verification system achieves 95%+ accuracy by cross-referencing at least 3 independent sources for each incident. We use confidence scores to indicate reliability levels."
    },
    {
      question: "What data sources do you monitor?",
      answer: "We monitor 50+ sources including social media platforms (Twitter, Facebook, Telegram), news outlets, government feeds, NGO reports, and citizen journalism platforms across 195 countries."
    },
    {
      question: "How quickly are incidents detected and reported?",
      answer: "Our system processes data in real-time with average detection times under 60 seconds. Critical incidents trigger immediate alerts to relevant stakeholders."
    },
    {
      question: "Is the platform available for academic research?",
      answer: "Yes, we offer special academic licenses for researchers studying social movements, conflict analysis, and crisis management. Contact us for research partnership opportunities."
    },
    {
      question: "How do you ensure data privacy and security?",
      answer: "We follow strict data protection protocols, anonymize personal information, and comply with international privacy regulations. All data is encrypted and stored securely."
    },
    {
      question: "Can the system predict future protests or unrest?",
      answer: "Our AI models can identify patterns and risk factors that may indicate potential unrest, but we focus on monitoring existing incidents rather than prediction to avoid bias."
    },
    {
      question: "What makes your platform different from news monitoring?",
      answer: "Unlike traditional news monitoring, we provide real-time analysis, verification, geospatial clustering, sentiment analysis, and threat assessment specifically for civil unrest incidents."
    },
    {
      question: "How can organizations integrate with your platform?",
      answer: "We offer REST APIs, webhook notifications, and custom dashboard integrations. Our team works with clients to implement solutions that fit their specific workflows."
    }
  ];

  const toggleFaq = (index) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  return (
    <div className="bg-slate-950 min-h-screen font-['Inter'] relative overflow-hidden">
      <DynamicBackground />
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/10 via-purple-900/10 to-gray-900/10"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <Users className="h-16 w-16 text-blue-400 mx-auto mb-8" />
            <motion.h1
              className="text-4xl md:text-5xl lg:text-6xl font-extrabold mb-2 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              Meet Our Team
            </motion.h1>
            <motion.p
              className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              World-class experts in intelligence, technology, and crisis analysis working together to decode global disruption.
            </motion.p>
          </div>
        </div>
      </section>
      {/* Team Members */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Leadership Team
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Our diverse team brings together decades of experience in intelligence, technology, and crisis management from leading organizations worldwide.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 w-full">
            {teamMembers.map((member, index) => (
              <motion.div
                key={index}
                className="rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 p-8 shadow-xl glassmorphism text-center hover:shadow-2xl hover:scale-105 hover:border-blue-500/60 transition-all duration-300 group cursor-pointer"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <div className="w-20 h-20 bg-gradient-to-br from-gray-700 to-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
                  {member.icon}
                </div>
                <h3 className="text-xl font-bold text-white mb-2">
                  {member.name}
                </h3>
                <p className="text-blue-400 font-medium mb-4">
                  {member.role}
                </p>
                <p className="text-gray-300 text-sm mb-4 leading-relaxed">
                  {member.bio}
                </p>
                <div className="mb-4">
                  <h4 className="text-white font-semibold mb-2">Expertise:</h4>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {member.expertise.map((skill, skillIndex) => (
                      <span
                        key={skillIndex}
                        className="px-2 py-1 bg-blue-900/30 text-blue-300 text-xs rounded-full"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="flex justify-center space-x-4">
                  <a
                    href={member.social.github}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <Github className="h-5 w-5" />
                  </a>
                  <a
                    href={member.social.linkedin}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <Linkedin className="h-5 w-5" />
                  </a>
                  <a
                    href={`mailto:${member.social.email}`}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <Mail className="h-5 w-5" />
                  </a>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      {/* How It Works Timeline */}
      <section className="py-20 bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              How Our System Works
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Our sophisticated pipeline transforms raw data into actionable intelligence 
              through a carefully orchestrated five-stage process.
            </p>
          </div>
          <div className="space-y-8">
            {timeline.map((step, index) => (
              <div
                key={index}
                className="flex flex-col lg:flex-row items-start lg:items-center gap-8"
              >
                <div className="flex-shrink-0">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full flex items-center justify-center">
                    <span className="text-2xl font-bold text-white">
                      {index + 1}
                    </span>
                  </div>
                </div>
                <div className="flex-grow bg-gray-800/50 border border-gray-700 rounded-xl p-6">
                  <h3 className="text-2xl font-bold text-white mb-3">
                    {step.phase}
                  </h3>
                  <p className="text-lg text-gray-300 mb-3">
                    {step.description}
                  </p>
                  <p className="text-gray-400">
                    {step.details}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      {/* FAQ Section */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Frequently Asked Questions
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Find answers to common questions about our platform, technology, and services.
            </p>
          </div>
          <div className="space-y-6">
            {faqs.map((faq, idx) => (
              <motion.div
                key={idx}
                className={`rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 p-6 shadow-xl glassmorphism hover:shadow-2xl hover:scale-105 hover:border-blue-500/60 transition-all duration-300 group cursor-pointer relative ${openFaq === idx ? 'border-blue-500/60 scale-105 shadow-blue-500/40' : ''}`}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.4, delay: idx * 0.05 }}
                viewport={{ once: true }}
                onClick={() => setOpenFaq(idx)}
              >
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">
                    {faq.question}
                  </h3>
                </div>
              </motion.div>
            ))}
          </div>
          <AnimatePresence>
            {openFaq !== null && (
              <motion.div
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setOpenFaq(null)}
              >
                <motion.div
                  className="rounded-2xl bg-gradient-to-br from-slate-800/90 to-slate-900/90 border border-blue-500/60 p-10 shadow-2xl glassmorphism max-w-xl w-full text-center"
                  initial={{ scale: 0.95, y: 40, opacity: 0 }}
                  animate={{ scale: 1, y: 0, opacity: 1 }}
                  exit={{ scale: 0.95, y: 40, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  onClick={e => e.stopPropagation()}
                >
                  <h3 className="text-2xl font-bold text-white mb-4">{faqs[openFaq].question}</h3>
                  <p className="text-lg text-blue-200 mb-2">{faqs[openFaq].answer}</p>
                  <button
                    className="mt-6 px-6 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold shadow hover:scale-105 transition-transform"
                    onClick={() => setOpenFaq(null)}
                  >
                    Close
                  </button>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {/* Recognition Section */}
      <section className="py-20 bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <Award className="h-16 w-16 text-yellow-400 mx-auto mb-8" />
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Recognition & Partnerships
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Our work has been recognized by leading organizations in the intelligence, 
              technology, and humanitarian sectors.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <motion.div
              whileHover={{ scale: 1.05, boxShadow: '0 8px 32px 0 rgba(99,102,241,0.18)' }}
              transition={{ duration: 0.3 }}
              className="rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 p-6 text-center shadow-xl glassmorphism cursor-pointer hover:shadow-2xl hover:border-blue-500/60"
            >
              <h3 className="text-xl font-bold text-white mb-3">
                UN Partnership
              </h3>
              <p className="text-gray-300">
                Collaborating with UN agencies on crisis monitoring and humanitarian response.
              </p>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05, boxShadow: '0 8px 32px 0 rgba(99,102,241,0.18)' }}
              transition={{ duration: 0.3 }}
              className="rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 p-6 text-center shadow-xl glassmorphism cursor-pointer hover:shadow-2xl hover:border-blue-500/60"
            >
              <h3 className="text-xl font-bold text-white mb-3">
                Academic Network
              </h3>
              <p className="text-gray-300">
                Research partnerships with leading universities studying social movements.
              </p>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05, boxShadow: '0 8px 32px 0 rgba(99,102,241,0.18)' }}
              transition={{ duration: 0.3 }}
              className="rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 p-6 text-center shadow-xl glassmorphism cursor-pointer hover:shadow-2xl hover:border-blue-500/60"
            >
              <h3 className="text-xl font-bold text-white mb-3">
                Tech Innovation Award
              </h3>
              <p className="text-gray-300">
                Recognized for excellence in AI-powered crisis intelligence systems.
              </p>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Team;

