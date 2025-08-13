import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Github, Twitter, Linkedin, Mail, Globe, Shield, Heart } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const socialLinks = [
    {
      name: 'GitHub',
      icon: <Github className="h-5 w-5" />,
      href: 'https://github.com/protest-monitor',
      color: 'hover:text-gray-400'
    },
    {
      name: 'Twitter',
      icon: <Twitter className="h-5 w-5" />,
      href: 'https://twitter.com/protestmonitor',
      color: 'hover:text-blue-400'
    },
    {
      name: 'LinkedIn',
      icon: <Linkedin className="h-5 w-5" />,
      href: 'https://linkedin.com/company/protest-monitor',
      color: 'hover:text-blue-600'
    },
    {
      name: 'Email',
      icon: <Mail className="h-5 w-5" />,
      href: 'mailto:contact@protestmonitor.ai',
      color: 'hover:text-red-400'
    }
  ];

  const quickLinks = [
    { name: 'About', href: '/about' },
    { name: 'Dashboard', href: '/dashboard' },
    { name: 'Contact', href: '/contact' },
    { name: 'Team', href: '/team' }
  ];

  return (
    <footer className="relative z-20 bg-slate-900/80 backdrop-blur-sm border-t border-slate-800/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="col-span-1 md:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
            >
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg">
                  <Globe className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white">Protest Monitor</h3>
                  <p className="text-slate-400 text-sm">OSINT Intelligence Platform</p>
                </div>
              </div>
              
              <p className="text-slate-400 mb-6 leading-relaxed max-w-md">
                Advanced AI-powered monitoring system that transforms chaotic protest data 
                into actionable intelligence for governments, organizations, and researchers worldwide.
              </p>
              
              <div className="flex space-x-4">
                {socialLinks.map((social, index) => (
                  <motion.a
                    key={social.name}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`p-2 bg-slate-800/50 rounded-lg text-slate-400 transition-all duration-300 ${social.color} hover:bg-slate-700/50 hover:scale-110`}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    whileHover={{ y: -2 }}
                  >
                    {social.icon}
                  </motion.a>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Quick Links */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <h4 className="text-lg font-semibold text-white mb-6">Quick Links</h4>
              <ul className="space-y-3">
                {quickLinks.map((link, index) => (
                  <motion.li
                    key={link.name}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6, delay: 0.3 + index * 0.1 }}
                    viewport={{ once: true }}
                  >
                    <Link
                      to={link.href}
                      className="text-slate-400 hover:text-white transition-colors duration-300 flex items-center group"
                    >
                      <span className="w-2 h-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full mr-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
                      {link.name}
                    </Link>
                  </motion.li>
                ))}
              </ul>
            </motion.div>
          </div>

          {/* Resources */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              viewport={{ once: true }}
            >
              <h4 className="text-lg font-semibold text-white mb-6">Resources</h4>
              <ul className="space-y-3">
                <motion.li
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 0.5 }}
                  viewport={{ once: true }}
                >
                  <a
                    href="https://github.com/protest-monitor/docs"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-slate-400 hover:text-white transition-colors duration-300 flex items-center group"
                  >
                    <span className="w-2 h-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full mr-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
                    Documentation
                  </a>
                </motion.li>
                <motion.li
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 0.6 }}
                  viewport={{ once: true }}
                >
                  <a
                    href="https://github.com/protest-monitor/api"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-slate-400 hover:text-white transition-colors duration-300 flex items-center group"
                  >
                    <span className="w-2 h-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full mr-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
                    API Reference
                  </a>
                </motion.li>
                <motion.li
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 0.7 }}
                  viewport={{ once: true }}
                >
                  <a
                    href="/privacy"
                    className="text-slate-400 hover:text-white transition-colors duration-300 flex items-center group"
                  >
                    <span className="w-2 h-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full mr-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
                    Privacy Policy
                  </a>
                </motion.li>
                <motion.li
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 0.8 }}
                  viewport={{ once: true }}
                >
                  <a
                    href="/terms"
                    className="text-slate-400 hover:text-white transition-colors duration-300 flex items-center group"
                  >
                    <span className="w-2 h-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full mr-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
                    Terms of Service
                  </a>
                </motion.li>
              </ul>
            </motion.div>
          </div>
        </div>

        {/* Bottom Section */}
        <motion.div
          className="border-t border-slate-800/50 mt-12 pt-8"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          viewport={{ once: true }}
        >
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-2 text-slate-400 text-sm">
              <Shield className="h-4 w-4" />
              <span>© {currentYear} Protest Monitor. All rights reserved.</span>
            </div>
            
            <div className="flex items-center space-x-4 text-slate-400 text-sm">
              <span className="flex items-center">
                Made with <Heart className="h-4 w-4 mx-1 text-red-400" /> by the Cache Money Team
              </span>
            </div>
            
            <div className="text-slate-500 text-xs text-center md:text-right max-w-md">
              <p>
                This platform is designed for research and monitoring purposes. 
                Data is collected from publicly available sources and processed 
                using AI algorithms for analysis and verification.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </footer>
  );
};

export default Footer;

