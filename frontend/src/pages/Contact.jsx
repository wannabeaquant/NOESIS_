import { useState } from 'react';
import { Mail, Phone, MapPin, Send, CheckCircle, Globe, MessageSquare } from 'lucide-react';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { motion } from 'framer-motion';
import FloatingButton from '../components/FloatingButton';
import DynamicBackground from '../components/DynamicBackground';

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    organization: '',
    subject: '',
    message: ''
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);

    // Simulate form submission
    setTimeout(() => {
      console.log('Form submitted:', formData);
      toast.success('Message sent successfully! We\'ll get back to you within 24 hours.', {
        position: "top-right",
        autoClose: 5000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        theme: "dark"
      });
      
      // Reset form
      setFormData({
        name: '',
        email: '',
        organization: '',
        subject: '',
        message: ''
      });
      setIsSubmitting(false);
    }, 1500);
  };

  // Add subscribe handler
  const [newsletterEmail, setNewsletterEmail] = useState("");
  const handleSubscribe = (e) => {
    e.preventDefault();
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!newsletterEmail.trim() || !emailPattern.test(newsletterEmail)) {
      toast.error('Please provide an Email !', {
        position: "top-center",
        autoClose: 4000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        theme: "colored",
        style: {
          background: 'linear-gradient(90deg, #ef4444 0%, #a21caf 100%)',
          color: '#fff',
          fontWeight: 'bold',
          fontSize: '1.1rem',
          borderRadius: '1rem',
          boxShadow: '0 8px 32px 0 rgba(239,68,68,0.18)'
        },
        icon: '⚠️'
      });
      return;
    }
    toast.success('Thank you for subscribing !! You will be notified on your given Email', {
      position: "top-center",
      autoClose: 4000,
      hideProgressBar: false,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
      theme: "colored",
      style: {
        background: 'linear-gradient(90deg, #6366f1 0%, #a21caf 100%)',
        color: '#fff',
        fontWeight: 'bold',
        fontSize: '1.1rem',
        borderRadius: '1rem',
        boxShadow: '0 8px 32px 0 rgba(99,102,241,0.18)'
      },
      icon: '🎉'
    });
  };

  const contactInfo = [
    {
      icon: <Mail className="h-6 w-6 text-blue-400" />,
      title: "Email",
      details: "contact@protestmonitor.com",
      description: "Send us an email anytime"
    },
    {
      icon: <Phone className="h-6 w-6 text-green-400" />,
      title: "Phone",
      details: "+1 (555) 123-4567",
      description: "Mon-Fri from 8am to 6pm EST"
    },
    {
      icon: <MapPin className="h-6 w-6 text-purple-400" />,
      title: "Office",
      details: "San Francisco, CA",
      description: "Global headquarters"
    },
    {
      icon: <MessageSquare className="h-6 w-6 text-yellow-400" />,
      title: "Telegram",
      details: "@ProtestMonitorBot",
      description: "Real-time alerts and updates"
    }
  ];

  return (
    <div className="bg-slate-950 min-h-screen font-['Inter'] relative overflow-hidden">
      <DynamicBackground />
      <ToastContainer />
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 z-10">
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <Globe className="h-16 w-16 text-blue-400 mx-auto mb-8" />
            <motion.h1
              className="text-4xl md:text-5xl lg:text-6xl font-extrabold mb-2 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              Get In Touch
            </motion.h1>
            <motion.p
              className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              Ready to transform your approach to global intelligence? Let's discuss how our platform can serve your needs.
            </motion.p>
          </div>
        </div>
      </section>
      {/* Contact Form & Info */}
      <section className="py-20 z-10 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* Contact Form */}
            <motion.div
              className="rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 p-10 shadow-xl glassmorphism hover:shadow-2xl transition-all duration-300"
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              viewport={{ once: true }}
            >
              <h2 className="text-3xl font-extrabold mb-6 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">Send us a message</h2>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">
                      Full Name *
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      required
                      value={formData.name}
                      onChange={handleChange}
                      className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="John Doe"
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                      Email Address *
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      required
                      value={formData.email}
                      onChange={handleChange}
                      className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="john@example.com"
                    />
                  </div>
                </div>
                
                <div>
                  <label htmlFor="organization" className="block text-sm font-medium text-gray-300 mb-2">
                    Organization
                  </label>
                  <input
                    type="text"
                    id="organization"
                    name="organization"
                    value={formData.organization}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Your organization"
                  />
                </div>

                <div>
                  <label htmlFor="subject" className="block text-sm font-medium text-gray-300 mb-2">
                    Subject *
                  </label>
                  <select
                    id="subject"
                    name="subject"
                    required
                    value={formData.subject}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Select a subject</option>
                    <option value="demo">Request a Demo</option>
                    <option value="pricing">Pricing Information</option>
                    <option value="partnership">Partnership Opportunities</option>
                    <option value="technical">Technical Support</option>
                    <option value="media">Media Inquiry</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="message" className="block text-sm font-medium text-gray-300 mb-2">
                    Message *
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    required
                    rows={6}
                    value={formData.message}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    placeholder="Tell us about your needs and how we can help..."
                  />
                </div>

                <FloatingButton
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full mt-2"
                  size="lg"
                  icon={<Send className="h-5 w-5" />}
                  expandedContent={<div className="text-left text-white"><b>We respond within 24 hours!</b></div>}
                >
                  {isSubmitting ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Sending...
                    </>
                  ) : (
                    <>Send Message</>
                  )}
                </FloatingButton>
              </form>
            </motion.div>
            {/* Contact Information */}
            <div className="space-y-8">
              <motion.div
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                viewport={{ once: true }}
              >
                <h2 className="text-3xl font-extrabold mb-6 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">Contact Information</h2>
                <p className="text-lg text-gray-300 mb-8">
                  We're here to help you harness the power of global intelligence. Reach out through any of these channels and we'll respond promptly.
                </p>
              </motion.div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {contactInfo.map((info, index) => (
                  <motion.div
                    key={index}
                    className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 rounded-2xl p-6 shadow-xl glassmorphism hover:shadow-2xl hover:scale-105 hover:border-blue-500/60 transition-all duration-300 group cursor-pointer relative"
                    initial={{ opacity: 0, scale: 0.95, y: 20 }}
                    whileInView={{ opacity: 1, scale: 1, y: 0 }}
                    whileHover={{ scale: 1.05 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    viewport={{ once: true }}
                  >
                    <div className="flex items-center space-x-3 mb-3">
                      {info.icon}
                      <h3 className="text-xl font-semibold text-white">
                        {info.title}
                      </h3>
                    </div>
                    <p className="text-lg font-medium text-gray-200 mb-1">
                      {info.details}
                    </p>
                    <p className="text-sm text-gray-400">
                      {info.description}
                    </p>
                  </motion.div>
                ))}
              </div>
              {/* Newsletter Signup */}
              <motion.div
                className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 border border-gray-700 rounded-2xl p-8 shadow-xl glassmorphism"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                viewport={{ once: true }}
              >
                <h3 className="text-2xl font-extrabold mb-4 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
                  Stay Updated
                </h3>
                <p className="text-gray-300 mb-6">
                  Subscribe to our newsletter for the latest updates on global protest activities and platform enhancements.
                </p>
                <div className="flex flex-col sm:flex-row gap-4">
                  <input
                    type="email"
                    placeholder="Enter your email"
                    className="flex-1 px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={newsletterEmail}
                    onChange={e => setNewsletterEmail(e.target.value)}
                  />
                  <FloatingButton
                    type="button"
                    size="md"
                    className="w-full sm:w-auto"
                    variant="primary"
                    expandedContent={<div className="text-white">Get news, no spam!</div>}
                    onClick={handleSubscribe}
                  >
                    Subscribe
                  </FloatingButton>
                </div>
              </motion.div>
              {/* Response Time */}
              <motion.div
                className="bg-gradient-to-br from-green-900/20 to-slate-900/80 border border-green-800/50 rounded-2xl p-6 shadow-xl glassmorphism hover:shadow-2xl hover:scale-105 hover:border-green-500/60 transition-all duration-300 group cursor-pointer"
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                whileHover={{ scale: 1.05 }}
                transition={{ duration: 0.5 }}
                viewport={{ once: true }}
              >
                <div className="flex items-center space-x-3 mb-3">
                  <CheckCircle className="h-6 w-6 text-green-400" />
                  <h3 className="text-xl font-semibold text-white">
                    Quick Response Guarantee
                  </h3>
                </div>
                <p className="text-gray-300">
                  We typically respond to all inquiries within 24 hours during business days. For urgent matters, please call our direct line.
                </p>
              </motion.div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Contact;

