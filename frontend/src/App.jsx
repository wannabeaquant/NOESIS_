import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Layout from './components/Layout';
import Home from './pages/Home';
import About from './pages/About';
import Dashboard from './pages/Dashboard';
import PredictiveDashboard from './pages/PredictiveDashboard';
import Contact from './pages/Contact';
import Team from './pages/Team';
import './App.css';

// Page transition component
const PageTransition = ({ children }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5, ease: "easeInOut" }}
    >
      {children}
    </motion.div>
  );
};

// Animated routes wrapper
const AnimatedRoutes = () => {
  const location = useLocation();
  
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={
          <PageTransition>
            <Home />
          </PageTransition>
        } />
        <Route path="/about" element={
          <PageTransition>
            <About />
          </PageTransition>
        } />
        <Route path="/dashboard" element={
          <PageTransition>
            <Dashboard />
          </PageTransition>
        } />
        <Route path="/predictive" element={
          <PageTransition>
            <PredictiveDashboard />
          </PageTransition>
        } />
        <Route path="/contact" element={
          <PageTransition>
            <Contact />
          </PageTransition>
        } />
        <Route path="/team" element={
          <PageTransition>
            <Team />
          </PageTransition>
        } />
      </Routes>
    </AnimatePresence>
  );
};

function App() {
  return (
    <Router>
      <Layout>
        <AnimatedRoutes />
      </Layout>
    </Router>
  );
}

export default App;

