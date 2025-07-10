import Link from 'next/link';
import { ArrowRight, Brain, BarChart3, GitBranch, Layers, Sparkles, Zap, Eye, TrendingUp } from 'lucide-react';

export const metadata = {
  title: 'Interactive Model Visualization - ML Portfolio',
  description: 'Experiment with ML models and see how parameters affect performance in real-time',
};

export default function VisualizePage() {
  const models = [
    {
      id: 'neural-network',
      title: 'Neural Network Playground',
      description: 'Build and train neural networks with interactive architecture design',
      icon: Brain,
      features: ['Drag & Drop Layers', 'Real-time Training', 'Decision Boundaries'],
      color: 'bg-purple-500',
      available: true
    },
    {
      id: 'linear-regression',
      title: 'Linear Regression Explorer',
      description: 'Experiment with polynomial regression and regularization techniques',
      icon: BarChart3,
      features: ['Polynomial Fitting', 'Regularization', 'Residual Analysis'],
      color: 'bg-blue-500',
      available: true
    },
    {
      id: 'decision-tree',
      title: 'Decision Tree Visualizer',
      description: 'Visualize how decision trees split data and make predictions',
      icon: GitBranch,
      features: ['Interactive Tree', 'Pruning Effects', 'Feature Importance'],
      color: 'bg-green-500',
      available: false
    },
    {
      id: 'clustering',
      title: 'Clustering Sandbox',
      description: 'Explore K-means, DBSCAN, and hierarchical clustering algorithms',
      icon: Sparkles,
      features: ['Multiple Algorithms', 'Elbow Method', 'Silhouette Analysis'],
      color: 'bg-orange-500',
      available: false
    },
    {
      id: 'cnn',
      title: 'CNN Architecture Builder',
      description: 'Design convolutional neural networks for image processing',
      icon: Layers,
      features: ['3D Visualization', 'Filter Exploration', 'Feature Maps'],
      color: 'bg-red-500',
      available: false
    }
  ];

  const features = [
    {
      icon: Zap,
      title: 'Real-time Updates',
      description: 'See how parameter changes instantly affect model performance and visualizations'
    },
    {
      icon: Brain,
      title: 'Interactive Learning',
      description: 'Hands-on experimentation with machine learning concepts and algorithms'
    },
    {
      icon: BarChart3,
      title: 'Performance Metrics',
      description: 'Track accuracy, loss, and other metrics as your models train and evolve'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-purple-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-2">
              <Brain className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </Link>
            <div className="hidden md:flex space-x-8">
              <Link href="/" className="text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors">
                Home
              </Link>
              <Link href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors">
                Projects
              </Link>
              <Link href="/topics" className="text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors">
                Topics
              </Link>
              <Link href="/visualize" className="text-purple-600 dark:text-purple-400 font-medium">
                Visualize
              </Link>
              <Link href="/resources" className="text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors">
                Resources
              </Link>
              <Link href="/about" className="text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors">
                About
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Hero Section */}
        <div className="text-center py-16 lg:py-24">
          <div className="inline-flex items-center gap-2 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 px-4 py-2 rounded-full text-sm font-medium mb-8">
            <Sparkles className="w-4 h-4" />
            Interactive Machine Learning Playground
          </div>
          
          <div className="flex items-center justify-center space-x-3 mb-6">
            <Eye className="w-12 h-12 text-purple-600 dark:text-purple-400" />
            <h1 className="text-4xl md:text-6xl font-bold text-slate-900 dark:text-white">
              Interactive Model{' '}
              <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                Visualization
              </span>
            </h1>
          </div>
          
          <p className="text-xl md:text-2xl text-slate-700 dark:text-slate-300 max-w-4xl mx-auto leading-relaxed mb-12">
            Experience machine learning through interactive visualizations. Adjust parameters, 
            see real-time results, and understand how different algorithms work under the hood.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
            <Link 
              href="#models"
              className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-xl text-lg font-semibold transition-all duration-200 transform hover:scale-105"
            >
              <TrendingUp className="w-5 h-5 mr-2" />
              Explore Models
              <ArrowRight className="w-5 h-5 ml-2" />
            </Link>
            <Link 
              href="/topics"
              className="border-2 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-200 hover:border-slate-400 dark:hover:border-slate-500"
            >
              Learn Concepts
            </Link>
          </div>
        </div>

        {/* Features Section */}
        <div className="py-16">
          <h2 className="text-3xl md:text-4xl font-bold text-center text-slate-900 dark:text-white mb-12">
            Why Interactive Learning?
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="text-center p-6">
                <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-purple-500 to-blue-500 rounded-2xl mb-6">
                  <feature.icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
                  {feature.title}
                </h3>
                <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Models Section */}
        <div id="models" className="py-16">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white mb-6">
              Interactive Model Playground
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
              Choose from our collection of interactive machine learning models. 
              Each one offers unique insights into different aspects of ML.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-8">
            {models.map((model) => (
              <div 
                key={model.id}
                className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 border border-slate-100 dark:border-slate-700"
              >
                <div className="flex items-center gap-4 mb-6">
                  <div className={`${model.color} p-3 rounded-xl`}>
                    <model.icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                      {model.title}
                    </h3>
                    {!model.available && (
                      <span className="inline-block bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 text-xs px-2 py-1 rounded-full mt-1">
                        Coming Soon
                      </span>
                    )}
                  </div>
                </div>
                
                <p className="text-slate-600 dark:text-slate-300 mb-6 leading-relaxed">
                  {model.description}
                </p>
                
                <div className="space-y-2 mb-8">
                  {model.features.map((feature, index) => (
                    <div key={index} className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
                      <div className="w-1.5 h-1.5 bg-purple-500 rounded-full"></div>
                      {feature}
                    </div>
                  ))}
                </div>
                
                {model.available ? (
                  <Link
                    href={`/visualize/${model.id}`}
                    className="block text-center py-3 px-6 rounded-xl font-semibold transition-all duration-200 bg-slate-900 dark:bg-white text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-100"
                  >
                    Explore
                  </Link>
                ) : (
                  <div className="block text-center py-3 px-6 rounded-xl font-semibold transition-all duration-200 bg-slate-100 dark:bg-slate-700 text-slate-400 dark:text-slate-500 cursor-not-allowed">
                    Coming Soon
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Development Status */}
        <div className="py-16">
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-slate-800 dark:to-slate-700 rounded-2xl p-8 text-center border border-purple-100 dark:border-slate-600">
            <div className="flex justify-center mb-6">
              <div className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-full">
                <Zap className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              </div>
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-slate-900 dark:text-white mb-4">
              🚧 Under Active Development
            </h2>
            <p className="text-lg text-slate-600 dark:text-slate-300 mb-6 max-w-2xl mx-auto">
              We're building these interactive visualization tools to make machine learning 
              more accessible and engaging. Each model will offer hands-on experimentation 
              with real-time feedback.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link 
                href="/projects"
                className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors"
              >
                View My Projects
              </Link>
              <Link 
                href="/topics"
                className="border border-slate-300 dark:border-slate-500 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-600 px-6 py-3 rounded-lg font-semibold transition-colors"
              >
                Learn ML Concepts
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
