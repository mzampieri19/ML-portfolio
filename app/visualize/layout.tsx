import Link from 'next/link';
import { Brain, ArrowLeft } from 'lucide-react';

interface VisualizationLayoutProps {
  children: React.ReactNode;
}

export default function VisualizationLayout({ children }: VisualizationLayoutProps) {
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
            
            {/* Main Navigation */}
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

            {/* Back to Visualize button for model pages */}
            <Link 
              href="/visualize"
              className="flex items-center space-x-2 px-4 py-2 bg-slate-900 dark:bg-white text-white dark:text-slate-900 rounded-lg hover:bg-slate-700 dark:hover:bg-slate-200 transition-colors md:hidden"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back</span>
            </Link>
          </div>
        </div>
      </nav>

      {/* Visualize Sub-Navigation */}
      <div className="border-b border-slate-200 dark:border-slate-700 bg-white/50 dark:bg-slate-900/50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center space-x-6 h-12 overflow-x-auto">
            <Link 
              href="/visualize" 
              className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors whitespace-nowrap"
            >
              Overview
            </Link>
            <Link 
              href="/visualize/linear-regression" 
              className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors whitespace-nowrap"
            >
              Linear Regression
            </Link>
            <Link 
              href="/visualize/neural-network" 
              className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors whitespace-nowrap"
            >
              Neural Network
            </Link>
            <Link 
              href="/visualize/decision-tree" 
              className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors whitespace-nowrap opacity-50 cursor-not-allowed"
            >
              Decision Tree
            </Link>
            <Link 
              href="/visualize/clustering" 
              className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors whitespace-nowrap opacity-50 cursor-not-allowed"
            >
              Clustering
            </Link>
            <Link 
              href="/visualize/cnn" 
              className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors whitespace-nowrap opacity-50 cursor-not-allowed"
            >
              CNN Builder
            </Link>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main>
        {children}
      </main>
    </div>
  );
}
