import Link from "next/link";
import { Brain, Calendar, Tag } from "lucide-react";

export default function ProjectsPage() {
  const projects = [
    {
      title: "Neural Network Image Classifier",
      description: "Deep learning model for image classification using PyTorch and CNNs",
      tags: ["PyTorch", "Computer Vision", "Deep Learning"],
      slug: "image-classifier",
      date: "2024-12",
      status: "Completed"
    },
    {
      title: "Natural Language Processing",
      description: "Sentiment analysis and text classification using transformers",
      tags: ["NLP", "Transformers", "BERT"],
      slug: "nlp-sentiment",
      date: "2024-11",
      status: "Completed"
    },
    {
      title: "Time Series Forecasting",
      description: "Stock price prediction using LSTM and statistical models",
      tags: ["Time Series", "LSTM", "Finance"],
      slug: "time-series",
      date: "2024-10",
      status: "In Progress"
    },
    {
      title: "Recommendation System",
      description: "Collaborative filtering system for movie recommendations",
      tags: ["Recommender Systems", "Matrix Factorization"],
      slug: "recommendation",
      date: "2024-09",
      status: "Completed"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-2">
              <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </Link>
            <div className="hidden md:flex space-x-8">
              <Link href="/" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Home
              </Link>
              <Link href="/projects" className="text-blue-600 dark:text-blue-400 font-medium">
                Projects
              </Link>
              <Link href="/about" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                About
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-6">
            ML Projects
          </h1>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
            A collection of machine learning projects showcasing various techniques and applications.
          </p>
        </div>

        <div className="grid gap-8">
          {projects.map((project) => (
            <Link 
              key={project.slug}
              href={`/projects/${project.slug}`}
              className="group block"
            >
              <div className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 border border-slate-200 dark:border-slate-700 group-hover:border-blue-300 dark:group-hover:border-blue-500">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
                  <h2 className="text-2xl font-bold text-slate-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors mb-2 md:mb-0">
                    {project.title}
                  </h2>
                  <div className="flex items-center space-x-4 text-sm text-slate-500 dark:text-slate-400">
                    <div className="flex items-center space-x-1">
                      <Calendar className="w-4 h-4" />
                      <span>{project.date}</span>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      project.status === 'Completed' 
                        ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
                        : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300'
                    }`}>
                      {project.status}
                    </span>
                  </div>
                </div>
                
                <p className="text-slate-600 dark:text-slate-300 mb-6 leading-relaxed">
                  {project.description}
                </p>
                
                <div className="flex flex-wrap gap-2">
                  {project.tags.map((tag) => (
                    <span 
                      key={tag}
                      className="flex items-center space-x-1 px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 text-sm font-medium rounded-full"
                    >
                      <Tag className="w-3 h-3" />
                      <span>{tag}</span>
                    </span>
                  ))}
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
