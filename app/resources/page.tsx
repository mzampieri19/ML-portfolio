'use client';

import { useState, useMemo } from 'react';
import { Brain, Search, Filter, Tag, Wrench, ExternalLink, Code, Database, Layers, Zap, Monitor, Globe, Book, Download } from "lucide-react";

interface Resource {
  id: string;
  name: string;
  description: string;
  category: string;
  type: 'Documentation' | 'Download' | 'Tutorial' | 'Tool';
  url: string;
  tags: string[];
  icon: React.ReactNode;
}

const resources: Resource[] = [
  // IDEs & Editors
  {
    id: "vscode",
    name: "Visual Studio Code",
    description: "Free, open-source code editor with excellent ML extensions and debugging capabilities",
    category: "IDEs & Editors",
    type: "Download",
    url: "https://code.visualstudio.com/",
    tags: ["Editor", "Free", "Extensions", "Debugging"],
    icon: <Code className="w-6 h-6" />
  },
  {
    id: "jupyter",
    name: "Jupyter Notebook",
    description: "Interactive computing environment perfect for data science and ML experimentation",
    category: "IDEs & Editors",
    type: "Documentation",
    url: "https://jupyter.org/",
    tags: ["Notebook", "Interactive", "Data Science"],
    icon: <Book className="w-6 h-6" />
  },
  {
    id: "colab",
    name: "Google Colab",
    description: "Free cloud-based Jupyter notebook environment with GPU/TPU access",
    category: "IDEs & Editors",
    type: "Tool",
    url: "https://colab.research.google.com/",
    tags: ["Cloud", "Free", "GPU", "Collaborative"],
    icon: <Globe className="w-6 h-6" />
  },

  // Programming Languages
  {
    id: "python",
    name: "Python",
    description: "The most popular programming language for machine learning and data science",
    category: "Programming Languages",
    type: "Download",
    url: "https://www.python.org/",
    tags: ["Programming", "ML", "Data Science", "Beginner Friendly"],
    icon: <Code className="w-6 h-6" />
  },
  {
    id: "r",
    name: "R",
    description: "Statistical programming language excellent for data analysis and visualization",
    category: "Programming Languages",
    type: "Download",
    url: "https://www.r-project.org/",
    tags: ["Statistics", "Data Analysis", "Visualization"],
    icon: <Code className="w-6 h-6" />
  },
  {
    id: "julia",
    name: "Julia",
    description: "High-performance programming language designed for numerical and scientific computing",
    category: "Programming Languages",
    type: "Download",
    url: "https://julialang.org/",
    tags: ["High Performance", "Scientific Computing", "Numerical"],
    icon: <Zap className="w-6 h-6" />
  },

  // ML Libraries
  {
    id: "pytorch",
    name: "PyTorch",
    description: "Dynamic neural network library with excellent research flexibility and production support",
    category: "ML Libraries",
    type: "Documentation",
    url: "https://pytorch.org/",
    tags: ["Deep Learning", "Neural Networks", "Research", "Production"],
    icon: <Layers className="w-6 h-6" />
  },
  {
    id: "tensorflow",
    name: "TensorFlow",
    description: "End-to-end ML platform with comprehensive tools for building and deploying models",
    category: "ML Libraries",
    type: "Documentation",
    url: "https://www.tensorflow.org/",
    tags: ["Deep Learning", "Production", "Deployment", "Comprehensive"],
    icon: <Layers className="w-6 h-6" />
  },
  {
    id: "scikit-learn",
    name: "Scikit-learn",
    description: "Simple and efficient tools for machine learning built on NumPy, SciPy, and matplotlib",
    category: "ML Libraries",
    type: "Documentation",
    url: "https://scikit-learn.org/",
    tags: ["Traditional ML", "Beginner Friendly", "Comprehensive"],
    icon: <Wrench className="w-6 h-6" />
  },
  {
    id: "huggingface",
    name: "Hugging Face",
    description: "Platform for sharing and using pre-trained models, especially for NLP",
    category: "ML Libraries",
    type: "Documentation",
    url: "https://huggingface.co/",
    tags: ["Pre-trained Models", "NLP", "Transformers", "Community"],
    icon: <Brain className="w-6 h-6" />
  },

  // Data Science Libraries
  {
    id: "pandas",
    name: "Pandas",
    description: "Powerful data manipulation and analysis library for Python",
    category: "Data Science Libraries",
    type: "Documentation",
    url: "https://pandas.pydata.org/",
    tags: ["Data Manipulation", "DataFrames", "Analysis"],
    icon: <Database className="w-6 h-6" />
  },
  {
    id: "numpy",
    name: "NumPy",
    description: "Fundamental package for scientific computing with Python",
    category: "Data Science Libraries",
    type: "Documentation",
    url: "https://numpy.org/",
    tags: ["Numerical Computing", "Arrays", "Scientific"],
    icon: <Database className="w-6 h-6" />
  },
  {
    id: "matplotlib",
    name: "Matplotlib",
    description: "Comprehensive library for creating static, animated, and interactive visualizations",
    category: "Data Science Libraries",
    type: "Documentation",
    url: "https://matplotlib.org/",
    tags: ["Visualization", "Plotting", "Charts"],
    icon: <Monitor className="w-6 h-6" />
  },
  {
    id: "seaborn",
    name: "Seaborn",
    description: "Statistical data visualization library based on matplotlib",
    category: "Data Science Libraries",
    type: "Documentation",
    url: "https://seaborn.pydata.org/",
    tags: ["Visualization", "Statistical", "Beautiful"],
    icon: <Monitor className="w-6 h-6" />
  },

  // Cloud Platforms
  {
    id: "aws-sagemaker",
    name: "AWS SageMaker",
    description: "Fully managed service to build, train, and deploy ML models at scale",
    category: "Cloud Platforms",
    type: "Documentation",
    url: "https://aws.amazon.com/sagemaker/",
    tags: ["Cloud", "Managed", "Scalable", "Production"],
    icon: <Globe className="w-6 h-6" />
  },
  {
    id: "gcp-ai",
    name: "Google Cloud AI",
    description: "Suite of ML tools and services for building and deploying AI applications",
    category: "Cloud Platforms",
    type: "Documentation",
    url: "https://cloud.google.com/ai",
    tags: ["Cloud", "AI Services", "AutoML", "Production"],
    icon: <Globe className="w-6 h-6" />
  },
  {
    id: "azure-ml",
    name: "Azure Machine Learning",
    description: "Cloud service for accelerating and managing the ML project lifecycle",
    category: "Cloud Platforms",
    type: "Documentation",
    url: "https://azure.microsoft.com/en-us/services/machine-learning/",
    tags: ["Cloud", "MLOps", "Enterprise", "Lifecycle"],
    icon: <Globe className="w-6 h-6" />
  },

  // Version Control & Deployment
  {
    id: "git",
    name: "Git",
    description: "Distributed version control system for tracking changes in source code",
    category: "Version Control & Deployment",
    type: "Download",
    url: "https://git-scm.com/",
    tags: ["Version Control", "Collaboration", "Open Source"],
    icon: <Code className="w-6 h-6" />
  },
  {
    id: "docker",
    name: "Docker",
    description: "Platform for developing, shipping, and running applications in containers",
    category: "Version Control & Deployment",
    type: "Download",
    url: "https://www.docker.com/",
    tags: ["Containerization", "Deployment", "DevOps"],
    icon: <Layers className="w-6 h-6" />
  },
  {
    id: "mlflow",
    name: "MLflow",
    description: "Open source platform for managing the ML lifecycle, including tracking and deployment",
    category: "Version Control & Deployment",
    type: "Documentation",
    url: "https://mlflow.org/",
    tags: ["MLOps", "Tracking", "Deployment", "Open Source"],
    icon: <Wrench className="w-6 h-6" />
  },

  // Learning Resources
  {
    id: "coursera-ml",
    name: "Coursera ML Course",
    description: "Andrew Ng's famous Machine Learning course - perfect for beginners",
    category: "Learning Resources",
    type: "Tutorial",
    url: "https://www.coursera.org/learn/machine-learning",
    tags: ["Course", "Beginner", "Andrew Ng", "Comprehensive"],
    icon: <Book className="w-6 h-6" />
  },
  {
    id: "fast-ai",
    name: "Fast.ai",
    description: "Practical deep learning course and library focusing on real-world applications",
    category: "Learning Resources",
    type: "Tutorial",
    url: "https://www.fast.ai/",
    tags: ["Deep Learning", "Practical", "Top-down", "Free"],
    icon: <Book className="w-6 h-6" />
  },
  {
    id: "kaggle-learn",
    name: "Kaggle Learn",
    description: "Free micro-courses on data science and machine learning topics",
    category: "Learning Resources",
    type: "Tutorial",
    url: "https://www.kaggle.com/learn",
    tags: ["Free", "Practical", "Short Courses", "Hands-on"],
    icon: <Book className="w-6 h-6" />
  },

  // YouTube Resources
  {
    id: "3blue1brown",
    name: "3Blue1Brown - Neural Networks",
    description: "Beautiful visual explanations of neural networks and deep learning concepts",
    category: "YouTube Channels",
    type: "Tutorial",
    url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
    tags: ["Visual Learning", "Deep Learning", "Mathematics", "Intuitive"],
    icon: <Monitor className="w-6 h-6" />
  },
  {
    id: "two-minute-papers",
    name: "Two Minute Papers",
    description: "Latest AI research papers explained in an accessible and exciting way",
    category: "YouTube Channels",
    type: "Tutorial",
    url: "https://www.youtube.com/c/K%C3%A1rolyZsolnai",
    tags: ["Research", "AI News", "Paper Reviews", "Cutting-edge"],
    icon: <Monitor className="w-6 h-6" />
  },
  {
    id: "sentdex",
    name: "Sentdex",
    description: "Practical Python programming tutorials with focus on ML, data analysis, and web development",
    category: "YouTube Channels",
    type: "Tutorial",
    url: "https://www.youtube.com/c/sentdex",
    tags: ["Python", "Practical", "Tutorials", "Hands-on"],
    icon: <Monitor className="w-6 h-6" />
  },
  {
    id: "yannic-kilcher",
    name: "Yannic Kilcher",
    description: "In-depth paper reviews and discussions about latest ML research developments",
    category: "YouTube Channels",
    type: "Tutorial",
    url: "https://www.youtube.com/c/YannicKilcher",
    tags: ["Research Papers", "Deep Dives", "Academic", "Analysis"],
    icon: <Monitor className="w-6 h-6" />
  },
  {
    id: "ai-explained",
    name: "AI Explained",
    description: "Clear explanations of AI concepts, research papers, and industry developments",
    category: "YouTube Channels",
    type: "Tutorial",
    url: "https://www.youtube.com/@aiexplained-official",
    tags: ["AI Concepts", "Explanations", "Industry News", "Accessible"],
    icon: <Monitor className="w-6 h-6" />
  },
  {
    id: "deeplearning-ai",
    name: "DeepLearning.AI",
    description: "Andrew Ng's channel with courses and insights on deep learning and AI",
    category: "YouTube Channels",
    type: "Tutorial",
    url: "https://www.youtube.com/c/Deeplearningai",
    tags: ["Andrew Ng", "Courses", "Deep Learning", "Structured Learning"],
    icon: <Monitor className="w-6 h-6" />
  },
  {
    id: "custom-diffusion-from-scratch",
    name: "Coding Stable Diffusion from Scratch",
    description: "A comprehensive guide to building a custom diffusion model from scratch, including encoding, decoding, UNET, DDPM, and Diffusion techniques.",
    category: "YouTube Channels",
    type: "Tutorial",
    url: "https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=2246s",
    tags: ["Diffusion", "From Scratch", "Neural Network", "GenAI"],
    icon: <Monitor className="w-6 h-6" />
  },
  {
    id: "LLM-from-scratch",
    name: "Create a Large Language Model from Scratch with Python",
    description: "A comprehensive guide to building a custom large language model from scratch, including data collection, preprocessing, model architecture, and training techniques.",
    category: "YouTube Channels",
    type: "Tutorial",
    url: "https://www.youtube.com/watch?v=UU1WVnMk4E8&t=6402s",
    tags: ["LLM", "From Scratch", "Neural Network", "GenAI"],
    icon: <Monitor className="w-6 h-6" />
  },
{
    id: "RL-from-scratch",
    name: "Implement Deep Q-Learning with PyTorch",
    description: "A comprehensive guide to building a custom reinforcement learning model from scratch, including environment setup, algorithm implementation, and training techniques.",
    category: "YouTube Channels",
    type: "Tutorial",
    url: "https://www.youtube.com/watch?v=arR7KzlYs4w&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi",
    tags: ["RL", "From Scratch", "Neural Network", "GenAI"],
    icon: <Monitor className="w-6 h-6" />
  }
];

const categories = [
  "All Categories",
  "IDEs & Editors",
  "Programming Languages",
  "ML Libraries",
  "Data Science Libraries",
  "Cloud Platforms",
  "Version Control & Deployment",
  "Learning Resources",
  "YouTube Channels"
];

const types = [
  "All Types",
  "Documentation",
  "Download",
  "Tutorial",
  "Tool"
];

const typeColors = {
  "Documentation": "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
  "Download": "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  "Tutorial": "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
  "Tool": "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300"
};

export default function ResourcesPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All Categories");
  const [selectedType, setSelectedType] = useState("All Types");

  const filteredResources = useMemo(() => {
    return resources.filter(resource => {
      const matchesSearch = resource.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           resource.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           resource.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesCategory = selectedCategory === "All Categories" || resource.category === selectedCategory;
      const matchesType = selectedType === "All Types" || resource.type === selectedType;
      
      return matchesSearch && matchesCategory && matchesType;
    });
  }, [searchTerm, selectedCategory, selectedType]);

  const sortedResources = useMemo(() => {
    return [...filteredResources].sort((a, b) => a.name.localeCompare(b.name));
  }, [filteredResources]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Projects
              </a>
              <a href="/topics" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Topics
              </a>
              <a href="/visualize" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Visualize
              </a>
              <a href="/resources" className="text-blue-600 dark:text-blue-400 font-medium">
                Resources
              </a>
              <a href="/about" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                About
              </a>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <Wrench className="w-12 h-12 text-blue-600 dark:text-blue-400" />
            <h1 className="text-4xl font-bold text-slate-900 dark:text-white">
              ML Resources
            </h1>
          </div>
          <p className="text-xl text-slate-700 dark:text-slate-300 max-w-3xl mx-auto leading-relaxed">
            Essential tools, libraries, and learning resources for machine learning and data science. 
            Find documentation, downloads, and tutorials for all the technologies you need.
          </p>

          <div className="flex justify-center mt-6">
            <p className="text-slate-600 dark:text-slate-400">
              Curated collection of the most useful resources for ML practitioners at all levels.
            </p>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-6 mb-12 shadow-sm">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search resources..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
              />
            </div>

            {/* Category Filter */}
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors appearance-none"
              >
                {categories.map(category => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </div>

            {/* Type Filter */}
            <div className="relative">
              <Tag className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors appearance-none"
              >
                {types.map(type => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Results count */}
          <div className="mt-4 text-sm text-slate-600 dark:text-slate-400">
            {sortedResources.length} resource{sortedResources.length !== 1 ? 's' : ''} found
          </div>
        </div>

        {/* Resources Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {sortedResources.map((resource) => (
            <a 
              key={resource.id} 
              href={resource.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group block"
            >
              <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-6 hover:shadow-lg hover:border-blue-300 dark:hover:border-blue-600 transition-all duration-300 group-hover:-translate-y-1">
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                      {resource.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-slate-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                        {resource.name}
                      </h3>
                      <p className="text-sm text-slate-600 dark:text-slate-400">
                        {resource.category}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 text-xs font-medium rounded-lg ${typeColors[resource.type]}`}>
                      {resource.type}
                    </span>
                    <ExternalLink className="w-4 h-4 text-slate-400 group-hover:text-blue-500 transition-colors" />
                  </div>
                </div>

                {/* Description */}
                <p className="text-slate-700 dark:text-slate-300 text-sm leading-relaxed mb-4">
                  {resource.description}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-2">
                  {resource.tags.map((tag) => (
                    <span 
                      key={tag}
                      className="px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs rounded-lg"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </a>
          ))}
        </div>

        {/* No results */}
        {sortedResources.length === 0 && (
          <div className="text-center py-16">
            <Wrench className="w-16 h-16 text-slate-400 dark:text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              No resources found
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              Try adjusting your search terms or filters
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
