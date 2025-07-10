'use client';

import { useState, useMemo } from 'react';
import Link from "next/link";
import { Brain, Search, Filter, Tag, Book, Code, Image, FileText, List, Hash, ArrowRight, ArrowDown, ChevronRight } from "lucide-react";

interface Topic {
  id: string;
  title: string;
  description: string;
  category: string;
  projects: string[];
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  tags: string[];
}

const topics: Topic[] = [
  {
    id: "convolutional-neural-networks",
    title: "Convolutional Neural Networks (CNNs)",
    description: "Deep learning architecture designed for processing grid-like data such as images",
    category: "Deep Learning",
    projects: ["image-classifier", "tamid-image-classifier"],
    difficulty: "Intermediate",
    tags: ["Computer Vision", "Deep Learning", "Image Processing"]
  },
  {
    id: "reinforcement-learning",
    title: "Reinforcement Learning",
    description: "Machine learning paradigm where agents learn through interaction with an environment",
    category: "Machine Learning",
    projects: ["dqn-flappy-bird"],
    difficulty: "Advanced",
    tags: ["RL", "Gaming", "Neural Networks"]
  },
  {
    id: "clustering-algorithms",
    title: "Clustering Algorithms",
    description: "Unsupervised learning techniques for grouping similar data points",
    category: "Machine Learning",
    projects: ["clustering-exploration"],
    difficulty: "Beginner",
    tags: ["Unsupervised Learning", "K-means", "TSNE"]
  },
  {
    id: "diffusion-models",
    title: "Diffusion Models",
    description: "Generative models that learn to reverse a gradual noising process",
    category: "Generative AI",
    projects: ["custom-diffusion-model"],
    difficulty: "Advanced",
    tags: ["Generative AI", "Image Generation", "DDPM"]
  },
  {
    id: "large-language-models",
    title: "Large Language Models (LLMs)",
    description: "Neural networks trained on vast amounts of text data to understand and generate language",
    category: "Natural Language Processing",
    projects: ["custom-gpt-llm", "real-salary"],
    difficulty: "Advanced",
    tags: ["NLP", "GPT", "Transformers"]
  },
  {
    id: "data-preprocessing",
    title: "Data Preprocessing",
    description: "Techniques for cleaning, transforming, and preparing data for machine learning",
    category: "Data Science",
    projects: ["image-classifier", "tamid-image-classifier", "real-salary", "clustering-exploration"],
    difficulty: "Beginner",
    tags: ["Data Science", "Feature Engineering", "Data Cleaning"]
  },
  {
    id: "neural-network-architectures",
    title: "Neural Network Architectures",
    description: "Different structures and designs of neural networks for various tasks",
    category: "Deep Learning",
    projects: ["dqn-flappy-bird", "custom-diffusion-model", "custom-gpt-llm", "image-classifier"],
    difficulty: "Intermediate",
    tags: ["Deep Learning", "Architecture Design", "Neural Networks"]
  },
  {
    id: "computer-vision",
    title: "Computer Vision",
    description: "Field of AI that enables computers to interpret and understand visual information",
    category: "AI Applications",
    projects: ["image-classifier", "tamid-image-classifier", "custom-diffusion-model"],
    difficulty: "Intermediate",
    tags: ["Computer Vision", "Image Processing", "Object Detection"]
  },
  // New Topics
  {
    id: "gradient-descent",
    title: "Gradient Descent",
    description: "Fundamental optimization algorithm used to minimize loss functions in machine learning",
    category: "Fundamentals",
    projects: ["image-classifier", "custom-gpt-llm", "dqn-flappy-bird"],
    difficulty: "Beginner",
    tags: ["Optimization", "Mathematics", "Training", "Fundamentals"]
  },
  {
    id: "regression",
    title: "Regression Analysis",
    description: "Statistical method for modeling relationships between variables and making predictions",
    category: "Machine Learning",
    projects: ["real-salary"],
    difficulty: "Beginner",
    tags: ["Supervised Learning", "Prediction", "Statistics", "Linear Models"]
  },
  {
    id: "rnns",
    title: "Recurrent Neural Networks (RNNs)",
    description: "Neural networks designed for sequential data with memory capabilities",
    category: "Deep Learning",
    projects: ["custom-gpt-llm"],
    difficulty: "Intermediate",
    tags: ["Sequential Data", "NLP", "Time Series", "Memory"]
  },
  {
    id: "lstm-models",
    title: "LSTM Networks",
    description: "Long Short-Term Memory networks that solve the vanishing gradient problem in RNNs",
    category: "Deep Learning",
    projects: ["custom-gpt-llm"],
    difficulty: "Intermediate",
    tags: ["Sequential Data", "NLP", "Memory", "Vanishing Gradient"]
  },
  {
    id: "unet",
    title: "U-Net Architecture",
    description: "Convolutional network architecture for biomedical image segmentation with skip connections",
    category: "Deep Learning",
    projects: ["custom-diffusion-model"],
    difficulty: "Advanced",
    tags: ["Image Segmentation", "Medical Imaging", "Skip Connections", "CNN"]
  },
  {
    id: "ddpm",
    title: "Denoising Diffusion Probabilistic Models (DDPM)",
    description: "Probabilistic generative models that learn to reverse a diffusion process",
    category: "Generative AI",
    projects: ["custom-diffusion-model"],
    difficulty: "Advanced",
    tags: ["Generative AI", "Probabilistic Models", "Denoising", "Diffusion"]
  },
  {
    id: "encoders-decoders",
    title: "Encoders and Decoders",
    description: "Neural network components that compress and reconstruct data representations",
    category: "Deep Learning",
    projects: ["custom-diffusion-model", "custom-gpt-llm"],
    difficulty: "Intermediate",
    tags: ["Representation Learning", "Autoencoders", "Sequence-to-Sequence", "Transformers"]
  },
  // Additional Essential Topics
  {
    id: "transformers",
    title: "Transformer Architecture",
    description: "Attention-based neural network architecture that revolutionized NLP and beyond",
    category: "Deep Learning",
    projects: ["custom-gpt-llm"],
    difficulty: "Advanced",
    tags: ["Attention", "NLP", "Self-Attention", "Transformers"]
  },
  {
    id: "attention-mechanisms",
    title: "Attention Mechanisms",
    description: "Neural network components that focus on relevant parts of input data",
    category: "Deep Learning",
    projects: ["custom-gpt-llm"],
    difficulty: "Intermediate",
    tags: ["Attention", "Focus", "Sequence Modeling", "NLP"]
  },
  {
    id: "loss-functions",
    title: "Loss Functions",
    description: "Mathematical functions that measure the difference between predicted and actual values",
    category: "Fundamentals",
    projects: ["image-classifier", "custom-gpt-llm", "dqn-flappy-bird", "real-salary"],
    difficulty: "Beginner",
    tags: ["Training", "Optimization", "Mathematics", "Evaluation"]
  },
  {
    id: "regularization",
    title: "Regularization Techniques",
    description: "Methods to prevent overfitting and improve model generalization",
    category: "Fundamentals",
    projects: ["image-classifier", "custom-gpt-llm", "real-salary"],
    difficulty: "Intermediate",
    tags: ["Overfitting", "Generalization", "Dropout", "Batch Normalization"]
  },
  {
    id: "evaluation-metrics",
    title: "Evaluation Metrics",
    description: "Metrics and techniques for assessing machine learning model performance",
    category: "Fundamentals",
    projects: ["image-classifier", "tamid-image-classifier", "real-salary", "clustering-exploration"],
    difficulty: "Beginner",
    tags: ["Evaluation", "Performance", "Metrics", "Validation"]
  }
];

const projects = [
  { id: "all", name: "All Projects" },
  { id: "dqn-flappy-bird", name: "DQN Flappy Bird" },
  { id: "custom-diffusion-model", name: "Custom Diffusion Model" },
  { id: "custom-gpt-llm", name: "Custom GPT LLM" },
  { id: "tamid-image-classifier", name: "TAMID Image Classifier" },
  { id: "image-classifier", name: "Image Classifier" },
  { id: "clustering-exploration", name: "Clustering Exploration" },
  { id: "real-salary", name: "Real Salary" }
];

const categories = [
  "All Categories",
  "Fundamentals",
  "Machine Learning",
  "Deep Learning", 
  "Generative AI",
  "Natural Language Processing",
  "Data Science",
  "AI Applications"
];

const difficultyColors = {
  "Beginner": "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  "Intermediate": "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
  "Advanced": "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
};

export default function TopicsPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedProject, setSelectedProject] = useState("all");
  const [selectedCategory, setSelectedCategory] = useState("All Categories");

  const filteredTopics = useMemo(() => {
    return topics.filter(topic => {
      const matchesSearch = topic.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           topic.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           topic.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesProject = selectedProject === "all" || topic.projects.includes(selectedProject);
      
      const matchesCategory = selectedCategory === "All Categories" || topic.category === selectedCategory;
      
      return matchesSearch && matchesProject && matchesCategory;
    });
  }, [searchTerm, selectedProject, selectedCategory]);

  const sortedTopics = useMemo(() => {
    return [...filteredTopics].sort((a, b) => a.title.localeCompare(b.title));
  }, [filteredTopics]);

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
              <Link href="/topics" className="text-purple-600 dark:text-purple-400 font-medium">
                Topics
              </Link>
              <Link href="/visualize" className="text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors">
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

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <Book className="w-12 h-12 text-purple-600 dark:text-purple-400" />
            <h1 className="text-4xl font-bold text-slate-900 dark:text-white">
              Machine Learning Topics
            </h1>
          </div>
          <p className="text-xl text-slate-700 dark:text-slate-300 max-w-3xl mx-auto leading-relaxed">
            Explore the fundamental concepts, techniques, and algorithms that power modern machine learning. 
            Each topic includes detailed explanations, code examples, and connections to real projects.
          </p>

          <div className="flex justify-center">
            <p className="text-slate-600 dark:text-slate-400">
                Note that the content of these pages come from my own knowledge and experience, and may not be perfectly accurate or up-to-date, please refer to the references for further reading.
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
                placeholder="Search topics..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-colors"
              />
            </div>

            {/* Project Filter */}
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
              <select
                value={selectedProject}
                onChange={(e) => setSelectedProject(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-colors appearance-none"
              >
                {projects.map(project => (
                  <option key={project.id} value={project.id}>
                    {project.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Category Filter */}
            <div className="relative">
              <Hash className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-50 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-colors appearance-none"
              >
                {categories.map(category => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Results count */}
          <div className="mt-4 text-sm text-slate-600 dark:text-slate-400">
            {sortedTopics.length} topic{sortedTopics.length !== 1 ? 's' : ''} found
          </div>
        </div>

        {/* Topics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {sortedTopics.map((topic) => (
            <Link 
              key={topic.id} 
              href={`/topics/${topic.id}`}
              className="group block"
            >
              <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-6 hover:shadow-lg hover:border-purple-300 dark:hover:border-purple-600 transition-all duration-300 group-hover:-translate-y-1">
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
                      {topic.title}
                    </h3>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                      {topic.category}
                    </p>
                  </div>
                  <span className={`px-2 py-1 text-xs font-medium rounded-lg ${difficultyColors[topic.difficulty]}`}>
                    {topic.difficulty}
                  </span>
                </div>

                {/* Description */}
                <p className="text-slate-700 dark:text-slate-300 text-sm leading-relaxed mb-4">
                  {topic.description}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {topic.tags.map((tag) => (
                    <span 
                      key={tag}
                      className="flex items-center space-x-1 px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300 text-xs font-medium rounded-lg"
                    >
                      <Tag className="w-3 h-3" />
                      <span>{tag}</span>
                    </span>
                  ))}
                </div>

                {/* Related Projects */}
                <div className="border-t border-slate-200 dark:border-slate-700 pt-4">
                  <p className="text-xs text-slate-600 dark:text-slate-400 mb-2">
                    Used in {topic.projects.length} project{topic.projects.length !== 1 ? 's' : ''}
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {topic.projects.slice(0, 3).map((projectId) => {
                      const project = projects.find(p => p.id === projectId);
                      return (
                        <span 
                          key={projectId}
                          className="px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs rounded"
                        >
                          {project?.name}
                        </span>
                      );
                    })}
                    {topic.projects.length > 3 && (
                      <span className="px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs rounded">
                        +{topic.projects.length - 3} more
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>

        {/* No results */}
        {sortedTopics.length === 0 && (
          <div className="text-center py-16">
            <Book className="w-16 h-16 text-slate-400 dark:text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              No topics found
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              Try adjusting your search terms or filters
            </p>
          </div>
        )}

        {/* Topic Progression Diagram */}
        {sortedTopics.length > 0 && (
          <div className="mt-20">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-4">
                Learning Path
              </h2>
              <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
                Follow this progression to build your ML knowledge systematically from fundamentals to advanced topics
              </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-8 shadow-sm">
              {/* Foundation Level */}
              <div className="mb-12">
                <div className="flex items-center mb-6">
                  <div className="w-8 h-8 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center mr-3">
                    <span className="text-green-600 dark:text-green-400 font-bold text-sm">1</span>
                  </div>
                  <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Foundation</h3>
                  <div className="ml-2 px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 text-xs font-medium rounded">
                    Start Here
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 ml-11">
                  <Link href="/topics/data-preprocessing" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Data Preprocessing
                        </h4>
                        <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 px-2 py-1 rounded">
                          Beginner
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Essential data cleaning and preparation
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/regression" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Regression Analysis
                        </h4>
                        <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 px-2 py-1 rounded">
                          Beginner
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Statistical modeling and prediction
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/clustering-algorithms" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Clustering Algorithms
                        </h4>
                        <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 px-2 py-1 rounded">
                          Beginner
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Introduction to unsupervised learning
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/gradient-descent" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Gradient Descent
                        </h4>
                        <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 px-2 py-1 rounded">
                          Beginner
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Fundamental optimization algorithm
                      </p>
                    </div>
                  </Link>
                </div>
                {/* Additional Foundation Topics */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 ml-11 mt-4">
                  <Link href="/topics/loss-functions" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Loss Functions
                        </h4>
                        <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 px-2 py-1 rounded">
                          Beginner
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Measuring model performance
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/evaluation-metrics" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Evaluation Metrics
                        </h4>
                        <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 px-2 py-1 rounded">
                          Beginner
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Assessing model performance
                      </p>
                    </div>
                  </Link>
                </div>
              </div>

              {/* Arrow Down */}
              <div className="flex justify-center mb-12">
                <ArrowDown className="w-8 h-8 text-slate-400" />
              </div>

              {/* Intermediate Level */}
              <div className="mb-12">
                <div className="flex items-center mb-6">
                  <div className="w-8 h-8 bg-yellow-100 dark:bg-yellow-900/30 rounded-full flex items-center justify-center mr-3">
                    <span className="text-yellow-600 dark:text-yellow-400 font-bold text-sm">2</span>
                  </div>
                  <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Core Concepts</h3>
                  <div className="ml-2 px-2 py-1 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400 text-xs font-medium rounded">
                    Intermediate
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4 ml-11">
                  <Link href="/topics/neural-network-architectures" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Neural Networks
                        </h4>
                        <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400 px-2 py-1 rounded">
                          Intermediate
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Foundation of deep learning
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/convolutional-neural-networks" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          CNNs
                        </h4>
                        <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400 px-2 py-1 rounded">
                          Intermediate
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Computer vision fundamentals
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/rnns" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          RNNs
                        </h4>
                        <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400 px-2 py-1 rounded">
                          Intermediate
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Sequential data processing
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/lstm-models" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          LSTM Networks
                        </h4>
                        <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400 px-2 py-1 rounded">
                          Intermediate
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Advanced sequential modeling
                      </p>
                    </div>
                  </Link>
                </div>
                {/* Additional Intermediate Topics */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 ml-11 mt-4">
                  <Link href="/topics/attention-mechanisms" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Attention Mechanisms
                        </h4>
                        <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400 px-2 py-1 rounded">
                          Intermediate
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Focus and relevance in models
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/encoders-decoders" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Encoders & Decoders
                        </h4>
                        <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400 px-2 py-1 rounded">
                          Intermediate
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Data representation learning
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/computer-vision" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Computer Vision
                        </h4>
                        <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400 px-2 py-1 rounded">
                          Intermediate
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Image processing applications
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/regularization" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Regularization
                        </h4>
                        <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400 px-2 py-1 rounded">
                          Intermediate
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Preventing overfitting
                      </p>
                    </div>
                  </Link>
                </div>
              </div>

              {/* Arrow Down */}
              <div className="flex justify-center mb-12">
                <ArrowDown className="w-8 h-8 text-slate-400" />
              </div>

              {/* Advanced Level */}
              <div>
                <div className="flex items-center mb-6">
                  <div className="w-8 h-8 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mr-3">
                    <span className="text-red-600 dark:text-red-400 font-bold text-sm">3</span>
                  </div>
                  <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Advanced Topics</h3>
                  <div className="ml-2 px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 text-xs font-medium rounded">
                    Advanced
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 ml-11">
                  <Link href="/topics/reinforcement-learning" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Reinforcement Learning
                        </h4>
                        <span className="text-xs bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 px-2 py-1 rounded">
                          Advanced
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Learning through interaction
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/large-language-models" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Large Language Models
                        </h4>
                        <span className="text-xs bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 px-2 py-1 rounded">
                          Advanced
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        Natural language processing
                      </p>
                    </div>
                  </Link>
                  <Link href="/topics/diffusion-models" className="group">
                    <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 border border-slate-200 dark:border-slate-600 hover:border-purple-300 dark:hover:border-purple-600 transition-colors">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-400">
                          Diffusion Models
                        </h4>
                        <span className="text-xs bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 px-2 py-1 rounded">
                          Advanced
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                        State-of-the-art generation
                      </p>
                    </div>
                  </Link>
                </div>
              </div>

              {/* Learning Tips */}
              <div className="mt-12 pt-8 border-t border-slate-200 dark:border-slate-600">
                <h4 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">💡 Learning Tips</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-600 dark:text-slate-300">
                  <div className="flex items-start space-x-2">
                    <ChevronRight className="w-4 h-4 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span>Start with <strong>Data Preprocessing</strong> to understand how data flows through ML systems</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <ChevronRight className="w-4 h-4 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span>Master <strong>Neural Networks</strong> before diving into specialized architectures</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <ChevronRight className="w-4 h-4 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span>Practice with <strong>hands-on projects</strong> to reinforce theoretical concepts</span>
                  </div>
                </div>
              </div>

              {/* Topic Connections */}
              <div className="mt-8 pt-8 border-t border-slate-200 dark:border-slate-600">
                <h4 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">🔗 Topic Connections</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                  <div className="space-y-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <span className="text-slate-600 dark:text-slate-300">
                        <strong>CNNs</strong> → <strong>Computer Vision</strong> → <strong>Diffusion Models</strong>
                      </span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      <span className="text-slate-600 dark:text-slate-300">
                        <strong>Neural Networks</strong> → <strong>LLMs</strong> → <strong>Advanced NLP</strong>
                      </span>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                      <span className="text-slate-600 dark:text-slate-300">
                        <strong>Data Preprocessing</strong> → <strong>All ML Topics</strong>
                      </span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                      <span className="text-slate-600 dark:text-slate-300">
                        <strong>Neural Networks</strong> → <strong>Reinforcement Learning</strong>
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
