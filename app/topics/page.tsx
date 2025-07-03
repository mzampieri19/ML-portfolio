'use client';

import { useState, useMemo } from 'react';
import Link from "next/link";
import { Brain, Search, Filter, Tag, Book, Code, Image, FileText, List, Hash } from "lucide-react";

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
  "Deep Learning",
  "Machine Learning", 
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
      </div>
    </div>
  );
}
