import Link from "next/link";
import { Brain, Calendar, Tag } from "lucide-react";

export default function ProjectsPage() {
  const projects = [
    {
      title: "DQN Flappy Bird",
      description: "Created a DQN to train and play the game Flappy Bird",
      tags: ["Reinforcement Learning", "From Scratch", "Neural Network"],
      slug: "dqn-flappy-bird",
      date: "2025-06",
      status: "Completed"
    },
    {
      title: "Custom Diffusion Model",
      description: "Diffusion model from scratch, includes encoding, decoding, UNET, DDPM, Diffusion",
      tags: ["Diffusion", "From Scratch", "Neural Network", "GenAI"],
      slug: "custom-diffusion-model",
      date: "2025-06",
      status: "Completed"
    },
    {
      title: "Custom GPT LLM",
      description: "GPT LLM from scratch, includes encoding, decoding, data extraction, and training",
      tags: ["LLM", "GPT", "From Scratch", "Neural Network", "GenAI"],
      slug: "custom-gpt-llm",
      date: "2025-06",
      status: "Completed"
    },
    {
      title: "TAMID Image Classifier",
      description: "CNN model created for the TAMID club to classify different types of plastics",
      tags: ["CNN", "Group Work", "Startup", "Data Preparation"],
      slug: "tamid-image-classifier",
      date: "2025-03",
      status: "Completed"
    },
    {
      title: "Image Classifier",
      description: "CNN model to classify different satellite images into different classes",
      tags: ["CNN", "Data Preparation", "Neural Network"],
      slug: "image-classifier",
      date: "2025-02",
      status: "Completed"
    },
    {
      title: "Clustering Exploration",
      description: "Exploration of various clustering techniques on financial data",
      tags: ["Clustering", "Supervised Learning", "TSNE", "KMEANS"],
      slug: "clustering-exploration",
      date: "2025-02",
      status: "Completed"
    },
    {
      title: "Real Salary",
      description: "Group work for the BTTAI industry project with the company Real Salary",
      tags: ["Data Science", "Group Work", "Data Analysis", "LLMs"],
      slug: "real-salary",
      date: "2024-12",
      status: "Completed"
    },
    {
      title: "Break-Through-Tech-AI-2024",
      description: "A collection of various projects, created in the BTTAI program 2024 cohort",
      tags: ["Introduction", "Data Science", "CNN", "Regression", "Random Forest"],
      slug: "break-through-tech-ai",
      date: "2024-08",
      status: "Completed"
    },
    {
      title: "example-project",
      description: "This is an example project to showcase the structure of a project page",
      tags: ["Example", "Template"],
      slug: "example-project",
      date: "2024-01",
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
              <Link href="/topics" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Topics
              </Link>
              <Link href="/visualize" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Visualize
              </Link>
              <Link href="/resources" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Resources
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
