import Image from "next/image";
import Link from "next/link";
import { ArrowRight, Brain, Code, Database, GitBranch } from "lucide-react";

export default function Home() {
  const projects = [
    {
      title: "DQN Flappy Bird",
      description: "Created a DQN to train and play the game Flappy Bird",
      tags: ["Reinforcement Learning", "From Scratch", "Neural Network"],
      slug: "dqn-flappy-bird",
      featured: true
    },
    {
      title: "Custom Diffusion Model",
      description: "Diffusion model from scratch, includes encoding, decoding, UNET, DDPM, Diffusion",
      tags: ["Diffusion", "From Scratch", "Neural Network", "GenAI"],
      slug: "custom-diffusion-model",
      featured: true
    },
    {
      title: "Custom GPT LLM",
      description: "GPT LLM from scratch, includes encoding, decoding, data extraction, and training",
      tags: ["LLM", "GPT", "From Scratch", "Neural Network", "GenAI"],
      slug: "custom-gpt-llm",
      featured: true
    },
    {
      title: "TAMID Image Classifier",
      description: "CNN model created for the TAMID club to classify different types of plastics",
      tags: ["CNN", "Group Work", "Startup", "Data Preparation"],
      slug: "tamid-image-classifier",
      featured: true
    }
  ];

  const skills = [
    { name: "Python", icon: <Code className="w-6 h-6" /> },
    { name: "TensorFlow", icon: <Brain className="w-6 h-6" /> },
    { name: "PyTorch", icon: <Brain className="w-6 h-6" /> },
    { name: "SQL", icon: <Database className="w-6 h-6" /> },
    { name: "Git", icon: <GitBranch className="w-6 h-6" /> },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </div>
            <div className="hidden md:flex space-x-8">
              <Link href="/" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Home
              </Link>
              <Link href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Projects
              </Link>
              <Link href="/about" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                About
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-20 pb-32 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center">
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-slate-900 dark:text-white mb-6">
              Machine Learning
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                Portfolio
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-slate-600 dark:text-slate-300 mb-8 max-w-3xl mx-auto leading-relaxed">
              Exploring the frontiers of artificial intelligence through hands-on projects, 
              research, and innovative solutions to real-world problems.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link 
                href="/projects"
                className="inline-flex items-center px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl transition-all duration-200 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                View Projects
                <ArrowRight className="ml-2 w-5 h-5" />
              </Link>
              <Link 
                href="/about"
                className="inline-flex items-center px-8 py-4 border-2 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 font-semibold rounded-xl hover:bg-slate-50 dark:hover:bg-slate-800 transition-all duration-200"
              >
                About Me
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Projects */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-white dark:bg-slate-800">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white mb-12 text-center">
            Featured Projects
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            {projects.filter(p => p.featured).map((project, index) => (
              <Link 
                key={project.slug}
                href={`/projects/${project.slug}`}
                className="group block"
              >
                <div className="bg-gradient-to-br from-white to-slate-50 dark:from-slate-700 dark:to-slate-800 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 border border-slate-200 dark:border-slate-600 group-hover:border-blue-300 dark:group-hover:border-blue-500">
                  <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                    {project.title}
                  </h3>
                  <p className="text-slate-600 dark:text-slate-300 mb-6 leading-relaxed">
                    {project.description}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {project.tags.map((tag) => (
                      <span 
                        key={tag}
                        className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 text-sm font-medium rounded-full"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white mb-12 text-center">
            Technologies & Tools
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
            {skills.map((skill) => (
              <div 
                key={skill.name}
                className="flex flex-col items-center p-6 bg-white dark:bg-slate-800 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 border border-slate-200 dark:border-slate-700"
              >
                <div className="text-blue-600 dark:text-blue-400 mb-3">
                  {skill.icon}
                </div>
                <span className="text-slate-700 dark:text-slate-300 font-medium">
                  {skill.name}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 dark:bg-slate-950 text-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <Brain className="w-6 h-6 text-blue-400" />
            <span className="text-lg font-semibold">ML Portfolio</span>
          </div>
          <p className="text-slate-400 mb-6">
            Building the future with machine learning, one project at a time.
          </p>
          <div className="flex justify-center space-x-6">
            <Link href="/projects" className="text-slate-400 hover:text-white transition-colors">
              Projects
            </Link>
            <Link href="/about" className="text-slate-400 hover:text-white transition-colors">
              About
            </Link>
            <a href="https://github.com" className="text-slate-400 hover:text-white transition-colors">
              GitHub
            </a>
            <a href="https://linkedin.com" className="text-slate-400 hover:text-white transition-colors">
              LinkedIn
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
