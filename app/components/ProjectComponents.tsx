'use client';

import React from 'react';
import Link from 'next/link';
import { ArrowLeft, ExternalLink, Github, Calendar, Clock, Tag } from 'lucide-react';

// Project Header Component
interface ProjectHeaderProps {
  title: string;
  date: string;
  readTime: string;
  description: string;
  githubUrl?: string;
  demoUrl?: string;
  gradientFrom: string;
  gradientTo: string;
  borderColor: string;
  children?: React.ReactNode; // For any custom content in the header
}

export function ProjectHeader({
  date,
  readTime,
  description,
  githubUrl,
  demoUrl,
  gradientFrom,
  gradientTo,
  borderColor,
  children
}: ProjectHeaderProps) {
  return (
    <div className={`flex items-center justify-between mb-8 p-6 bg-gradient-to-r ${gradientFrom} ${gradientTo} rounded-2xl border ${borderColor}`}>
      <div>
        <div className="flex items-center space-x-4 text-sm text-slate-600 dark:text-slate-400 mb-2">
          <div className="flex items-center space-x-1">
            <Calendar className="w-4 h-4" />
            <span>{date}</span>
          </div>
          <div className="flex items-center space-x-1">
            <Clock className="w-4 h-4" />
            <span>{readTime}</span>
          </div>
        </div>
        <p className="text-lg text-slate-700 dark:text-slate-300">
          {description}
        </p>
        {children}
      </div>
      <div className="flex space-x-3">
        {githubUrl && (
          <a 
            href={githubUrl}
            className="flex items-center space-x-2 px-4 py-2 bg-slate-900 dark:bg-white text-white dark:text-slate-900 rounded-lg hover:bg-slate-700 dark:hover:bg-slate-200 transition-colors"
          >
            <Github className="w-4 h-4" />
            <span>Code</span>
          </a>
        )}
        {demoUrl && (
          <a 
            href={demoUrl}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            <span>Demo</span>
          </a>
        )}
      </div>
    </div>
  );
}

// Project Tags Component
interface ProjectTagsProps {
  tags: string[];
  colorScheme?: 'blue' | 'green' | 'purple' | 'orange' | 'red' | 'indigo' | 'emerald' | 'yellow';
}

export function ProjectTags({ tags, colorScheme = 'blue' }: ProjectTagsProps) {
  const colorMap = {
    blue: 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300',
    green: 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300',
    purple: 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300',
    orange: 'bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-300',
    red: 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300',
    indigo: 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-800 dark:text-indigo-300',
    emerald: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-800 dark:text-emerald-300',
    yellow: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300',
  };

  return (
    <div className="flex flex-wrap gap-2 mb-8">
      {tags.map((tag) => (
        <span 
          key={tag}
          className={`flex items-center space-x-1 px-3 py-1 ${colorMap[colorScheme]} text-sm font-medium rounded-full`}
        >
          <Tag className="w-3 h-3" />
          <span>{tag}</span>
        </span>
      ))}
    </div>
  );
}

// Project Navigation Component
interface ProjectNavigationProps {
  backUrl?: string;
  backLabel?: string;
  colorScheme?: 'blue' | 'green' | 'purple' | 'orange' | 'red' | 'indigo' | 'emerald' | 'yellow';
}

export function ProjectNavigation({ 
  backUrl = "/projects", 
  backLabel = "Back to Projects",
  colorScheme = 'blue' 
}: ProjectNavigationProps) {
  const colorMap = {
    blue: 'text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300',
    green: 'text-green-600 dark:text-green-400 hover:text-green-700 dark:hover:text-green-300',
    purple: 'text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300',
    orange: 'text-orange-600 dark:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300',
    red: 'text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300',
    indigo: 'text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-300',
    emerald: 'text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300',
    yellow: 'text-yellow-600 dark:text-yellow-400 hover:text-yellow-700 dark:hover:text-yellow-300',
  };

  return (
    <div className="mt-12 pt-8 border-t border-slate-200 dark:border-slate-700">
      <Link 
        href={backUrl}
        className={`inline-flex items-center ${colorMap[colorScheme]} transition-colors`}
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        {backLabel}
      </Link>
    </div>
  );
}

// Project Layout Component
interface ProjectLayoutProps {
  children: React.ReactNode;
  gradientFrom?: string;
  gradientTo?: string;
  gradientVia?: string;
}

export function ProjectLayout({ 
  children, 
  gradientFrom = "from-slate-50",
  gradientTo = "to-blue-50",
  gradientVia = "via-white"
}: ProjectLayoutProps) {
  return (
    <div className={`min-h-screen bg-gradient-to-br ${gradientFrom} ${gradientVia} ${gradientTo} dark:from-slate-900 dark:via-slate-800 dark:to-slate-900`}>
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </Link>
            <div className="hidden md:flex space-x-8">
              <Link href="/" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Home
              </Link>
              <Link href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Projects
              </Link>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        {children}
      </article>
    </div>
  );
}

// Section Components for common content structures
interface SectionProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function ProjectSection({ title, children, className = '' }: SectionProps) {
  return (
    <section className={`mb-8 ${className}`}>
      <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-6 mt-12">
        {title}
      </h2>
      {children}
    </section>
  );
}

export function ProjectSubsection({ title, children, className = '' }: SectionProps) {
  return (
    <div className={`mb-6 ${className}`}>
      <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4 mt-8">
        {title}
      </h3>
      {children}
    </div>
  );
}

// Results Section Component
interface ResultsSectionProps {
  title?: string;
  children: React.ReactNode;
  highlight?: boolean;
}

export function ResultsSection({ title = "Results", children, highlight = false }: ResultsSectionProps) {
  return (
    <section className={`mb-8 ${highlight ? 'p-6 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-200 dark:border-slate-700' : ''}`}>
      <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-6 mt-12">
        {title}
      </h2>
      {children}
    </section>
  );
}

// Architecture Section Component
interface ArchitectureSectionProps {
  title?: string;
  children: React.ReactNode;
  diagram?: React.ReactNode;
}

export function ArchitectureSection({ title = "Model Architecture", children, diagram }: ArchitectureSectionProps) {
  return (
    <section className="mb-8">
      <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-6 mt-12">
        {title}
      </h2>
      {children}
      {diagram && (
        <div className="mt-6 p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-200 dark:border-slate-700">
          {diagram}
        </div>
      )}
    </section>
  );
}
