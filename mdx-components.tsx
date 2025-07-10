import type { MDXComponents } from 'mdx/types'
import CodeBlock from './app/components/CodeBlock'
import InteractiveDemo from './app/components/InteractiveDemo'
import Table, { 
  TableHeader, 
  TableBody, 
  TableRow, 
  TableCell, 
  MetricsTable, 
  ClassificationTable, 
  FileTable 
} from './app/components/Table'
import PageBuilder, {
  Heading,
  Paragraph,
  List,
  ListItem,
  HighlightBox,
  Quote,
  Divider,
  Section,
  TwoColumn,
  FeatureList,
  MetricsGrid,
  Timeline
} from './app/components/PageBuilder'
import {
  ProjectHeader,
  ProjectTags,
  ProjectNavigation,
  ProjectLayout,
  ProjectSection,
  ProjectSubsection,
  ResultsSection,
  ArchitectureSection
} from './app/components/ProjectComponents'

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    // Custom components
    CodeBlock,
    InteractiveDemo,
    Table,
    TableHeader,
    TableBody,
    TableRow,
    TableCell,
    MetricsTable,
    ClassificationTable,
    FileTable,
    
    // Page Builder Components
    PageBuilder,
    Heading,
    Paragraph,
    List,
    ListItem,
    HighlightBox,
    Quote,
    Divider,
    Section,
    TwoColumn,
    FeatureList,
    MetricsGrid,
    Timeline,
    ProjectHeader,
    ProjectTags,
    ProjectNavigation,
    ProjectLayout,
    ProjectSection,
    ProjectSubsection,
    ResultsSection,
    ArchitectureSection,
    
    // Enhanced HTML elements
    h1: ({ children, ...props }) => (
      <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-6" {...props}>
        {children}
      </h1>
    ),
    h2: ({ children, ...props }) => (
      <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-6 mt-12" {...props}>
        {children}
      </h2>
    ),
    h3: ({ children, ...props }) => (
      <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4 mt-8" {...props}>
        {children}
      </h3>
    ),
    p: ({ children, ...props }) => (
      <p className="text-slate-700 dark:text-slate-300 mb-4 leading-relaxed" {...props}>
        {children}
      </p>
    ),
    ul: ({ children, ...props }) => (
      <ul className="list-disc list-inside text-slate-700 dark:text-slate-300 mb-4 space-y-2" {...props}>
        {children}
      </ul>
    ),
    ol: ({ children, ...props }) => (
      <ol className="list-decimal list-inside text-slate-700 dark:text-slate-300 mb-4 space-y-2" {...props}>
        {children}
      </ol>
    ),
    li: ({ children, ...props }) => (
      <li className="text-slate-700 dark:text-slate-300" {...props}>
        {children}
      </li>
    ),
    blockquote: ({ children, ...props }) => (
      <blockquote className="border-l-4 border-blue-500 pl-4 italic text-slate-600 dark:text-slate-400 mb-4" {...props}>
        {children}
      </blockquote>
    ),
    table: ({ children, ...props }) => (
      <div className="overflow-x-auto mb-6">
        <table className="min-w-full border border-slate-300 dark:border-slate-600" {...props}>
          {children}
        </table>
      </div>
    ),
    thead: ({ children, ...props }) => (
      <thead className="bg-slate-100 dark:bg-slate-700" {...props}>
        {children}
      </thead>
    ),
    th: ({ children, ...props }) => (
      <th className="px-4 py-2 text-left font-semibold text-slate-900 dark:text-white border-b border-slate-300 dark:border-slate-600" {...props}>
        {children}
      </th>
    ),
    td: ({ children, ...props }) => (
      <td className="px-4 py-2 text-slate-700 dark:text-slate-300 border-b border-slate-200 dark:border-slate-700" {...props}>
        {children}
      </td>
    ),
    strong: ({ children, ...props }) => (
      <strong className="font-semibold text-slate-900 dark:text-white" {...props}>
        {children}
      </strong>
    ),
    
    ...components,
  }
}
