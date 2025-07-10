'use client';

import React from 'react';
import Link from 'next/link';
import { ArrowLeft, ExternalLink, Github, Calendar, Clock, Tag, Users} from 'lucide-react';

// ============================================================================
// CONTENT COMPONENTS
// ============================================================================

// Heading Components
interface HeadingProps {
  children: React.ReactNode;
  level?: 1 | 2 | 3 | 4 | 5 | 6;
  className?: string;
  id?: string;
}

export function Heading({ children, level = 2, className = '', id }: HeadingProps) {
  const baseClasses = 'font-bold text-slate-900 dark:text-white';
  const levelClasses = {
    1: 'text-4xl mb-6',
    2: 'text-3xl mb-6 mt-12',
    3: 'text-2xl mb-4 mt-8',
    4: 'text-xl mb-3 mt-6',
    5: 'text-lg mb-2 mt-4',
    6: 'text-base mb-2 mt-4'
  };

  const combinedClasses = `${baseClasses} ${levelClasses[level]} ${className}`;
  
  switch (level) {
    case 1:
      return <h1 className={combinedClasses} id={id}>{children}</h1>;
    case 2:
      return <h2 className={combinedClasses} id={id}>{children}</h2>;
    case 3:
      return <h3 className={combinedClasses} id={id}>{children}</h3>;
    case 4:
      return <h4 className={combinedClasses} id={id}>{children}</h4>;
    case 5:
      return <h5 className={combinedClasses} id={id}>{children}</h5>;
    case 6:
      return <h6 className={combinedClasses} id={id}>{children}</h6>;
    default:
      return <h2 className={combinedClasses} id={id}>{children}</h2>;
  }
}

// Paragraph Component
interface ParagraphProps {
  children: React.ReactNode;
  className?: string;
  size?: 'sm' | 'base' | 'lg';
  emphasis?: boolean;
}

export function Paragraph({ children, className = '', size = 'base', emphasis = false }: ParagraphProps) {
  const sizeClasses = {
    sm: 'text-sm',
    base: 'text-base',
    lg: 'text-lg'
  };
  
  const baseClasses = `text-slate-700 dark:text-slate-300 mb-4 leading-relaxed ${sizeClasses[size]}`;
  const emphasisClasses = emphasis ? 'font-medium text-slate-800 dark:text-slate-200' : '';
  
  return (
    <p className={`${baseClasses} ${emphasisClasses} ${className}`}>
      {children}
    </p>
  );
}

// List Components
interface ListProps {
  children: React.ReactNode;
  ordered?: boolean;
  className?: string;
  variant?: 'default' | 'bullet' | 'check' | 'arrow';
}

export function List({ children, ordered = false, className = '', variant = 'default' }: ListProps) {
  const Component = ordered ? 'ol' : 'ul';
  const baseClasses = 'text-slate-700 dark:text-slate-300 mb-4 space-y-2';
  
  const variantClasses = {
    default: ordered ? 'list-decimal list-inside' : 'list-disc list-inside',
    bullet: 'list-disc list-inside',
    check: 'list-none',
    arrow: 'list-none'
  };

  return (
    <Component className={`${baseClasses} ${variantClasses[variant]} ${className}`}>
      {children}
    </Component>
  );
}

interface ListItemProps {
  children: React.ReactNode;
  className?: string;
  icon?: React.ReactNode;
}

export function ListItem({ children, className = '', icon }: ListItemProps) {
  return (
    <li className={`text-slate-700 dark:text-slate-300 ${className}`}>
      {icon && <span className="inline-flex items-center mr-2">{icon}</span>}
      {children}
    </li>
  );
}

// Highlight Box Component
interface HighlightBoxProps {
  children: React.ReactNode;
  variant?: 'info' | 'success' | 'warning' | 'error' | 'neutral';
  title?: string;
  icon?: React.ReactNode;
  className?: string;
}

export function HighlightBox({ 
  children, 
  variant = 'neutral', 
  title, 
  icon, 
  className = '' 
}: HighlightBoxProps) {
  const variantClasses = {
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
    warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
    error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
    neutral: 'bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700'
  };

  return (
    <div className={`p-6 rounded-xl border ${variantClasses[variant]} mb-6 ${className}`}>
      {(title || icon) && (
        <div className="flex items-center mb-3">
          {icon && <span className="mr-2">{icon}</span>}
          {title && (
            <h4 className="text-lg font-semibold text-slate-900 dark:text-white">
              {title}
            </h4>
          )}
        </div>
      )}
      <div className="text-slate-700 dark:text-slate-300">
        {children}
      </div>
    </div>
  );
}

// Quote Component
interface QuoteProps {
  children: React.ReactNode;
  author?: string;
  source?: string;
  className?: string;
}

export function Quote({ children, author, source, className = '' }: QuoteProps) {
  return (
    <blockquote className={`border-l-4 border-blue-500 pl-6 py-4 italic text-slate-600 dark:text-slate-400 mb-6 ${className}`}>
      <div className="text-lg mb-2">&ldquo;{children}&rdquo;</div>
      {(author || source) && (
        <footer className="text-sm font-medium text-slate-500 dark:text-slate-500">
          {author && <span>— {author}</span>}
          {source && <span className="ml-2">{source}</span>}
        </footer>
      )}
    </blockquote>
  );
}

// Divider Component
interface DividerProps {
  className?: string;
  variant?: 'solid' | 'dashed' | 'gradient';
}

export function Divider({ className = '', variant = 'solid' }: DividerProps) {
  const variantClasses = {
    solid: 'border-slate-200 dark:border-slate-700',
    dashed: 'border-slate-200 dark:border-slate-700 border-dashed',
    gradient: 'border-0 h-px bg-gradient-to-r from-transparent via-slate-300 dark:via-slate-600 to-transparent'
  };

  if (variant === 'gradient') {
    return <div className={`my-8 ${variantClasses[variant]} ${className}`} />;
  }

  return <hr className={`my-8 border-t ${variantClasses[variant]} ${className}`} />;
}

// ============================================================================
// LAYOUT COMPONENTS
// ============================================================================

// Section Component
interface SectionProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  className?: string;
  background?: boolean;
  id?: string;
}

export function Section({ 
  children, 
  title, 
  subtitle, 
  className = '', 
  background = false,
  id 
}: SectionProps) {
  const bgClasses = background 
    ? 'p-8 bg-slate-50 dark:bg-slate-800/50 rounded-2xl border border-slate-200 dark:border-slate-700' 
    : '';

  return (
    <section className={`mb-12 ${bgClasses} ${className}`} id={id}>
      {title && <Heading level={2}>{title}</Heading>}
      {subtitle && <Paragraph size="lg" emphasis>{subtitle}</Paragraph>}
      {children}
    </section>
  );
}

// Two Column Layout
interface TwoColumnProps {
  left: React.ReactNode;
  right: React.ReactNode;
  className?: string;
  ratio?: '1:1' | '1:2' | '2:1';
}

export function TwoColumn({ left, right, className = '', ratio = '1:1' }: TwoColumnProps) {
  const ratioClasses = {
    '1:1': 'grid-cols-1 md:grid-cols-2',
    '1:2': 'grid-cols-1 md:grid-cols-3',
    '2:1': 'grid-cols-1 md:grid-cols-3'
  };

  const leftClasses = {
    '1:1': '',
    '1:2': 'md:col-span-1',
    '2:1': 'md:col-span-2'
  };

  const rightClasses = {
    '1:1': '',
    '1:2': 'md:col-span-2',
    '2:1': 'md:col-span-1'
  };

  return (
    <div className={`grid ${ratioClasses[ratio]} gap-8 mb-8 ${className}`}>
      <div className={leftClasses[ratio]}>{left}</div>
      <div className={rightClasses[ratio]}>{right}</div>
    </div>
  );
}

// ============================================================================
// SPECIALIZED CONTENT COMPONENTS
// ============================================================================

// Feature List Component
interface FeatureProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

interface FeatureListProps {
  features: FeatureProps[];
  columns?: 1 | 2 | 3;
  className?: string;
}

export function FeatureList({ features, columns = 2, className = '' }: FeatureListProps) {
  const gridClasses = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
  };

  return (
    <div className={`grid ${gridClasses[columns]} gap-6 mb-8 ${className}`}>
      {features.map((feature, index) => (
        <div key={index} className="flex items-start space-x-4 p-4 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex-shrink-0 w-8 h-8 flex items-center justify-center text-blue-600 dark:text-blue-400">
            {feature.icon}
          </div>
          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-2">
              {feature.title}
            </h4>
            <p className="text-slate-600 dark:text-slate-400 text-sm">
              {feature.description}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}

// Metrics Component
interface MetricProps {
  label: string;
  value: string | number;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
}

interface MetricsGridProps {
  metrics: MetricProps[];
  columns?: 2 | 3 | 4;
  className?: string;
}

export function MetricsGrid({ metrics, columns = 3, className = '' }: MetricsGridProps) {
  const gridClasses = {
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-2 lg:grid-cols-4'
  };

  const trendColors = {
    up: 'text-green-600 dark:text-green-400',
    down: 'text-red-600 dark:text-red-400',
    neutral: 'text-slate-600 dark:text-slate-400'
  };

  return (
    <div className={`grid ${gridClasses[columns]} gap-4 mb-8 ${className}`}>
      {metrics.map((metric, index) => (
        <div key={index} className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="text-2xl font-bold text-slate-900 dark:text-white mb-1">
            {metric.value}
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">
            {metric.label}
          </div>
          {metric.change && (
            <div className={`text-xs ${trendColors[metric.trend || 'neutral']}`}>
              {metric.change}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// Timeline Component
interface TimelineItemProps {
  title: string;
  date: string;
  description: string;
  icon?: React.ReactNode;
}

interface TimelineProps {
  items: TimelineItemProps[];
  className?: string;
}

export function Timeline({ items, className = '' }: TimelineProps) {
  return (
    <div className={`relative ${className}`}>
      <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-slate-200 dark:bg-slate-700"></div>
      {items.map((item, index) => (
        <div key={index} className="relative flex items-start mb-8">
          <div className="flex-shrink-0 w-8 h-8 bg-blue-600 dark:bg-blue-500 rounded-full flex items-center justify-center text-white">
            {item.icon || <span className="text-xs font-bold">{index + 1}</span>}
          </div>
          <div className="ml-6">
            <div className="flex items-center space-x-2 mb-1">
              <h4 className="font-semibold text-slate-900 dark:text-white">
                {item.title}
              </h4>
              <span className="text-sm text-slate-500 dark:text-slate-400">
                {item.date}
              </span>
            </div>
            <p className="text-slate-600 dark:text-slate-400">
              {item.description}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// PAGE BUILDER COMPONENT
// ============================================================================

interface ContentBlock {
  type: 'heading' | 'paragraph' | 'list' | 'highlight' | 'quote' | 'divider' | 'section' | 'twoColumn' | 'features' | 'metrics' | 'timeline' | 'custom';
  props?: any;
  content?: React.ReactNode;
  children?: ContentBlock[];
  component?: React.ReactNode; // Add component property for custom blocks
}

interface PageBuilderProps {
  title: string;
  header?: {
    date: string;
    readTime: string;
    description: string;
    githubUrl?: string;
    demoUrl?: string;
    gradientFrom: string;
    gradientTo: string;
    borderColor: string;
    badges?: string[];
    collaborators?: string;
  };
  tags?: {
    items: string[];
    colorScheme?: 'blue' | 'green' | 'purple' | 'orange' | 'red' | 'indigo' | 'emerald' | 'yellow';
  };
  blocks: ContentBlock[];
  navigation?: {
    backUrl?: string;
    backLabel?: string;
    colorScheme?: 'blue' | 'green' | 'purple' | 'orange' | 'red' | 'indigo' | 'emerald' | 'yellow';
  };
  layout?: {
    gradientFrom?: string;
    gradientTo?: string;
    gradientVia?: string;
  };
}

export function PageBuilder({ 
  title, 
  header, 
  tags, 
  blocks, 
  navigation,
}: PageBuilderProps) {
  const renderBlock = (block: ContentBlock, index: number): React.ReactNode => {
    switch (block.type) {
      case 'heading':
        return <Heading key={index} {...block.props}>{block.content}</Heading>;
      
      case 'paragraph':
        return <Paragraph key={index} {...block.props}>{block.content}</Paragraph>;
      
      case 'list':
        return (
          <List key={index} {...block.props}>
            {block.children?.map((child, i) => (
              <ListItem key={i} {...child.props}>{child.content}</ListItem>
            ))}
          </List>
        );
      
      case 'highlight':
        return (
          <HighlightBox key={index} {...block.props}>
            {block.children?.map((child, i) => renderBlock(child, i))}
          </HighlightBox>
        );
      
      case 'quote':
        return <Quote key={index} {...block.props}>{block.content}</Quote>;
      
      case 'divider':
        return <Divider key={index} {...block.props} />;
      
      case 'section':
        return (
          <Section key={index} {...block.props}>
            {block.children?.map((child, i) => renderBlock(child, i))}
          </Section>
        );
      
      case 'twoColumn':
        return (
          <TwoColumn 
            key={index} 
            {...block.props}
            left={block.props?.left?.map((child: ContentBlock, i: number) => renderBlock(child, i))}
            right={block.props?.right?.map((child: ContentBlock, i: number) => renderBlock(child, i))}
          />
        );
      
      case 'features':
        return <FeatureList key={index} {...block.props} />;
      
      case 'metrics':
        return <MetricsGrid key={index} {...block.props} />;
      
      case 'timeline':
        return <Timeline key={index} {...block.props} />;
      
      case 'custom':
        return <div key={index}>{block.component || block.content}</div>;
      
      default:
        return null;
    }
  };

  return (
    <>
      {/* Page Title */}
      <Heading level={1}>{title}</Heading>
      
      {/* Header Section */}
      {header && (
        <div className={`flex items-center justify-between mb-8 p-6 bg-gradient-to-r ${header.gradientFrom} ${header.gradientTo} rounded-2xl border ${header.borderColor}`}>
          <div>
            <div className="flex items-center space-x-4 text-sm text-slate-600 dark:text-slate-400 mb-2">
              <div className="flex items-center space-x-1">
                <Calendar className="w-4 h-4" />
                <span>{header.date}</span>
              </div>
              <div className="flex items-center space-x-1">
                <Clock className="w-4 h-4" />
                <span>{header.readTime}</span>
              </div>
              {header.collaborators && (
                <div className="flex items-center space-x-1">
                  <Users className="w-4 h-4" />
                  <span>{header.collaborators}</span>
                </div>
              )}
            </div>
            <p className="text-lg text-slate-700 dark:text-slate-300">
              {header.description}
            </p>
          </div>
          <div className="flex space-x-3">
            {header.githubUrl && (
              <a 
                href={header.githubUrl}
                className="flex items-center space-x-2 px-4 py-2 bg-slate-900 dark:bg-white text-white dark:text-slate-900 rounded-lg hover:bg-slate-700 dark:hover:bg-slate-200 transition-colors"
              >
                <Github className="w-4 h-4" />
                <span>Code</span>
              </a>
            )}
            {header.demoUrl && (
              <a 
                href={header.demoUrl}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                <span>Demo</span>
              </a>
            )}
          </div>
        </div>
      )}

      {/* Tags Section */}
      {tags && (
        <div className="flex flex-wrap gap-2 mb-8">
          {tags.items.map((tag) => (
            <span 
              key={tag}
              className={`flex items-center space-x-1 px-3 py-1 bg-${tags.colorScheme || 'blue'}-100 dark:bg-${tags.colorScheme || 'blue'}-900/30 text-${tags.colorScheme || 'blue'}-800 dark:text-${tags.colorScheme || 'blue'}-300 text-sm font-medium rounded-full`}
            >
              <Tag className="w-3 h-3" />
              <span>{tag}</span>
            </span>
          ))}
        </div>
      )}

      {/* Content Blocks */}
      {blocks.map((block, index) => renderBlock(block, index))}

      {/* Navigation Footer */}
      {navigation && (
        <div className="mt-12 pt-8 border-t border-slate-200 dark:border-slate-700">
          <Link 
            href={navigation.backUrl || "/projects"}
            className={`inline-flex items-center text-${navigation.colorScheme || 'blue'}-600 dark:text-${navigation.colorScheme || 'blue'}-400 hover:text-${navigation.colorScheme || 'blue'}-700 dark:hover:text-${navigation.colorScheme || 'blue'}-300 transition-colors`}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            {navigation.backLabel || "Back to Projects"}
          </Link>
        </div>
      )}
    </>
  );
}

export default PageBuilder;
