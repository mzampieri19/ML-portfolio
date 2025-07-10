import React from 'react';
import { Calendar, Clock, Tag, ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import CodeBlock from './CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';


interface TopicBlock {
  type: 'heading' | 'paragraph' | 'math' |'list' | 'image' | 'codeBlock' | 'section' | 'highlight' | 'twoColumn' | 'timeline' | 'features';
  content?: string;
  props?: Record<string, any>;
  children?: TopicBlock[];
  component?: React.ReactNode;
}

interface TopicPageData {
  title: string;
  header: {
    category: string;
    difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
    readTime: string;
    description: string;
    relatedProjects?: string[];
    gradientFrom?: string;
    gradientTo?: string;
    borderColor?: string;
  };
  tags: {
    items: string[];
    colorScheme: 'blue' | 'green' | 'purple' | 'orange' | 'red' | 'yellow';
  };
  blocks: TopicBlock[];
  navigation?: {
    colorScheme: 'blue' | 'green' | 'purple' | 'orange' | 'red' | 'yellow';
  };
}

const difficultyColors = {
  "Beginner": "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  "Intermediate": "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
  "Advanced": "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
};

const colorSchemes = {
  blue: {
    tag: 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300',
    nav: 'hover:text-blue-600 dark:hover:text-blue-400',
    highlight: {
      info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
      success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
      error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
    }
  },
  green: {
    tag: 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300',
    nav: 'hover:text-green-600 dark:hover:text-green-400',
    highlight: {
      info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
      success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
      error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
    }
  },
  purple: {
    tag: 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300',
    nav: 'hover:text-purple-600 dark:hover:text-purple-400',
    highlight: {
      info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
      success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
      error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
    }
  },
  orange: {
    tag: 'bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-300',
    nav: 'hover:text-orange-600 dark:hover:text-orange-400',
    highlight: {
      info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
      success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
      error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
    }
  },
  red: {
    tag: 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300',
    nav: 'hover:text-red-600 dark:hover:text-red-400',
    highlight: {
      info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
      success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
      error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
    }
  },
  yellow: {
    tag: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300',
    nav: 'hover:text-yellow-600 dark:hover:text-yellow-400',
    highlight: {
      info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
      success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
      warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
      error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
    }
  }
};

function BlockRenderer({ block, colorScheme }: { block: TopicBlock; colorScheme: any }) {
  switch (block.type) {
    case 'heading':
      const level = block.props?.level || 2;
      const headingClasses = {
        1: 'text-4xl font-bold text-slate-900 dark:text-white mb-6',
        2: 'text-3xl font-bold text-slate-900 dark:text-white mb-4',
        3: 'text-2xl font-semibold text-slate-900 dark:text-white mb-4',
        4: 'text-xl font-semibold text-slate-900 dark:text-white mb-3',
        5: 'text-lg font-medium text-slate-900 dark:text-white mb-3',
        6: 'text-base font-medium text-slate-900 dark:text-white mb-2'
      };
      
      const className = headingClasses[level as keyof typeof headingClasses];
      
      switch (level) {
        case 1:
          return <h1 className={className}>{block.content}</h1>;
        case 2:
          return <h2 className={className}>{block.content}</h2>;
        case 3:
          return <h3 className={className}>{block.content}</h3>;
        case 4:
          return <h4 className={className}>{block.content}</h4>;
        case 5:
          return <h5 className={className}>{block.content}</h5>;
        case 6:
          return <h6 className={className}>{block.content}</h6>;
        default:
          return <h2 className={className}>{block.content}</h2>;
      }

    case 'paragraph':
      return (
        <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-6">
          {block.content}
        </p>
      );

    case 'list':
      const isOrdered = block.props?.ordered;
      
      return (
        <div className="space-y-3 mb-6">
          {block.props?.items?.map((item: string, index: number) => (
            <div key={index} className="flex items-start space-x-3 p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
              <div className="flex-shrink-0 mt-0.5">
                {isOrdered ? (
                  <span className="flex items-center justify-center w-6 h-6 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm font-medium rounded-full">
                    {index + 1}
                  </span>
                ) : (
                  <div className="w-2 h-2 bg-slate-400 dark:bg-slate-500 rounded-full mt-2"></div>
                )}
              </div>
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed flex-1">
                {item}
              </p>
            </div>
          ))}
        </div>
      );

    case 'math':
      return block.props?.block
        ? <BlockMath math={block.content ?? ''} />
        : <InlineMath math={block.content ?? ''} />;

    case 'image':
      return (
        <div className="mb-8">
          <img 
            src={block.props?.src} 
            alt={block.props?.alt || ''} 
            className="w-full rounded-lg shadow-lg"
          />
          {block.props?.caption && (
            <p className="text-sm text-slate-600 dark:text-slate-400 text-center mt-2">
              {block.props.caption}
            </p>
          )}
        </div>
      );

    case 'codeBlock':
      return (
        <div className="mb-8">
          <CodeBlock 
            language={block.props?.language || 'python'} 
            filename={block.props?.filename}
          >
            {block.props?.code || block.content || ''}
          </CodeBlock>
        </div>
      );

    case 'section':
      return (
        <section className={`mb-12 ${block.props?.background ? 'bg-slate-50 dark:bg-slate-800/50 rounded-2xl p-8' : ''}`}>
          {block.props?.title && (
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-8">
              {block.props.title}
            </h2>
          )}
          {block.children?.map((child, index) => (
            <BlockRenderer key={index} block={child} colorScheme={colorScheme} />
          ))}
        </section>
      );

    case 'highlight':
      const variant = block.props?.variant || 'info';
      return (
        <div className={`p-6 rounded-2xl border mb-6 ${colorScheme.highlight[variant]}`}>
          {block.props?.title && (
            <div className="flex items-center space-x-2 mb-3">
              {block.props?.icon}
              <h4 className="font-semibold text-slate-900 dark:text-white">
                {block.props.title}
              </h4>
            </div>
          )}
          {block.children?.map((child, index) => (
            <BlockRenderer key={index} block={child} colorScheme={colorScheme} />
          ))}
        </div>
      );

    case 'twoColumn':
      const ratio = block.props?.ratio || '1:1';
      const [leftRatio, rightRatio] = ratio.split(':').map(Number);
      const leftClass = leftRatio === 1 && rightRatio === 1 ? 'md:w-1/2' : 
                       leftRatio === 2 && rightRatio === 1 ? 'md:w-2/3' :
                       leftRatio === 1 && rightRatio === 2 ? 'md:w-1/3' : 'md:w-1/2';
      const rightClass = leftRatio === 1 && rightRatio === 1 ? 'md:w-1/2' : 
                        leftRatio === 2 && rightRatio === 1 ? 'md:w-1/3' :
                        leftRatio === 1 && rightRatio === 2 ? 'md:w-2/3' : 'md:w-1/2';

      return (
        <div className="flex flex-col md:flex-row gap-8 mb-8">
          <div className={leftClass}>
            {block.props?.left?.map((child: TopicBlock, index: number) => (
              <BlockRenderer key={index} block={child} colorScheme={colorScheme} />
            ))}
          </div>
          <div className={rightClass}>
            {block.props?.right?.map((child: TopicBlock, index: number) => (
              <BlockRenderer key={index} block={child} colorScheme={colorScheme} />
            ))}
          </div>
        </div>
      );

    case 'timeline':
      return (
        <div className="space-y-6 mb-8">
          {block.props?.items?.map((item: any, index: number) => (
            <div key={index} className="flex space-x-4">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-slate-200 dark:bg-slate-700 rounded-full flex items-center justify-center">
                  <div className="w-3 h-3 bg-slate-600 dark:bg-slate-400 rounded-full"></div>
                </div>
              </div>
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  <h4 className="font-semibold text-slate-900 dark:text-white">{item.title}</h4>
                  <span className="text-sm text-slate-500 dark:text-slate-400">{item.date}</span>
                </div>
                <p className="text-slate-700 dark:text-slate-300">{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      );

    case 'features':
      const columns = block.props?.columns || 2;
      const gridClass = columns === 1 ? 'grid-cols-1' : 
                       columns === 2 ? 'grid-cols-1 md:grid-cols-2' :
                       columns === 3 ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3' :
                       'grid-cols-1 md:grid-cols-2 lg:grid-cols-4';
      
      return (
        <div className={`grid ${gridClass} gap-6 mb-8`}>
          {block.props?.features?.map((feature: any, index: number) => (
            <div key={index} className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700">
              <div className="flex items-center space-x-3 mb-3">
                <div className="text-slate-600 dark:text-slate-400">
                  {feature.icon}
                </div>
                <h4 className="font-semibold text-slate-900 dark:text-white">{feature.title}</h4>
              </div>
              <p className="text-sm text-slate-700 dark:text-slate-300">{feature.description}</p>
            </div>
          ))}
        </div>
      );

    default:
      return null;
  }
}

export default function TopicPageBuilder(data: TopicPageData) {
  const colorScheme = colorSchemes[data.tags.colorScheme];

  return (
    <div>
      {/* Header */}
      <div className={`mb-16 p-8 bg-gradient-to-r ${data.header.gradientFrom || 'from-purple-50 to-blue-50'} ${data.header.gradientTo || 'dark:from-purple-900/20 dark:to-blue-900/20'} rounded-2xl border ${data.header.borderColor || 'border-purple-200 dark:border-purple-800'}`}>
        <div className="flex items-center justify-between mb-6">
          <div>
            <div className="flex items-center space-x-4 text-sm text-slate-600 dark:text-slate-400 mb-2">
              <div className="flex items-center space-x-1">
                <Tag className="w-4 h-4" />
                <span>{data.header.category}</span>
              </div>
              <div className="flex items-center space-x-1">
                <Clock className="w-4 h-4" />
                <span>{data.header.readTime}</span>
              </div>
              <span className={`px-2 py-1 text-xs font-medium rounded-lg ${difficultyColors[data.header.difficulty]}`}>
                {data.header.difficulty}
              </span>
            </div>
            <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">
              {data.title}
            </h1>
            <p className="text-lg text-slate-700 dark:text-slate-300">
              {data.header.description}
            </p>
          </div>
          <Link 
            href="/topics"
            className="flex items-center space-x-2 px-4 py-2 bg-slate-900 dark:bg-white text-white dark:text-slate-900 rounded-lg hover:bg-slate-700 dark:hover:bg-slate-200 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Topics</span>
          </Link>
        </div>

        {/* Related Projects */}
        {data.header.relatedProjects && data.header.relatedProjects.length > 0 && (
          <div className="border-t border-slate-200 dark:border-slate-700 pt-4">
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
              Used in projects:
            </p>
            <div className="flex flex-wrap gap-2">
              {data.header.relatedProjects.map((project) => (
                <span 
                  key={project}
                  className="px-3 py-1 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm rounded-lg border border-slate-200 dark:border-slate-600"
                >
                  {project}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-2 mb-12">
        {data.tags.items.map((tag) => (
          <span 
            key={tag}
            className={`flex items-center space-x-1 px-3 py-1 text-sm font-medium rounded-full ${colorScheme.tag}`}
          >
            <Tag className="w-3 h-3" />
            <span>{tag}</span>
          </span>
        ))}
      </div>

      {/* Content Blocks */}
      <div>
        {data.blocks.map((block, index) => (
          <BlockRenderer key={index} block={block} colorScheme={colorScheme} />
        ))}
      </div>
    </div>
  );
}
