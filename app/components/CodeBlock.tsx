'use client';

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';

interface CodeBlockProps {
  children: string | React.ReactNode;
  language: string;
  filename?: string;
}

export default function CodeBlock({ children, language, filename }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  // Ensure children is always a string
  const codeContent = React.useMemo(() => {
    // Debug: log the type and structure of children
    console.log('CodeBlock children:', typeof children, children);
    
    if (typeof children === 'string') {
      return children;
    }
    
    // Handle arrays (multiple children)
    if (Array.isArray(children)) {
      return children.join('');
    }
    
    // Handle React elements - try to extract text content
    if (React.isValidElement(children)) {
      // For simple text elements, try to get the text
      const props = children.props as { children?: unknown };
      if (props?.children && typeof props.children === 'string') {
        return props.children;
      }
    }
    
    // Handle null, undefined, numbers, etc.
    if (children == null) {
      return '';
    }
    
    // Fallback: convert to string
    return String(children);
  }, [children]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(codeContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group mb-6">
      {filename && (
        <div className="flex items-center justify-between bg-slate-800 text-slate-300 px-4 py-2 text-sm font-mono border-b border-slate-700 rounded-t-lg">
          <span>{filename}</span>
          <button
            onClick={handleCopy}
            className="flex items-center space-x-1 text-slate-400 hover:text-white transition-colors opacity-0 group-hover:opacity-100"
          >
            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
            <span className="text-xs">{copied ? 'Copied!' : 'Copy'}</span>
          </button>
        </div>
      )}
      <div className="relative">
        {!filename && (
          <button
            onClick={handleCopy}
            className="absolute top-3 right-3 z-10 flex items-center space-x-1 bg-slate-700 text-slate-300 px-2 py-1 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity"
          >
            {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            <span>{copied ? 'Copied!' : 'Copy'}</span>
          </button>
        )}
        <SyntaxHighlighter
          language={language}
          style={oneDark}
          customStyle={{
            margin: 0,
            borderRadius: filename ? '0 0 0.5rem 0.5rem' : '0.5rem',
            fontSize: '0.875rem',
            lineHeight: '1.5',
          }}
          showLineNumbers={true}
          wrapLines={true}
        >
          {codeContent}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
