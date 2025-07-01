'use client';

import React from 'react';

interface TableProps {
  children: React.ReactNode;
  className?: string;
  striped?: boolean;
  hoverable?: boolean;
  compact?: boolean;
}

interface TableHeaderProps {
  children: React.ReactNode;
  className?: string;
}

interface TableBodyProps {
  children: React.ReactNode;
  className?: string;
}

interface TableRowProps {
  children: React.ReactNode;
  className?: string;
}

interface TableCellProps {
  children: React.ReactNode;
  className?: string;
  header?: boolean;
  align?: 'left' | 'center' | 'right';
}

// Main Table component
export function Table({ 
  children, 
  className = '', 
  striped = false, 
  hoverable = false,
  compact = false 
}: TableProps) {
  const baseClasses = 'min-w-full border-collapse border border-slate-300 dark:border-slate-600';
  const stripedClasses = striped ? 'table-striped' : '';
  const hoverClasses = hoverable ? 'table-hover' : '';
  const compactClasses = compact ? 'table-compact' : '';
  
  return (
    <div className="overflow-x-auto mb-6">
      <table className={`${baseClasses} ${stripedClasses} ${hoverClasses} ${compactClasses} ${className}`}>
        {children}
      </table>
    </div>
  );
}

// Table Header component
export function TableHeader({ children, className = '' }: TableHeaderProps) {
  return (
    <thead className={`bg-slate-100 dark:bg-slate-700 ${className}`}>
      {children}
    </thead>
  );
}

// Table Body component
export function TableBody({ children, className = '' }: TableBodyProps) {
  return (
    <tbody className={className}>
      {children}
    </tbody>
  );
}

// Table Row component
export function TableRow({ children, className = '' }: TableRowProps) {
  return (
    <tr className={`border-b border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 ${className}`}>
      {children}
    </tr>
  );
}

// Table Cell component
export function TableCell({ 
  children, 
  className = '', 
  header = false, 
  align = 'left' 
}: TableCellProps) {
  const Component = header ? 'th' : 'td';
  const alignClasses = {
    left: 'text-left',
    center: 'text-center',
    right: 'text-right'
  };
  
  const baseClasses = header
    ? 'px-4 py-3 font-semibold text-slate-900 dark:text-white border-b border-slate-300 dark:border-slate-600'
    : 'px-4 py-3 text-slate-700 dark:text-slate-300 border-b border-slate-200 dark:border-slate-700';
    
  return (
    <Component className={`${baseClasses} ${alignClasses[align]} ${className}`}>
      {children}
    </Component>
  );
}

// Pre-built table components for common use cases

// Results/Metrics Table
interface MetricsTableProps {
  data: Array<{ metric: string; value: string | number; }>;
  title?: string;
  className?: string;
}

export function MetricsTable({ data, title, className = '' }: MetricsTableProps) {
  return (
    <div className={`mb-6 ${className}`}>
      {title && (
        <h4 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">
          {title}
        </h4>
      )}
      <Table striped hoverable>
        <TableHeader>
          <TableRow>
            <TableCell header>Metric</TableCell>
            <TableCell header>Value</TableCell>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row, index) => (
            <TableRow key={index}>
              <TableCell>
                <strong>{row.metric}</strong>
              </TableCell>
              <TableCell>{row.value}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

// Classification Results Table
interface ClassificationTableProps {
  data: Array<{ 
    class: string; 
    precision: number; 
    recall: number; 
    f1Score: number; 
    samples: number; 
  }>;
  title?: string;
  className?: string;
}

export function ClassificationTable({ data, title, className = '' }: ClassificationTableProps) {
  return (
    <div className={`mb-6 ${className}`}>
      {title && (
        <h4 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">
          {title}
        </h4>
      )}
      <Table striped hoverable>
        <TableHeader>
          <TableRow>
            <TableCell header>Class</TableCell>
            <TableCell header align="center">Precision</TableCell>
            <TableCell header align="center">Recall</TableCell>
            <TableCell header align="center">F1-Score</TableCell>
            <TableCell header align="center">Samples</TableCell>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row, index) => (
            <TableRow key={index}>
              <TableCell>
                <strong>{row.class}</strong>
              </TableCell>
              <TableCell align="center">{row.precision.toFixed(2)}</TableCell>
              <TableCell align="center">{row.recall.toFixed(2)}</TableCell>
              <TableCell align="center">{row.f1Score.toFixed(2)}</TableCell>
              <TableCell align="center">{row.samples}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

// File Structure Table
interface FileTableProps {
  files: Array<{ file: string; description: string; }>;
  title?: string;
  className?: string;
}

export function FileTable({ files, title, className = '' }: FileTableProps) {
  return (
    <div className={`mb-6 ${className}`}>
      {title && (
        <h4 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">
          {title}
        </h4>
      )}
      <Table striped hoverable>
        <TableHeader>
          <TableRow>
            <TableCell header>File</TableCell>
            <TableCell header>Description</TableCell>
          </TableRow>
        </TableHeader>
        <TableBody>
          {files.map((row, index) => (
            <TableRow key={index}>
              <TableCell>
                <code className="bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded text-sm font-mono">
                  {row.file}
                </code>
              </TableCell>
              <TableCell>{row.description}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

// Default export (main Table component)
export default Table;
