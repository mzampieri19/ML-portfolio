'use client';

import React, { useState } from 'react';

interface InfoSection {
  id: string;
  title: string;
  content: string | React.ReactNode;
  type: 'explanation' | 'tip' | 'warning' | 'formula' | 'example';
}

interface InfoSidebarProps {
  sections: InfoSection[];
  isOpen?: boolean;
  onToggle?: () => void;
  position?: 'left' | 'right';
  className?: string;
}

interface InfoSectionProps {
  section: InfoSection;
  isExpanded: boolean;
  onToggle: () => void;
}

const InfoSectionComponent: React.FC<InfoSectionProps> = ({
  section,
  isExpanded,
  onToggle
}) => {
  const getIcon = () => {
    switch (section.type) {
      case 'explanation':
        return (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'tip':
        return (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        );
      case 'warning':
        return (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
      case 'formula':
        return (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
        );
      case 'example':
        return (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        );
      default:
        return null;
    }
  };

  const getColorClasses = () => {
    switch (section.type) {
      case 'explanation':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'tip':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'warning':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'formula':
        return 'text-purple-600 bg-purple-50 border-purple-200';
      case 'example':
        return 'text-gray-600 bg-gray-50 border-gray-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const typeLabels = {
    explanation: 'Explanation',
    tip: 'Tip',
    warning: 'Important',
    formula: 'Formula',
    example: 'Example'
  };

  return (
    <div className={`border rounded-lg mb-3 ${getColorClasses()}`}>
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 text-left flex items-center justify-between hover:bg-opacity-80 transition-colors"
      >
        <div className="flex items-center space-x-2">
          {getIcon()}
          <div>
            <div className="text-xs font-medium uppercase tracking-wide opacity-70">
              {typeLabels[section.type]}
            </div>
            <div className="font-medium">{section.title}</div>
          </div>
        </div>
        <svg 
          className={`w-5 h-5 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {isExpanded && (
        <div className="px-4 pb-4">
          <div className="text-sm leading-relaxed">
            {typeof section.content === 'string' ? (
              <div className="whitespace-pre-wrap">{section.content}</div>
            ) : (
              section.content
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export const InfoSidebar: React.FC<InfoSidebarProps> = ({
  sections,
  isOpen = true,
  onToggle,
  position = 'right',
  className = ''
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  const toggleSection = (sectionId: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  };

  const expandAll = () => {
    setExpandedSections(new Set(sections.map(s => s.id)));
  };

  const collapseAll = () => {
    setExpandedSections(new Set());
  };

  if (!isOpen) {
    return (
      <div className={`${position === 'left' ? 'order-first' : 'order-last'} ${className}`}>
        <button
          onClick={onToggle}
          className="fixed top-1/2 right-0 transform -translate-y-1/2 bg-blue-600 text-white p-2 rounded-l-lg shadow-lg hover:bg-blue-700 transition-colors z-10"
          title="Show information panel"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className={`bg-white border-l border-gray-200 flex flex-col ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-800">Information</h2>
          <div className="flex items-center space-x-2">
            <button
              onClick={expandAll}
              className="text-xs text-blue-600 hover:text-blue-700 px-2 py-1 rounded hover:bg-blue-50"
              title="Expand all sections"
            >
              Expand All
            </button>
            <button
              onClick={collapseAll}
              className="text-xs text-gray-600 hover:text-gray-700 px-2 py-1 rounded hover:bg-gray-50"
              title="Collapse all sections"
            >
              Collapse All
            </button>
            {onToggle && (
              <button
                onClick={onToggle}
                className="text-gray-400 hover:text-gray-600 p-1"
                title="Hide information panel"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {sections.length === 0 ? (
          <div className="text-center py-8">
            <div className="text-gray-400 mb-2">
              <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-1">No Information Available</h3>
            <p className="text-gray-500">Information about this model will appear here.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {sections.map(section => (
              <InfoSectionComponent
                key={section.id}
                section={section}
                isExpanded={expandedSections.has(section.id)}
                onToggle={() => toggleSection(section.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <p className="text-xs text-gray-500 text-center">
          Click on any section to expand and learn more about the concepts.
        </p>
      </div>
    </div>
  );
};

// Utility function to create common info sections
export const createInfoSection = (
  id: string,
  title: string,
  content: string | React.ReactNode,
  type: InfoSection['type'] = 'explanation'
): InfoSection => ({
  id,
  title,
  content,
  type
});

// Common info sections for different models
export const commonInfoSections = {
  linearRegression: [
    createInfoSection(
      'what-is-linear-regression',
      'What is Linear Regression?',
      'Linear regression is a fundamental machine learning algorithm that models the relationship between a dependent variable and independent variables by fitting a linear equation to observed data. It assumes that the relationship between variables is linear.',
      'explanation'
    ),
    createInfoSection(
      'linear-equation',
      'The Linear Equation',
      'y = mx + b\n\nWhere:\n• y = predicted value\n• m = slope (coefficient)\n• x = input feature\n• b = y-intercept (bias)',
      'formula'
    ),
    createInfoSection(
      'overfitting-tip',
      'Avoiding Overfitting',
      'Use regularization techniques like Ridge (L2) or Lasso (L1) to prevent overfitting, especially with polynomial features. Start with lower polynomial degrees and increase gradually.',
      'tip'
    )
  ],

  neuralNetwork: [
    createInfoSection(
      'what-is-neural-network',
      'What are Neural Networks?',
      'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that can learn complex patterns in data through training.',
      'explanation'
    ),
    createInfoSection(
      'activation-functions',
      'Activation Functions',
      'Activation functions introduce non-linearity to the network:\n• ReLU: Most common, fast computation\n• Tanh: Output between -1 and 1\n• Sigmoid: Output between 0 and 1\n• Linear: No activation (output layer)',
      'tip'
    ),
    createInfoSection(
      'learning-rate-warning',
      'Learning Rate Selection',
      'Choose learning rate carefully: too high and the model may not converge, too low and training will be very slow. Start with 0.01 and adjust based on performance.',
      'warning'
    )
  ],

  decisionTree: [
    createInfoSection(
      'what-is-decision-tree',
      'What are Decision Trees?',
      'Decision trees are tree-like models that make predictions by learning simple decision rules inferred from data features. Each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf represents a class label.',
      'explanation'
    ),
    createInfoSection(
      'tree-depth-tip',
      'Controlling Tree Depth',
      'Limit tree depth to prevent overfitting. Deeper trees can memorize training data but may not generalize well to new data. Start with depth 3-5 and increase gradually.',
      'tip'
    )
  ],

  clustering: [
    createInfoSection(
      'what-is-clustering',
      'What is Clustering?',
      'Clustering is an unsupervised learning technique that groups similar data points together. It discovers hidden patterns in data without requiring labeled examples.',
      'explanation'
    ),
    createInfoSection(
      'choosing-k',
      'Choosing the Number of Clusters',
      'Use the elbow method: plot the sum of squared distances for different K values and look for the "elbow" point where the rate of decrease sharply changes.',
      'tip'
    )
  ]
};

export default InfoSidebar;
