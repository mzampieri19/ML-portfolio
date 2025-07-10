'use client';

import React from 'react';
import { ModelMetrics } from '@/lib/visualize/types';
import { JSX } from 'react/jsx-runtime';

interface MetricsDashboardProps {
  metrics: ModelMetrics;
  isTraining?: boolean;
  className?: string;
}

interface MetricCardProps {
  title: string;
  value: number | null;
  format: 'percentage' | 'decimal' | 'integer';
  description: string;
  isLoading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({ 
  title, 
  value, 
  format, 
  description, 
  isLoading = false 
}) => {
  const formatValue = (val: number | null): string => {
    if (val === null || isNaN(val)) return '--';
    
    switch (format) {
      case 'percentage':
        return `${(val * 100).toFixed(2)}%`;
      case 'decimal':
        return val.toFixed(4);
      case 'integer':
        return Math.round(val).toString();
      default:
        return val.toString();
    }
  };

  const getValueColor = (): string => {
    if (value === null || isNaN(value)) return 'text-gray-400';
    
    // Color coding based on metric type and value
    if (title.toLowerCase().includes('accuracy') || title.toLowerCase().includes('r²')) {
      return value > 0.8 ? 'text-green-600' : value > 0.6 ? 'text-yellow-600' : 'text-red-600';
    }
    if (title.toLowerCase().includes('loss') || title.toLowerCase().includes('error')) {
      return value < 0.2 ? 'text-green-600' : value < 0.5 ? 'text-yellow-600' : 'text-red-600';
    }
    return 'text-blue-600';
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-700">{title}</h3>
        {isLoading && (
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
        )}
      </div>
      
      <div className={`text-2xl font-bold ${getValueColor()} mb-1`}>
        {isLoading ? '--' : formatValue(value)}
      </div>
      
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  );
};

const TrainingProgress: React.FC<{ 
  currentEpoch: number; 
  totalEpochs: number; 
  progress: number;
}> = ({ currentEpoch, totalEpochs, progress }) => (
  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-sm font-medium text-blue-700">Training Progress</h3>
      <span className="text-sm text-blue-600">
        Epoch {currentEpoch}/{totalEpochs}
      </span>
    </div>
    
    <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
      <div 
        className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
        style={{ width: `${Math.min(progress * 100, 100)}%` }}
      ></div>
    </div>
    
    <p className="text-xs text-blue-600">
      {(progress * 100).toFixed(1)}% complete
    </p>
  </div>
);

const LossChart: React.FC<{ 
  trainingLoss: number[]; 
  validationLoss?: number[];
  maxPoints?: number;
}> = ({ trainingLoss, validationLoss, maxPoints = 50 }) => {
  const displayTrainingLoss = trainingLoss.slice(-maxPoints);
  const displayValidationLoss = validationLoss?.slice(-maxPoints);
  
  const maxLoss = Math.max(
    ...displayTrainingLoss,
    ...(displayValidationLoss || [])
  );
  
  const minLoss = Math.min(
    ...displayTrainingLoss,
    ...(displayValidationLoss || [])
  );
  
  const range = maxLoss - minLoss || 1;
  
  const getY = (value: number): number => {
    return ((maxLoss - value) / range) * 100;
  };
  
  const createPath = (losses: number[]): string => {
    return losses
      .map((loss, i) => {
        const x = (i / (losses.length - 1)) * 100;
        const y = getY(loss);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  };

  if (displayTrainingLoss.length < 2) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 h-40 flex items-center justify-center">
        <p className="text-gray-500 text-sm">Training data will appear here...</p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-700 mb-3">Loss Over Time</h3>
      
      <div className="relative h-32">
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          {/* Grid lines */}
          <defs>
            <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#e5e7eb" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100" height="100" fill="url(#grid)" />
          
          {/* Training loss line */}
          <path
            d={createPath(displayTrainingLoss)}
            fill="none"
            stroke="#dc2626"
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
          
          {/* Validation loss line */}
          {displayValidationLoss && displayValidationLoss.length > 1 && (
            <path
              d={createPath(displayValidationLoss)}
              fill="none"
              stroke="#2563eb"
              strokeWidth="1"
              strokeDasharray="3,3"
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
        
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-gray-500 -ml-8">
          <span>{maxLoss.toFixed(3)}</span>
          <span>{((maxLoss + minLoss) / 2).toFixed(3)}</span>
          <span>{minLoss.toFixed(3)}</span>
        </div>
      </div>
      
      {/* Legend */}
      <div className="flex items-center justify-center space-x-4 mt-3 text-xs">
        <div className="flex items-center">
          <div className="w-3 h-0.5 bg-red-600 mr-1"></div>
          <span className="text-gray-600">Training Loss</span>
        </div>
        {displayValidationLoss && (
          <div className="flex items-center">
            <div className="w-3 h-0.5 bg-blue-600 border-dashed mr-1"></div>
            <span className="text-gray-600">Validation Loss</span>
          </div>
        )}
      </div>
    </div>
  );
};

export const MetricsDashboard: React.FC<MetricsDashboardProps> = ({
  metrics,
  isTraining = false,
  className = ''
}) => {
  const getMetricCards = () => {
    const cards: JSX.Element[] = [];
    
    // Core metrics based on model type
    if (metrics.accuracy !== null) {
      cards.push(
        <MetricCard
          key="accuracy"
          title="Accuracy"
          value={metrics.accuracy}
          format="percentage"
          description="Percentage of correct predictions"
          isLoading={isTraining}
        />
      );
    }
    
    if (metrics.loss !== null) {
      cards.push(
        <MetricCard
          key="loss"
          title="Loss"
          value={metrics.loss}
          format="decimal"
          description="Model's prediction error"
          isLoading={isTraining}
        />
      );
    }
    
    if (metrics.mse !== null) {
      cards.push(
        <MetricCard
          key="mse"
          title="MSE"
          value={metrics.mse}
          format="decimal"
          description="Mean Squared Error"
          isLoading={isTraining}
        />
      );
    }
    
    if (metrics.r2 !== null) {
      cards.push(
        <MetricCard
          key="r2"
          title="R² Score"
          value={metrics.r2}
          format="decimal"
          description="Coefficient of determination"
          isLoading={isTraining}
        />
      );
    }
    
    if (metrics.mae !== null) {
      cards.push(
        <MetricCard
          key="mae"
          title="MAE"
          value={metrics.mae}
          format="decimal"
          description="Mean Absolute Error"
          isLoading={isTraining}
        />
      );
    }
    
    if (metrics.precision !== null) {
      cards.push(
        <MetricCard
          key="precision"
          title="Precision"
          value={metrics.precision}
          format="percentage"
          description="True positives / (True + False positives)"
          isLoading={isTraining}
        />
      );
    }
    
    if (metrics.recall !== null) {
      cards.push(
        <MetricCard
          key="recall"
          title="Recall"
          value={metrics.recall}
          format="percentage"
          description="True positives / (True positives + False negatives)"
          isLoading={isTraining}
        />
      );
    }
    
    if (metrics.f1Score !== null) {
      cards.push(
        <MetricCard
          key="f1"
          title="F1 Score"
          value={metrics.f1Score}
          format="percentage"
          description="Harmonic mean of precision and recall"
          isLoading={isTraining}
        />
      );
    }
    
    return cards;
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Training Progress */}
      {isTraining && metrics.trainingProgress && (
        <TrainingProgress
          currentEpoch={metrics.trainingProgress.currentEpoch}
          totalEpochs={metrics.trainingProgress.totalEpochs}
          progress={metrics.trainingProgress.progress}
        />
      )}
      
      {/* Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {getMetricCards()}
      </div>
      
      {/* Loss Chart */}
      {metrics.lossHistory && metrics.lossHistory.length > 0 && (
        <LossChart
          trainingLoss={metrics.lossHistory}
          validationLoss={metrics.validationLossHistory}
        />
      )}
      
      {/* Additional Metrics */}
      {metrics.additionalMetrics && Object.keys(metrics.additionalMetrics).length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Additional Metrics</h3>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(metrics.additionalMetrics).map(([key, value]) => (
              <MetricCard
                key={key}
                title={key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                value={value}
                format="decimal"
                description={`Custom metric: ${key}`}
                isLoading={isTraining}
              />
            ))}
          </div>
        </div>
      )}
      
      {/* No Metrics State */}
      {getMetricCards().length === 0 && !isTraining && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
          <div className="text-gray-400 mb-2">
            <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-1">No Metrics Available</h3>
          <p className="text-gray-500">Train your model to see performance metrics here.</p>
        </div>
      )}
    </div>
  );
};

export default MetricsDashboard;
