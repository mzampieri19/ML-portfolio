'use client';

import React, { useRef, useEffect, useState } from 'react';
import { DataPoint, VisualizationConfig, VisualizationType } from '@/lib/visualize/types';

interface VisualizationAreaProps {
  data: DataPoint[];
  predictions?: DataPoint[];
  config: VisualizationConfig;
  title?: string;
  isLoading?: boolean;
  className?: string;
  onPointHover?: (point: DataPoint | null) => void;
  onPointClick?: (point: DataPoint) => void;
}

interface ScatterPlotProps {
  data: DataPoint[];
  predictions?: DataPoint[];
  width: number;
  height: number;
  config: VisualizationConfig;
  onPointHover?: (point: DataPoint | null) => void;
  onPointClick?: (point: DataPoint) => void;
}

interface LinePlotProps {
  data: DataPoint[];
  predictions?: DataPoint[];
  width: number;
  height: number;
  config: VisualizationConfig;
}

interface HeatmapProps {
  data: DataPoint[];
  width: number;
  height: number;
  config: VisualizationConfig;
}

const ScatterPlot: React.FC<ScatterPlotProps> = ({
  data,
  predictions,
  width,
  height,
  config,
  onPointHover,
  onPointClick
}) => {
  const padding = 40;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  // Calculate data bounds
  const xMin = Math.min(...data.map(d => d.x), ...(predictions?.map(d => d.x) || []));
  const xMax = Math.max(...data.map(d => d.x), ...(predictions?.map(d => d.x) || []));
  const yMin = Math.min(...data.map(d => d.y), ...(predictions?.map(d => d.y) || []));
  const yMax = Math.max(...data.map(d => d.y), ...(predictions?.map(d => d.y) || []));

  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const scaleX = (x: number) => ((x - xMin) / xRange) * plotWidth + padding;
  const scaleY = (y: number) => height - (((y - yMin) / yRange) * plotHeight + padding);

  return (
    <svg width={width} height={height} className="border border-gray-200 bg-white">
      {/* Grid lines */}
      <defs>
        <pattern id="scatter-grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f3f4f6" strokeWidth="1"/>
        </pattern>
      </defs>
      <rect width={plotWidth} height={plotHeight} x={padding} y={padding} fill="url(#scatter-grid)" />

      {/* Axes */}
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} 
            stroke="#374151" strokeWidth="2" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} 
            stroke="#374151" strokeWidth="2" />

      {/* Axis labels */}
      <text x={width / 2} y={height - 10} textAnchor="middle" className="text-sm text-gray-600">
        {config.xAxisLabel || 'X'}
      </text>
      <text x={15} y={height / 2} textAnchor="middle" transform={`rotate(-90 15 ${height / 2})`} 
            className="text-sm text-gray-600">
        {config.yAxisLabel || 'Y'}
      </text>

      {/* Tick marks and labels */}
      {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
        const x = padding + ratio * plotWidth;
        const y = height - padding + ratio * plotHeight;
        const xValue = xMin + ratio * xRange;
        const yValue = yMin + (1 - ratio) * yRange;
        
        return (
          <g key={ratio}>
            {/* X-axis ticks */}
            <line x1={x} y1={height - padding} x2={x} y2={height - padding + 5} 
                  stroke="#374151" strokeWidth="1" />
            <text x={x} y={height - padding + 20} textAnchor="middle" className="text-xs text-gray-500">
              {xValue.toFixed(1)}
            </text>
            
            {/* Y-axis ticks */}
            <line x1={padding} y1={y} x2={padding - 5} y2={y} 
                  stroke="#374151" strokeWidth="1" />
            <text x={padding - 10} y={y + 3} textAnchor="end" className="text-xs text-gray-500">
              {yValue.toFixed(1)}
            </text>
          </g>
        );
      })}

      {/* Prediction line/surface (for regression) */}
      {predictions && config.showPredictionLine && (
        <path
          d={predictions
            .sort((a, b) => a.x - b.x)
            .map((point, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(point.x)} ${scaleY(point.y)}`)
            .join(' ')}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
          opacity="0.7"
        />
      )}

      {/* Data points */}
      {data.map((point, i) => (
        <circle
          key={i}
          cx={scaleX(point.x)}
          cy={scaleY(point.y)}
          r="4"
          fill={point.color || (point.label !== undefined ? 
            `hsl(${(point.label * 60) % 360}, 70%, 50%)` : '#3b82f6')}
          stroke="white"
          strokeWidth="1"
          className="cursor-pointer hover:r-6 transition-all"
          onMouseEnter={() => onPointHover?.(point)}
          onMouseLeave={() => onPointHover?.(null)}
          onClick={() => onPointClick?.(point)}
        />
      ))}

      {/* Prediction points */}
      {predictions && !config.showPredictionLine && predictions.map((point, i) => (
        <circle
          key={`pred-${i}`}
          cx={scaleX(point.x)}
          cy={scaleY(point.y)}
          r="3"
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
          strokeDasharray="3,3"
        />
      ))}
    </svg>
  );
};

const LinePlot: React.FC<LinePlotProps> = ({
  data,
  predictions,
  width,
  height,
  config
}) => {
  const padding = 40;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  const sortedData = [...data].sort((a, b) => a.x - b.x);
  const sortedPredictions = predictions ? [...predictions].sort((a, b) => a.x - b.x) : [];

  const xMin = Math.min(...sortedData.map(d => d.x));
  const xMax = Math.max(...sortedData.map(d => d.x));
  const yMin = Math.min(...sortedData.map(d => d.y), ...(sortedPredictions.map(d => d.y) || []));
  const yMax = Math.max(...sortedData.map(d => d.y), ...(sortedPredictions.map(d => d.y) || []));

  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const scaleX = (x: number) => ((x - xMin) / xRange) * plotWidth + padding;
  const scaleY = (y: number) => height - (((y - yMin) / yRange) * plotHeight + padding);

  const createPath = (points: DataPoint[]) => {
    return points
      .map((point, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(point.x)} ${scaleY(point.y)}`)
      .join(' ');
  };

  return (
    <svg width={width} height={height} className="border border-gray-200 bg-white">
      {/* Grid */}
      <defs>
        <pattern id="line-grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f3f4f6" strokeWidth="1"/>
        </pattern>
      </defs>
      <rect width={plotWidth} height={plotHeight} x={padding} y={padding} fill="url(#line-grid)" />

      {/* Axes */}
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} 
            stroke="#374151" strokeWidth="2" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} 
            stroke="#374151" strokeWidth="2" />

      {/* Data line */}
      <path
        d={createPath(sortedData)}
        fill="none"
        stroke="#dc2626"
        strokeWidth="2"
      />

      {/* Prediction line */}
      {sortedPredictions.length > 0 && (
        <path
          d={createPath(sortedPredictions)}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
          strokeDasharray="5,5"
        />
      )}

      {/* Legend */}
      <g transform={`translate(${width - 150}, 30)`}>
        <rect width="140" height="50" fill="white" stroke="#e5e7eb" strokeWidth="1" rx="4" />
        <line x1="10" y1="20" x2="30" y2="20" stroke="#dc2626" strokeWidth="2" />
        <text x="35" y="25" className="text-xs text-gray-700">Actual</text>
        {sortedPredictions.length > 0 && (
          <>
            <line x1="10" y1="35" x2="30" y2="35" stroke="#3b82f6" strokeWidth="2" strokeDasharray="5,5" />
            <text x="35" y="40" className="text-xs text-gray-700">Predicted</text>
          </>
        )}
      </g>
    </svg>
  );
};

const Heatmap: React.FC<HeatmapProps> = ({
  data,
  width,
  height,
  config
}) => {
  const padding = 40;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  // Determine grid dimensions
  const gridSize = Math.sqrt(data.length);
  const cellWidth = plotWidth / gridSize;
  const cellHeight = plotHeight / gridSize;

  const values = data.map(d => d.y);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const valueRange = maxValue - minValue || 1;

  const getColor = (value: number) => {
    const intensity = (value - minValue) / valueRange;
    return `hsl(240, 100%, ${90 - intensity * 40}%)`;
  };

  return (
    <svg width={width} height={height} className="border border-gray-200 bg-white">
      {data.map((point, i) => {
        const row = Math.floor(i / gridSize);
        const col = i % gridSize;
        const x = col * cellWidth + padding;
        const y = row * cellHeight + padding;

        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={cellWidth}
            height={cellHeight}
            fill={getColor(point.y)}
            stroke="white"
            strokeWidth="1"
          />
        );
      })}
      
      {/* Color scale legend */}
      <g transform={`translate(${width - 30}, ${padding})`}>
        <defs>
          <linearGradient id="colorScale" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="hsl(240, 100%, 90%)" />
            <stop offset="100%" stopColor="hsl(240, 100%, 50%)" />
          </linearGradient>
        </defs>
        <rect width="20" height={plotHeight} fill="url(#colorScale)" stroke="#374151" strokeWidth="1" />
        <text x="25" y="15" className="text-xs text-gray-700">{maxValue.toFixed(2)}</text>
        <text x="25" y={plotHeight - 5} className="text-xs text-gray-700">{minValue.toFixed(2)}</text>
      </g>
    </svg>
  );
};

export const VisualizationArea: React.FC<VisualizationAreaProps> = ({
  data,
  predictions,
  config,
  title,
  isLoading = false,
  className = '',
  onPointHover,
  onPointClick
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 });

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width } = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: Math.max(400, width - 32), // Account for padding
          height: Math.max(300, width * 0.6) // Maintain aspect ratio
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const renderVisualization = () => {
    const { width, height } = dimensions;

    switch (config.type) {
      case 'scatter':
        return (
          <ScatterPlot
            data={data}
            predictions={predictions}
            width={width}
            height={height}
            config={config}
            onPointHover={onPointHover}
            onPointClick={onPointClick}
          />
        );
      
      case 'line':
        return (
          <LinePlot
            data={data}
            predictions={predictions}
            width={width}
            height={height}
            config={config}
          />
        );
      
      case 'heatmap':
        return (
          <Heatmap
            data={data}
            width={width}
            height={height}
            config={config}
          />
        );
      
      default:
        return (
          <div className="flex items-center justify-center h-64 bg-gray-50 border border-gray-200">
            <p className="text-gray-500">Unsupported visualization type: {config.type}</p>
          </div>
        );
    }
  };

  return (
    <div ref={containerRef} className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {/* Header */}
      {title && (
        <div className="px-4 py-3 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
            {isLoading && (
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span>Updating...</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Visualization Content */}
      <div className="p-4">
        {data.length === 0 ? (
          <div className="flex items-center justify-center h-64 bg-gray-50 border border-gray-200 rounded">
            <div className="text-center">
              <div className="text-gray-400 mb-2">
                <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-1">No Data Available</h3>
              <p className="text-gray-500">Generate or load data to see visualizations here.</p>
            </div>
          </div>
        ) : (
          <div className="relative">
            {isLoading && (
              <div className="absolute inset-0 bg-white bg-opacity-50 flex items-center justify-center z-10">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            )}
            {renderVisualization()}
          </div>
        )}
      </div>

      {/* Footer with data info */}
      {data.length > 0 && (
        <div className="px-4 py-2 bg-gray-50 border-t border-gray-200 text-sm text-gray-600">
          <div className="flex items-center justify-between">
            <span>{data.length} data points</span>
            {predictions && (
              <span>{predictions.length} predictions</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default VisualizationArea;
