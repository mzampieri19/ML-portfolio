'use client';

import React, { useRef, useEffect, useState } from 'react';
import { DataPoint, RegressionResult, VisualizationConfig } from '@/lib/visualize/types';

interface RegressionPlotProps {
  data: DataPoint[];
  predictions: DataPoint[];
  regressionResult: RegressionResult | null;
  config: VisualizationConfig;
  isTraining?: boolean;
  onPointHover?: (point: DataPoint | null) => void;
  onPointClick?: (point: DataPoint) => void;
}

export const RegressionPlot: React.FC<RegressionPlotProps> = ({
  data,
  predictions,
  regressionResult,
  config,
  isTraining = false,
  onPointHover,
  onPointClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 });
  const [hoveredPoint, setHoveredPoint] = useState<DataPoint | null>(null);

  useEffect(() => {
    const updateDimensions = () => {
      if (svgRef.current) {
        const container = svgRef.current.parentElement;
        if (container) {
          const { width } = container.getBoundingClientRect();
          setDimensions({
            width: Math.max(400, width - 32),
            height: Math.max(300, (width - 32) * 0.6)
          });
        }
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  if (data.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-8">
        <div className="text-center">
          <div className="text-gray-400 mb-2">
            <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-1">No Data Available</h3>
          <p className="text-gray-500">Generate or load data to see the regression plot.</p>
        </div>
      </div>
    );
  }

  const { width, height } = dimensions;
  const padding = 60;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  // Calculate data bounds
  const allPoints = [...data, ...predictions];
  const xMin = Math.min(...allPoints.map(d => d.x));
  const xMax = Math.max(...allPoints.map(d => d.x));
  const yMin = Math.min(...allPoints.map(d => d.y));
  const yMax = Math.max(...allPoints.map(d => d.y));

  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  // Add some padding to the bounds
  const xPadding = xRange * 0.1;
  const yPadding = yRange * 0.1;
  const xBounds = [xMin - xPadding, xMax + xPadding];
  const yBounds = [yMin - yPadding, yMax + yPadding];

  const scaleX = (x: number) => ((x - xBounds[0]) / (xBounds[1] - xBounds[0])) * plotWidth + padding;
  const scaleY = (y: number) => height - (((y - yBounds[0]) / (yBounds[1] - yBounds[0])) * plotHeight + padding);

  // Generate smooth regression line for visualization
  const generateRegressionLine = () => {
    if (!regressionResult) return '';

    const linePoints: { x: number, y: number }[] = [];
    const numPoints = 100;
    
    for (let i = 0; i <= numPoints; i++) {
      const x = xBounds[0] + (i / numPoints) * (xBounds[1] - xBounds[0]);
      let y = regressionResult.intercept;
      
      // Calculate polynomial prediction
      for (let d = 1; d <= regressionResult.coefficients.length; d++) {
        if (regressionResult.coefficients[d - 1] !== undefined) {
          y += regressionResult.coefficients[d - 1] * Math.pow(x, d);
        }
      }
      
      linePoints.push({ x, y });
    }

    return linePoints
      .map((point, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(point.x)} ${scaleY(point.y)}`)
      .join(' ');
  };

  const handlePointHover = (point: DataPoint | null) => {
    setHoveredPoint(point);
    onPointHover?.(point);
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Regression Plot</h3>
          {isTraining && (
            <div className="flex items-center space-x-2 text-sm text-gray-500">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
              <span>Training...</span>
            </div>
          )}
        </div>
      </div>

      {/* Plot Area */}
      <div className="p-4">
        <div className="relative">
          {isTraining && (
            <div className="absolute inset-0 bg-white bg-opacity-50 flex items-center justify-center z-10">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
          )}
          
          <svg ref={svgRef} width={width} height={height} className="border border-gray-200 bg-white">
            {/* Grid */}
            <defs>
              <pattern id="regression-grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f3f4f6" strokeWidth="1"/>
              </pattern>
            </defs>
            <rect width={plotWidth} height={plotHeight} x={padding} y={padding} fill="url(#regression-grid)" />

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
              const xValue = xBounds[0] + ratio * (xBounds[1] - xBounds[0]);
              const yValue = yBounds[0] + (1 - ratio) * (yBounds[1] - yBounds[0]);
              
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

            {/* Regression line */}
            {regressionResult && config.showPredictionLine && (
              <path
                d={generateRegressionLine()}
                fill="none"
                stroke="#3b82f6"
                strokeWidth="3"
                opacity="0.8"
              />
            )}

            {/* Prediction error lines */}
            {regressionResult && predictions.length > 0 && (
              <g opacity="0.3">
                {data.map((point, i) => {
                  if (i < predictions.length) {
                    return (
                      <line
                        key={`error-${i}`}
                        x1={scaleX(point.x)}
                        y1={scaleY(point.y)}
                        x2={scaleX(predictions[i].x)}
                        y2={scaleY(predictions[i].y)}
                        stroke="#ef4444"
                        strokeWidth="1"
                        strokeDasharray="2,2"
                      />
                    );
                  }
                  return null;
                })}
              </g>
            )}

            {/* Data points */}
            {data.map((point, i) => (
              <circle
                key={`data-${i}`}
                cx={scaleX(point.x)}
                cy={scaleY(point.y)}
                r="5"
                fill="#dc2626"
                stroke="white"
                strokeWidth="2"
                className="cursor-pointer hover:r-7 transition-all"
                onMouseEnter={() => handlePointHover(point)}
                onMouseLeave={() => handlePointHover(null)}
                onClick={() => onPointClick?.(point)}
              />
            ))}

            {/* Prediction points */}
            {predictions.map((point, i) => (
              <circle
                key={`pred-${i}`}
                cx={scaleX(point.x)}
                cy={scaleY(point.y)}
                r="3"
                fill="none"
                stroke="#3b82f6"
                strokeWidth="2"
              />
            ))}

            {/* Hover tooltip */}
            {hoveredPoint && (
              <g>
                <rect
                  x={scaleX(hoveredPoint.x) + 10}
                  y={scaleY(hoveredPoint.y) - 25}
                  width="80"
                  height="20"
                  fill="rgba(0, 0, 0, 0.8)"
                  rx="4"
                />
                <text
                  x={scaleX(hoveredPoint.x) + 50}
                  y={scaleY(hoveredPoint.y) - 10}
                  textAnchor="middle"
                  fill="white"
                  className="text-xs"
                >
                  {`(${hoveredPoint.x.toFixed(2)}, ${hoveredPoint.y.toFixed(2)})`}
                </text>
              </g>
            )}
          </svg>
        </div>
      </div>

      {/* Legend */}
      <div className="px-4 pb-4">
        <div className="flex items-center justify-center space-x-6 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-600 rounded-full mr-2"></div>
            <span className="text-gray-600">Actual Data</span>
          </div>
          {regressionResult && (
            <>
              <div className="flex items-center">
                <div className="w-4 h-1 bg-blue-600 mr-2"></div>
                <span className="text-gray-600">Regression Line</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 border-2 border-blue-600 rounded-full mr-2"></div>
                <span className="text-gray-600">Predictions</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Footer with equation */}
      {regressionResult && (
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
          <div className="text-center">
            <span className="text-sm font-medium text-gray-700">Equation: </span>
            <span className="text-sm text-gray-600 font-mono">
              y = {regressionResult.intercept.toFixed(3)}
              {regressionResult.coefficients.map((coef, i) => 
                ` ${coef >= 0 ? '+' : ''}${coef.toFixed(3)}x${i + 1 > 1 ? `^${i + 1}` : ''}`
              ).join('')}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default RegressionPlot;
