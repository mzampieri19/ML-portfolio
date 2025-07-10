'use client';

import React, { useRef, useEffect, useState } from 'react';

interface ResidualPlotProps {
  data: Array<{ x: number; y: number }>;
  residuals: number[];
  predictions: number[];
  className?: string;
}

export const ResidualPlot: React.FC<ResidualPlotProps> = ({
  data,
  residuals,
  predictions,
  className = ''
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 300 });

  useEffect(() => {
    const updateDimensions = () => {
      if (svgRef.current) {
        const container = svgRef.current.parentElement;
        if (container) {
          const { width } = container.getBoundingClientRect();
          setDimensions({
            width: Math.max(400, width - 32),
            height: 300
          });
        }
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  if (residuals.length === 0 || predictions.length === 0) {
    return (
      <div className={`bg-white border border-gray-200 rounded-lg p-8 ${className}`}>
        <div className="text-center">
          <h3 className="text-lg font-medium text-gray-900 mb-1">Residual Plot</h3>
          <p className="text-gray-500">Train the model to see residual analysis.</p>
        </div>
      </div>
    );
  }

  const { width, height } = dimensions;
  const padding = 60;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  // Calculate bounds
  const predMin = Math.min(...predictions);
  const predMax = Math.max(...predictions);
  const residMin = Math.min(...residuals);
  const residMax = Math.max(...residuals);

  const predRange = predMax - predMin || 1;
  const residRange = Math.max(Math.abs(residMin), Math.abs(residMax)) || 1;

  // Add padding to bounds
  const predPadding = predRange * 0.1;
  const residPadding = residRange * 0.1;
  
  const predBounds = [predMin - predPadding, predMax + predPadding];
  const residBounds = [-residRange - residPadding, residRange + residPadding];

  const scaleX = (pred: number) => ((pred - predBounds[0]) / (predBounds[1] - predBounds[0])) * plotWidth + padding;
  const scaleY = (resid: number) => height - (((resid - residBounds[0]) / (residBounds[1] - residBounds[0])) * plotHeight + padding);

  // Calculate residual statistics
  const meanResidual = residuals.reduce((sum, r) => sum + r, 0) / residuals.length;
  const stdResidual = Math.sqrt(residuals.reduce((sum, r) => sum + Math.pow(r - meanResidual, 2), 0) / residuals.length);

  return (
    <div className={`bg-white border border-gray-200 rounded-lg ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Residual Plot</h3>
        <p className="text-sm text-gray-600">
          Residuals vs Fitted values - Check for patterns that indicate model issues
        </p>
      </div>

      {/* Plot Area */}
      <div className="p-4">
        <svg ref={svgRef} width={width} height={height} className="border border-gray-200 bg-white">
          {/* Grid */}
          <defs>
            <pattern id="residual-grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f3f4f6" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width={plotWidth} height={plotHeight} x={padding} y={padding} fill="url(#residual-grid)" />

          {/* Axes */}
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} 
                stroke="#374151" strokeWidth="2" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} 
                stroke="#374151" strokeWidth="2" />

          {/* Zero line */}
          <line 
            x1={padding} 
            y1={scaleY(0)} 
            x2={width - padding} 
            y2={scaleY(0)} 
            stroke="#ef4444" 
            strokeWidth="2" 
            strokeDasharray="5,5"
          />

          {/* Axis labels */}
          <text x={width / 2} y={height - 10} textAnchor="middle" className="text-sm text-gray-600">
            Fitted Values
          </text>
          <text x={15} y={height / 2} textAnchor="middle" transform={`rotate(-90 15 ${height / 2})`} 
                className="text-sm text-gray-600">
            Residuals
          </text>

          {/* Tick marks and labels */}
          {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
            const x = padding + ratio * plotWidth;
            const y = height - padding + ratio * plotHeight;
            const predValue = predBounds[0] + ratio * (predBounds[1] - predBounds[0]);
            const residValue = residBounds[0] + (1 - ratio) * (residBounds[1] - residBounds[0]);
            
            return (
              <g key={ratio}>
                {/* X-axis ticks */}
                <line x1={x} y1={height - padding} x2={x} y2={height - padding + 5} 
                      stroke="#374151" strokeWidth="1" />
                <text x={x} y={height - padding + 20} textAnchor="middle" className="text-xs text-gray-500">
                  {predValue.toFixed(1)}
                </text>
                
                {/* Y-axis ticks */}
                <line x1={padding} y1={y} x2={padding - 5} y2={y} 
                      stroke="#374151" strokeWidth="1" />
                <text x={padding - 10} y={y + 3} textAnchor="end" className="text-xs text-gray-500">
                  {residValue.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Residual points */}
          {residuals.map((residual, i) => {
            if (i < predictions.length) {
              const prediction = predictions[i];
              return (
                <circle
                  key={i}
                  cx={scaleX(prediction)}
                  cy={scaleY(residual)}
                  r="4"
                  fill="#3b82f6"
                  stroke="white"
                  strokeWidth="1"
                  className="hover:r-6 transition-all cursor-pointer"
                  opacity="0.7"
                />
              );
            }
            return null;
          })}

          {/* Trend line (LOWESS-like smooth)*/}
          {(() => {
            const sortedData = residuals
              .map((residual, i) => ({ pred: predictions[i], resid: residual }))
              .sort((a, b) => a.pred - b.pred);
            
            const smoothedPoints: { x: number, y: number }[] = [];
            const windowSize = Math.max(3, Math.floor(sortedData.length / 10));
            
            for (let i = 0; i < sortedData.length; i++) {
              const start = Math.max(0, i - Math.floor(windowSize / 2));
              const end = Math.min(sortedData.length, start + windowSize);
              const window = sortedData.slice(start, end);
              
              const avgResid = window.reduce((sum, d) => sum + d.resid, 0) / window.length;
              smoothedPoints.push({ x: sortedData[i].pred, y: avgResid });
            }
            
            const smoothPath = smoothedPoints
              .map((point, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(point.x)} ${scaleY(point.y)}`)
              .join(' ');
            
            return (
              <path
                d={smoothPath}
                fill="none"
                stroke="#10b981"
                strokeWidth="2"
                opacity="0.8"
              />
            );
          })()}
        </svg>
      </div>

      {/* Statistics */}
      <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="font-medium text-gray-700">Mean Residual:</span>
            <div className="text-gray-600">{meanResidual.toFixed(4)}</div>
          </div>
          <div>
            <span className="font-medium text-gray-700">Std Residual:</span>
            <div className="text-gray-600">{stdResidual.toFixed(4)}</div>
          </div>
          <div>
            <span className="font-medium text-gray-700">Min Residual:</span>
            <div className="text-gray-600">{Math.min(...residuals).toFixed(4)}</div>
          </div>
          <div>
            <span className="font-medium text-gray-700">Max Residual:</span>
            <div className="text-gray-600">{Math.max(...residuals).toFixed(4)}</div>
          </div>
        </div>
      </div>

      {/* Interpretation */}
      <div className="px-4 py-3 bg-blue-50 border-t border-blue-200">
        <h4 className="text-sm font-medium text-blue-800 mb-1">Interpretation:</h4>
        <ul className="text-xs text-blue-700 space-y-1">
          <li>• Random scatter around zero indicates good model fit</li>
          <li>• Patterns suggest non-linear relationships or heteroscedasticity</li>
          <li>• Outliers may indicate data quality issues or model limitations</li>
        </ul>
      </div>
    </div>
  );
};

export default ResidualPlot;
