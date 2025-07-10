'use client';

import React, { useRef, useEffect, useState } from 'react';
import { LinearRegressionParams } from '@/lib/visualize/types';

interface CostFunctionProps {
  convergenceHistory: number[];
  parameters: LinearRegressionParams;
  className?: string;
}

const CostFunction: React.FC<CostFunctionProps> = ({
  convergenceHistory,
  parameters,
  className = ''
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 300 });
  const [showLogScale, setShowLogScale] = useState(false);

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

  if (convergenceHistory.length === 0) {
    return (
      <div className={`bg-white border border-gray-200 rounded-lg p-8 ${className}`}>
        <div className="text-center">
          <h3 className="text-lg font-medium text-gray-900 mb-1">Cost Function Evolution</h3>
          <p className="text-gray-500">Use gradient descent to see cost function evolution.</p>
        </div>
      </div>
    );
  }

  const { width, height } = dimensions;
  const padding = 60;
  const plotWidth = width - 2 * padding;
  const plotHeight = height - 2 * padding;

  // Process data for display
  const costs = showLogScale 
    ? convergenceHistory.map(cost => Math.log10(Math.max(cost, 1e-10)))
    : convergenceHistory;

  const minCost = Math.min(...costs);
  const maxCost = Math.max(...costs);
  const costRange = maxCost - minCost || 1;

  // Add padding to bounds
  const costPadding = costRange * 0.1;
  const costBounds = [minCost - costPadding, maxCost + costPadding];

  const scaleX = (epoch: number) => (epoch / (convergenceHistory.length - 1)) * plotWidth + padding;
  const scaleY = (cost: number) => height - (((cost - costBounds[0]) / (costBounds[1] - costBounds[0])) * plotHeight + padding);

  // Create cost function path
  const costPath = costs
    .map((cost, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(cost)}`)
    .join(' ');

  // Calculate convergence metrics
  const finalCost = convergenceHistory[convergenceHistory.length - 1];
  const initialCost = convergenceHistory[0];
  const improvement = ((initialCost - finalCost) / initialCost) * 100;
  const converged = convergenceHistory.length > 10 && 
    Math.abs(convergenceHistory[convergenceHistory.length - 1] - convergenceHistory[convergenceHistory.length - 10]) < 0.0001;

  // Calculate rate of convergence (slope of last 10% of points)
  const lastTenPercent = Math.floor(convergenceHistory.length * 0.1);
  const recentCosts = convergenceHistory.slice(-lastTenPercent);
  const avgRecentSlope = recentCosts.length > 1 
    ? (recentCosts[recentCosts.length - 1] - recentCosts[0]) / (recentCosts.length - 1)
    : 0;

  return (
    <div className={`bg-white border border-gray-200 rounded-lg ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">Cost Function Evolution</h3>
            <p className="text-sm text-gray-600">
              Mean Squared Error over {convergenceHistory.length} iterations
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <label className="flex items-center text-sm text-gray-600">
              <input
                type="checkbox"
                checked={showLogScale}
                onChange={(e) => setShowLogScale(e.target.checked)}
                className="mr-1"
              />
              Log Scale
            </label>
          </div>
        </div>
      </div>

      {/* Plot Area */}
      <div className="p-4">
        <svg ref={svgRef} width={width} height={height} className="border border-gray-200 bg-white">
          {/* Grid */}
          <defs>
            <pattern id="cost-grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f3f4f6" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width={plotWidth} height={plotHeight} x={padding} y={padding} fill="url(#cost-grid)" />

          {/* Axes */}
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} 
                stroke="#374151" strokeWidth="2" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} 
                stroke="#374151" strokeWidth="2" />

          {/* Axis labels */}
          <text x={width / 2} y={height - 10} textAnchor="middle" className="text-sm text-gray-600">
            Iteration
          </text>
          <text x={15} y={height / 2} textAnchor="middle" transform={`rotate(-90 15 ${height / 2})`} 
                className="text-sm text-gray-600">
            {showLogScale ? 'log₁₀(Cost)' : 'Cost (MSE)'}
          </text>

          {/* Tick marks and labels */}
          {[0, 0.25, 0.5, 0.75, 1].map(ratio => {
            const x = padding + ratio * plotWidth;
            const y = height - padding + ratio * plotHeight;
            const epochValue = Math.round(ratio * (convergenceHistory.length - 1));
            const costValue = costBounds[0] + (1 - ratio) * (costBounds[1] - costBounds[0]);
            
            return (
              <g key={ratio}>
                {/* X-axis ticks */}
                <line x1={x} y1={height - padding} x2={x} y2={height - padding + 5} 
                      stroke="#374151" strokeWidth="1" />
                <text x={x} y={height - padding + 20} textAnchor="middle" className="text-xs text-gray-500">
                  {epochValue}
                </text>
                
                {/* Y-axis ticks */}
                <line x1={padding} y1={y} x2={padding - 5} y2={y} 
                      stroke="#374151" strokeWidth="1" />
                <text x={padding - 10} y={y + 3} textAnchor="end" className="text-xs text-gray-500">
                  {showLogScale ? costValue.toFixed(1) : costValue.toFixed(4)}
                </text>
              </g>
            );
          })}

          {/* Cost function line */}
          <path
            d={costPath}
            fill="none"
            stroke="#dc2626"
            strokeWidth="2"
          />

          {/* Start and end points */}
          <circle
            cx={scaleX(0)}
            cy={scaleY(costs[0])}
            r="4"
            fill="#dc2626"
            stroke="white"
            strokeWidth="2"
          />
          <circle
            cx={scaleX(convergenceHistory.length - 1)}
            cy={scaleY(costs[costs.length - 1])}
            r="4"
            fill="#059669"
            stroke="white"
            strokeWidth="2"
          />

          {/* Convergence indicator */}
          {converged && (
            <text
              x={width - padding - 10}
              y={padding + 20}
              textAnchor="end"
              className="text-sm text-green-600 font-medium"
            >
              Converged ✓
            </text>
          )}
        </svg>
      </div>

      {/* Statistics */}
      <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="font-medium text-gray-700">Initial Cost:</span>
            <div className="text-gray-600">{initialCost.toExponential(3)}</div>
          </div>
          <div>
            <span className="font-medium text-gray-700">Final Cost:</span>
            <div className="text-gray-600">{finalCost.toExponential(3)}</div>
          </div>
          <div>
            <span className="font-medium text-gray-700">Improvement:</span>
            <div className="text-gray-600">{improvement.toFixed(2)}%</div>
          </div>
          <div>
            <span className="font-medium text-gray-700">Learning Rate:</span>
            <div className="text-gray-600">{parameters.learningRate}</div>
          </div>
        </div>
      </div>

      {/* Analysis */}
      <div className="px-4 py-3 bg-yellow-50 border-t border-yellow-200">
        <h4 className="text-sm font-medium text-yellow-800 mb-1">Analysis:</h4>
        <div className="text-xs text-yellow-700 space-y-1">
          {avgRecentSlope > 0.001 && (
            <div>⚠️ Cost still decreasing rapidly - consider more iterations</div>
          )}
          {avgRecentSlope < -0.001 && (
            <div>⚠️ Cost increasing - learning rate may be too high</div>
          )}
          {Math.abs(avgRecentSlope) <= 0.001 && (
            <div>✅ Cost stabilized - good convergence</div>
          )}
          {improvement < 1 && (
            <div>⚠️ Little improvement - check learning rate or model complexity</div>
          )}
          {improvement > 99 && (
            <div>✅ Excellent improvement - model learning effectively</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CostFunction;
