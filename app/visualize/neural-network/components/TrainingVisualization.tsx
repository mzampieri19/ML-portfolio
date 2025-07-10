'use client';

import React, { useEffect, useRef, useState } from 'react';
import { DataPoint, TrainingHistory } from '@/lib/visualize/types';

interface TrainingVisualizationProps {
  data: DataPoint[];
  predictions: DataPoint[];
  trainingHistory: TrainingHistory[];
  isTraining: boolean;
  currentEpoch: number;
  className?: string;
}

export const TrainingVisualization: React.FC<TrainingVisualizationProps> = ({
  data,
  predictions,
  trainingHistory,
  isTraining,
  currentEpoch,
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lossChartRef = useRef<HTMLCanvasElement>(null);
  const [viewMode, setViewMode] = useState<'data' | 'predictions' | 'both'>('both');
  const [showDecisionBoundary, setShowDecisionBoundary] = useState(true);

  // Draw the data and predictions visualization
  const drawDataVisualization = () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Calculate bounds
    const allPoints = [...data, ...predictions];
    if (allPoints.length === 0) return;

    const margin = 40;
    const xMin = Math.min(...allPoints.map(p => p.x)) - 0.5;
    const xMax = Math.max(...allPoints.map(p => p.x)) + 0.5;
    const yMin = Math.min(...allPoints.map(p => p.y)) - 0.5;
    const yMax = Math.max(...allPoints.map(p => p.y)) + 0.5;

    const scaleX = (width - 2 * margin) / (xMax - xMin);
    const scaleY = (height - 2 * margin) / (yMax - yMin);

    const toCanvasX = (x: number) => margin + (x - xMin) * scaleX;
    const toCanvasY = (y: number) => height - margin - (y - yMin) * scaleY;

    // Draw decision boundary if predictions are available
    if (showDecisionBoundary && predictions.length > 0) {
      drawDecisionBoundary(ctx, width, height, margin, xMin, xMax, yMin, yMax, scaleX, scaleY);
    }

    // Draw grid
    ctx.strokeStyle = '#E5E7EB';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = margin + (i / 10) * (width - 2 * margin);
      ctx.beginPath();
      ctx.moveTo(x, margin);
      ctx.lineTo(x, height - margin);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = margin + (i / 10) * (height - 2 * margin);
      ctx.beginPath();
      ctx.moveTo(margin, y);
      ctx.lineTo(width - margin, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(margin, height - margin);
    ctx.lineTo(width - margin, height - margin);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, height - margin);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Feature 1', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Feature 2', 0, 0);
    ctx.restore();

    // Draw data points
    if (viewMode === 'data' || viewMode === 'both') {
      data.forEach(point => {
        const x = toCanvasX(point.x);
        const y = toCanvasY(point.y);
        
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = point.label === 1 ? '#3B82F6' : '#EF4444';
        ctx.fill();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }

    // Draw predictions
    if (viewMode === 'predictions' || viewMode === 'both') {
      predictions.forEach(point => {
        const x = toCanvasX(point.x);
        const y = toCanvasY(point.y);
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = point.label === 1 ? '#10B981' : '#F59E0B';
        ctx.fill();
        
        // Add a different stroke style for predictions
        if (viewMode === 'both') {
          ctx.strokeStyle = '#000000';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      });
    }

    // Draw legend
    if (viewMode === 'both') {
      const legendX = width - 150;
      const legendY = margin + 20;
      
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(legendX - 10, legendY - 10, 140, 80);
      ctx.strokeStyle = '#D1D5DB';
      ctx.lineWidth = 1;
      ctx.strokeRect(legendX - 10, legendY - 10, 140, 80);
      
      // Actual data legend
      ctx.beginPath();
      ctx.arc(legendX, legendY, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#3B82F6';
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      ctx.fillStyle = '#374151';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Actual (Class 1)', legendX + 10, legendY + 4);
      
      ctx.beginPath();
      ctx.arc(legendX, legendY + 20, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#EF4444';
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      ctx.fillText('Actual (Class 0)', legendX + 10, legendY + 24);
      
      // Predictions legend
      ctx.beginPath();
      ctx.arc(legendX, legendY + 40, 3, 0, 2 * Math.PI);
      ctx.fillStyle = '#10B981';
      ctx.fill();
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.stroke();
      
      ctx.fillText('Pred (Class 1)', legendX + 10, legendY + 44);
      
      ctx.beginPath();
      ctx.arc(legendX, legendY + 60, 3, 0, 2 * Math.PI);
      ctx.fillStyle = '#F59E0B';
      ctx.fill();
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.stroke();
      
      ctx.fillText('Pred (Class 0)', legendX + 10, legendY + 64);
    }
  };

  // Draw decision boundary (simplified version)
  const drawDecisionBoundary = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    margin: number,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    scaleX: number,
    scaleY: number
  ) => {
    // Create a simple decision boundary visualization
    // In a real implementation, this would use the actual model predictions
    const resolution = 20;
    const imageData = ctx.createImageData(width - 2 * margin, height - 2 * margin);
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = xMin + (i / resolution) * (xMax - xMin);
        const y = yMin + (j / resolution) * (yMax - yMin);
        
        // Simple decision boundary based on distance from center
        const centerX = (xMin + xMax) / 2;
        const centerY = (yMin + yMax) / 2;
        const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
        const prediction = distance < 1 ? 1 : 0;
        
        const canvasX = Math.floor((x - xMin) * scaleX);
        const canvasY = Math.floor((yMax - y) * scaleY);
        
        if (canvasX >= 0 && canvasX < width - 2 * margin && 
            canvasY >= 0 && canvasY < height - 2 * margin) {
          const pixelIndex = (canvasY * (width - 2 * margin) + canvasX) * 4;
          
          if (prediction === 1) {
            imageData.data[pixelIndex] = 59;     // R
            imageData.data[pixelIndex + 1] = 130; // G
            imageData.data[pixelIndex + 2] = 246; // B
            imageData.data[pixelIndex + 3] = 30;  // A (low opacity)
          } else {
            imageData.data[pixelIndex] = 239;     // R
            imageData.data[pixelIndex + 1] = 68;  // G
            imageData.data[pixelIndex + 2] = 68;  // B
            imageData.data[pixelIndex + 3] = 30;  // A (low opacity)
          }
        }
      }
    }
    
    ctx.putImageData(imageData, margin, margin);
  };

  // Draw loss chart
  const drawLossChart = () => {
    if (!lossChartRef.current || trainingHistory.length === 0) return;

    const canvas = lossChartRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = rect.width;
    const height = rect.height;
    const margin = 40;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    const maxLoss = Math.max(...trainingHistory.map(h => h.loss));
    const minLoss = Math.min(...trainingHistory.map(h => h.loss));
    const lossRange = maxLoss - minLoss || 1;

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(margin, height - margin);
    ctx.lineTo(width - margin, height - margin);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, height - margin);
    ctx.stroke();

    // Draw loss curve
    if (trainingHistory.length > 1) {
      ctx.strokeStyle = '#3B82F6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      trainingHistory.forEach((history, index) => {
        const x = margin + (index / (trainingHistory.length - 1)) * (width - 2 * margin);
        const y = height - margin - ((history.loss - minLoss) / lossRange) * (height - 2 * margin);
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    }

    // Draw current epoch indicator
    if (isTraining && currentEpoch > 0) {
      const x = margin + (currentEpoch / trainingHistory.length) * (width - 2 * margin);
      ctx.strokeStyle = '#EF4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(x, margin);
      ctx.lineTo(x, height - margin);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

    // Draw loss value
    if (trainingHistory.length > 0) {
      const currentLoss = trainingHistory[trainingHistory.length - 1].loss;
      ctx.fillStyle = '#3B82F6';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`Loss: ${currentLoss.toFixed(4)}`, width - margin, margin + 20);
    }
  };

  // Redraw visualizations when data changes
  useEffect(() => {
    drawDataVisualization();
  }, [data, predictions, viewMode, showDecisionBoundary]);

  useEffect(() => {
    drawLossChart();
  }, [trainingHistory, isTraining, currentEpoch]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setTimeout(() => {
        drawDataVisualization();
        drawLossChart();
      }, 100);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className={`bg-white border border-gray-200 rounded-lg ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Training Visualization</h3>
          <div className="flex items-center space-x-2">
            {/* View Mode Controls */}
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as 'data' | 'predictions' | 'both')}
              className="px-2 py-1 border border-gray-300 rounded text-sm"
            >
              <option value="data">Data Only</option>
              <option value="predictions">Predictions Only</option>
              <option value="both">Both</option>
            </select>
            
            {/* Decision Boundary Toggle */}
            <label className="flex items-center text-sm">
              <input
                type="checkbox"
                checked={showDecisionBoundary}
                onChange={(e) => setShowDecisionBoundary(e.target.checked)}
                className="mr-1"
              />
              Decision Boundary
            </label>
          </div>
        </div>
        <p className="text-sm text-gray-600 mt-1">
          {isTraining 
            ? `Training in progress... Epoch ${currentEpoch}`
            : 'Real-time visualization of model predictions and training progress'
          }
        </p>
      </div>

      {/* Visualizations */}
      <div className="p-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Data Visualization */}
          <div>
            <h4 className="text-md font-medium text-gray-700 mb-2">Data & Predictions</h4>
            <canvas
              ref={canvasRef}
              className="w-full border border-gray-200 rounded"
              style={{ height: '300px' }}
            />
          </div>

          {/* Loss Chart */}
          <div>
            <h4 className="text-md font-medium text-gray-700 mb-2">Training Progress</h4>
            <canvas
              ref={lossChartRef}
              className="w-full border border-gray-200 rounded"
              style={{ height: '300px' }}
            />
          </div>
        </div>

        {/* Training Statistics */}
        {trainingHistory.length > 0 && (
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-gray-50 rounded">
              <div className="text-sm text-gray-600">Current Loss</div>
              <div className="text-lg font-semibold text-blue-600">
                {trainingHistory[trainingHistory.length - 1]?.loss.toFixed(4) || 'N/A'}
              </div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded">
              <div className="text-sm text-gray-600">Accuracy</div>
              <div className="text-lg font-semibold text-green-600">
                {trainingHistory[trainingHistory.length - 1]?.accuracy 
                  ? `${(trainingHistory[trainingHistory.length - 1].accuracy! * 100).toFixed(1)}%`
                  : 'N/A'
                }
              </div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded">
              <div className="text-sm text-gray-600">Epochs</div>
              <div className="text-lg font-semibold text-purple-600">
                {currentEpoch}
              </div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded">
              <div className="text-sm text-gray-600">Data Points</div>
              <div className="text-lg font-semibold text-orange-600">
                {data.length}
              </div>
            </div>
          </div>
        )}

        {/* Progress Bar */}
        {isTraining && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Training Progress</span>
              <span>{currentEpoch} epochs completed</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${Math.min((currentEpoch / trainingHistory.length) * 100, 100)}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingVisualization;
