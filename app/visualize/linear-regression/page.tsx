'use client';

import React, { useState, useEffect } from 'react';
import { DataPoint, LinearRegressionParams, RegressionResult } from '@/lib/visualize/types';
import { linearRegressionNormal, linearRegressionGradientDescent } from '@/lib/visualize/algorithms';
import { useParameterControl } from '@/hooks/useParameterControl';
import { useDataset } from '@/hooks/useDataset';
import ControlPanel from '../components/ControlPanel';
import MetricsDashboard from '../components/MetricsDashboard';
import InfoSidebar, { commonInfoSections } from '../components/InfoSidebar';
import RegressionPlot from './components/RegressionPlot';
import ResidualPlot from './components/ResidualPlot';
import CostFunction from './components/CostFunction';

const initialParams: LinearRegressionParams = {
  polynomialDegree: 1,
  regularizationType: 'none',
  regularizationStrength: 0.01,
  learningRate: 0.01,
  iterations: 100
};

export default function LinearRegressionPage() {
  const [isTraining, setIsTraining] = useState(false);
  const [showResidualPlot, setShowResidualPlot] = useState(false);
  const [showCostFunction, setShowCostFunction] = useState(false);
  const [regressionResult, setRegressionResult] = useState<RegressionResult | null>(null);
  const [trainingMethod, setTrainingMethod] = useState<'normal' | 'gradient'>('normal');
  const [infoSidebarOpen, setInfoSidebarOpen] = useState(true);

  // Custom hooks
  const { parameters, updateParameter, resetParameters } = useParameterControl({
    initialParameters: initialParams
  });

  const { currentDataset, generateDataset, isLoading: datasetLoading } = useDataset({
    onDatasetChange: () => setRegressionResult(null) // Clear results when dataset changes
  });

  // Generate initial dataset on component mount
  useEffect(() => {
    if (!currentDataset) {
      generateDataset('linear', { numPoints: 50, noise: 0.1 });
    }
  }, [currentDataset, generateDataset]);

  const handleTrainModel = async () => {
    if (!currentDataset) return;
    
    setIsTraining(true);
    try {
      let result: RegressionResult;
      
      if (trainingMethod === 'gradient') {
        result = await linearRegressionGradientDescent(
          currentDataset.points,
          parameters,
          (epoch, loss, weights) => {
            // Optional: Update progress during training
            console.log(`Epoch ${epoch}: Loss ${loss}`);
          }
        );
      } else {
        result = await linearRegressionNormal(currentDataset.points, parameters);
      }
      
      setRegressionResult(result);
    } catch (error) {
      console.error('Training failed:', error);
    } finally {
      setIsTraining(false);
    }
  };

  const handleGenerateDataset = async (type: string, options?: any) => {
    setRegressionResult(null);
    await generateDataset(type, options);
  };

  const handleParameterChange = (parameterId: string, value: any) => {
    updateParameter(parameterId as keyof LinearRegressionParams, value);
    setRegressionResult(null);
  };

  const getDataPoints = (): DataPoint[] => {
    return currentDataset?.points || [];
  };

  const getPredictionPoints = (): DataPoint[] => {
    if (!regressionResult || !currentDataset) return [];
    
    return currentDataset.points.map((point, i) => ({
      id: `pred-${i}`,
      x: point.x,
      y: regressionResult.predictions[i],
      label: point.label
    }));
  };

  const controlPanelConfig = {
    title: 'Linear Regression Parameters',
    controls: [
      // Flatten all controls into a single array for the interface
      {
        id: 'polynomialDegree',
        label: 'Polynomial Degree',
        type: 'slider' as const,
        min: 1,
        max: 10,
        step: 1,
        value: parameters.polynomialDegree
      },
      {
        id: 'regularizationType',
        label: 'Regularization',
        type: 'select' as const,
        options: [
          { value: 'none', label: 'None' },
          { value: 'ridge', label: 'Ridge (L2)' },
          { value: 'lasso', label: 'Lasso (L1)' }
        ],
        value: parameters.regularizationType
      },
      {
        id: 'regularizationStrength',
        label: 'Regularization Strength',
        type: 'slider' as const,
        min: 0,
        max: 1,
        step: 0.01,
        value: parameters.regularizationStrength,
        disabled: parameters.regularizationType === 'none'
      },
      {
        id: 'trainingMethod',
        label: 'Algorithm',
        type: 'select' as const,
        options: [
          { value: 'normal', label: 'Normal Equation' },
          { value: 'gradient', label: 'Gradient Descent' }
        ],
        value: trainingMethod
      },
      {
        id: 'learningRate',
        label: 'Learning Rate',
        type: 'slider' as const,
        min: 0.001,
        max: 0.1,
        step: 0.001,
        value: parameters.learningRate,
        disabled: trainingMethod === 'normal'
      },
      {
        id: 'iterations',
        label: 'Max Iterations',
        type: 'slider' as const,
        min: 10,
        max: 1000,
        step: 10,
        value: parameters.iterations,
        disabled: trainingMethod === 'normal'
      }
    ],
    groups: [
      {
        title: 'Model Configuration',
        controls: [
          {
            id: 'polynomialDegree',
            label: 'Polynomial Degree',
            type: 'slider' as const,
            min: 1,
            max: 10,
            step: 1,
            value: parameters.polynomialDegree
          },
          {
            id: 'regularizationType',
            label: 'Regularization',
            type: 'select' as const,
            options: [
              { value: 'none', label: 'None' },
              { value: 'ridge', label: 'Ridge (L2)' },
              { value: 'lasso', label: 'Lasso (L1)' }
            ],
            value: parameters.regularizationType
          },
          {
            id: 'regularizationStrength',
            label: 'Regularization Strength',
            type: 'slider' as const,
            min: 0,
            max: 1,
            step: 0.01,
            value: parameters.regularizationStrength,
            disabled: parameters.regularizationType === 'none'
          }
        ]
      },
      {
        title: 'Training Method',
        controls: [
          {
            id: 'trainingMethod',
            label: 'Algorithm',
            type: 'select' as const,
            options: [
              { value: 'normal', label: 'Normal Equation' },
              { value: 'gradient', label: 'Gradient Descent' }
            ],
            value: trainingMethod
          },
          {
            id: 'learningRate',
            label: 'Learning Rate',
            type: 'slider' as const,
            min: 0.001,
            max: 0.1,
            step: 0.001,
            value: parameters.learningRate,
            disabled: trainingMethod === 'normal'
          },
          {
            id: 'iterations',
            label: 'Max Iterations',
            type: 'slider' as const,
            min: 10,
            max: 1000,
            step: 10,
            value: parameters.iterations,
            disabled: trainingMethod === 'normal'
          }
        ]
      },
      {
        title: 'Dataset Options',
        controls: [
          {
            id: 'generateLinear',
            label: 'Generate Linear Data',
            type: 'button' as const
          },
          {
            id: 'generatePolynomial',
            label: 'Generate Polynomial Data',
            type: 'button' as const
          },
          {
            id: 'generateSine',
            label: 'Generate Sine Wave',
            type: 'button' as const
          }
        ]
      }
    ]
  };

  const visualizationConfig = {
    type: 'scatter' as const,
    width: 600,
    height: 400,
    margin: { top: 20, right: 20, bottom: 40, left: 40 },
    showGrid: true,
    showAxes: true,
    interactive: true,
    theme: 'light' as const,
    xAxisLabel: 'X',
    yAxisLabel: 'Y',
    showPredictionLine: true
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Linear Regression Explorer
          </h1>
          <p className="text-gray-600">
            Experiment with linear regression algorithms and see how different parameters affect model performance.
          </p>
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Control Panel */}
          <div className="col-span-12 lg:col-span-3">
            <ControlPanel
              config={controlPanelConfig}
              values={parameters}
              onChange={(id: string, value: any) => {
                if (id === 'trainingMethod') {
                  setTrainingMethod(value);
                } else if (id === 'generateLinear') {
                  handleGenerateDataset('linear', { numPoints: 50, noise: 0.1 });
                } else if (id === 'generatePolynomial') {
                  handleGenerateDataset('polynomial', { numPoints: 50, noise: 0.1, degree: 2 });
                } else if (id === 'generateSine') {
                  handleGenerateDataset('sine', { numPoints: 50, noise: 0.1 });
                } else {
                  handleParameterChange(id, value);
                }
              }}
              className="sticky top-4"
            />

            {/* Action Buttons */}
            <div className="mt-6 space-y-3">
              <button
                onClick={handleTrainModel}
                disabled={isTraining || datasetLoading || !currentDataset}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isTraining ? 'Training...' : 'Train Model'}
              </button>

              <button
                onClick={() => {
                  setRegressionResult(null);
                }}
                disabled={!regressionResult}
                className="w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Clear Model
              </button>

              <button
                onClick={resetParameters}
                className="w-full bg-gray-500 text-white py-2 px-4 rounded-lg hover:bg-gray-600 transition-colors"
              >
                Reset Parameters
              </button>
            </div>

            {/* View Options */}
            <div className="mt-6 space-y-2">
              <h3 className="text-sm font-medium text-gray-700">View Options</h3>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showResidualPlot}
                  onChange={(e) => setShowResidualPlot(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm text-gray-600">Show Residual Plot</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showCostFunction}
                  onChange={(e) => setShowCostFunction(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm text-gray-600">Show Cost Function</span>
              </label>
            </div>
          </div>

          {/* Main Content */}
          <div className={`col-span-12 ${infoSidebarOpen ? 'lg:col-span-6' : 'lg:col-span-9'}`}>
            <div className="space-y-6">
              {/* Metrics Dashboard */}
              {regressionResult && (
                <MetricsDashboard
                  metrics={{
                    loss: regressionResult.metrics?.mse || null,
                    mse: regressionResult.metrics?.mse || null,
                    mae: regressionResult.metrics?.mae || null,
                    r2: regressionResult.metrics?.r2Score || null,
                    accuracy: null,
                    precision: null,
                    recall: null,
                    f1Score: null,
                    lossHistory: regressionResult.convergenceHistory || [],
                    validationLossHistory: undefined,
                    trainingProgress: undefined,
                    additionalMetrics: undefined
                  }}
                  isTraining={isTraining}
                />
              )}

              {/* Main Regression Plot */}
              <RegressionPlot
                data={getDataPoints()}
                predictions={getPredictionPoints()}
                regressionResult={regressionResult}
                config={visualizationConfig}
                isTraining={isTraining}
              />

              {/* Additional Plots */}
              {showResidualPlot && regressionResult && (
                <ResidualPlot
                  data={getDataPoints()}
                  residuals={regressionResult.residuals}
                  predictions={regressionResult.predictions}
                />
              )}

              {showCostFunction && regressionResult?.convergenceHistory && (
                <CostFunction
                  convergenceHistory={regressionResult.convergenceHistory}
                  parameters={parameters}
                />
              )}
            </div>
          </div>

          {/* Info Sidebar */}
          <div className={`col-span-12 ${infoSidebarOpen ? 'lg:col-span-3' : 'lg:col-span-0'}`}>
            <InfoSidebar
              sections={commonInfoSections.linearRegression}
              isOpen={infoSidebarOpen}
              onToggle={() => setInfoSidebarOpen(!infoSidebarOpen)}
              className="sticky top-4 h-[calc(100vh-2rem)] overflow-hidden"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
