'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { DataPoint, NeuralNetworkParams } from '@/lib/visualize/types';
import { useParameterControl } from '@/hooks/useParameterControl';
import { useDataset } from '@/hooks/useDataset';
import { useModelTraining } from '@/hooks/useModelTraining';
import ControlPanel from '../components/ControlPanel';
import MetricsDashboard from '../components/MetricsDashboard';
import InfoSidebar, { commonInfoSections } from '../components/InfoSidebar';
import NetworkDiagram from './components/NetworkDiagram';
import TrainingVisualization from './components/TrainingVisualization';
import LayerControls from './components/LayerControls';

const initialParams: NeuralNetworkParams = {
  layers: [
    { 
      id: 'layer-1',
      type: 'dense', 
      units: 4, 
      activation: 'relu',
      position: { x: 0, y: 0 }
    },
    { 
      id: 'layer-2',
      type: 'dense', 
      units: 3, 
      activation: 'relu',
      position: { x: 0, y: 0 }
    },
    { 
      id: 'layer-3',
      type: 'dense', 
      units: 1, 
      activation: 'linear',
      position: { x: 0, y: 0 }
    }
  ],
  learningRate: 0.01,
  batchSize: 32,
  epochs: 100,
  optimizer: 'adam',
  activation: 'relu'
};

export default function NeuralNetworkPage() {
  const [isTraining, setIsTraining] = useState(false);
  const [showNetworkDiagram, setShowNetworkDiagram] = useState(true);
  const [showTrainingViz, setShowTrainingViz] = useState(true);
  const [infoSidebarOpen, setInfoSidebarOpen] = useState(true);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);

  // Custom hooks
  const { parameters, updateParameter, resetParameters } = useParameterControl({
    initialParameters: initialParams,
    onParameterChange: (parameterId, value) => {
      console.log(`Parameter ${String(parameterId)} changed to:`, value);
    }
  });

  const { currentDataset, generateDataset, isLoading: datasetLoading } = useDataset({
    onDatasetChange: () => {
      // Clear training results when dataset changes
      stopTraining();
    }
  });

  const {
    model,
    trainingHistory,
    isTraining: modelIsTraining,
    currentEpoch,
    stopTraining
  } = useModelTraining();

  // Generate initial dataset on component mount
  useEffect(() => {
    if (!currentDataset) {
      generateDataset('classification', { 
        numPoints: 200, 
        noise: 0.1,
        pattern: 'circle'
      });
    }
  }, [currentDataset, generateDataset]);

  // Update training state
  useEffect(() => {
    setIsTraining(modelIsTraining);
  }, [modelIsTraining]);

  const handleTrainModel = useCallback(async () => {
    if (!currentDataset) return;
    
    try {
      // TODO: Convert dataset to TensorFlow tensors and call trainModel
      // For now, this is a placeholder
      console.log('Training would start with dataset:', currentDataset);
    } catch (error) {
      console.error('Training failed:', error);
    }
  }, [currentDataset, parameters]);

  const handleGenerateDataset = async (type: string, options?: any) => {
    stopTraining();
    await generateDataset(type, options);
  };

  const handleParameterChange = (parameterId: string, value: any) => {
    if (parameterId === 'generateCircle') {
      handleGenerateDataset('circle', { numPoints: 200, noise: 0.1 });
    } else if (parameterId === 'generateSpiral') {
      handleGenerateDataset('spiral', { numPoints: 200, noise: 0.05 });
    } else if (parameterId === 'generateXOR') {
      handleGenerateDataset('xor', { numPoints: 200, noise: 0.1 });
    } else {
      updateParameter(parameterId as keyof NeuralNetworkParams, value);
    }
  };

  const handleLayerAdd = (layerConfig: any) => {
    const layers = Array.isArray(parameters.layers) ? parameters.layers : [];
    const newLayers = [...layers, layerConfig];
    updateParameter('layers', newLayers);
  };

  const handleLayerRemove = (index: number) => {
    const layers = Array.isArray(parameters.layers) ? parameters.layers : [];
    if (layers.length > 1) { // Keep at least one layer
      const newLayers = layers.filter((_, i) => i !== index);
      updateParameter('layers', newLayers);
    }
  };

  const handleLayerUpdate = (index: number, layerConfig: any) => {
    const layers = Array.isArray(parameters.layers) ? parameters.layers : [];
    const newLayers = [...layers];
    newLayers[index] = layerConfig;
    updateParameter('layers', newLayers);
  };

  const getDataPoints = (): DataPoint[] => {
    return currentDataset?.points || [];
  };

  const getPredictions = (): DataPoint[] => {
    if (!model || !currentDataset) return [];
    
    try {
      // Simple placeholder predictions for visualization
      // In a real implementation, this would use the actual model
      const predictions = currentDataset.points.map(point => {
        // Simple decision boundary for demonstration
        const distance = Math.sqrt(point.x * point.x + point.y * point.y);
        const prediction = distance < 1 ? 1 : 0;
        return {
          id: `pred-${point.id}`,
          x: point.x,
          y: point.y,
          label: prediction
        };
      });
      return predictions;
    } catch (error) {
      console.error('Prediction failed:', error);
      return [];
    }
  };

  const controlPanelConfig = {
    title: 'Neural Network Parameters',
    controls: [
      {
        id: 'learningRate',
        label: 'Learning Rate',
        type: 'slider' as const,
        min: 0.001,
        max: 0.1,
        step: 0.001,
        value: parameters.learningRate
      },
      {
        id: 'batchSize',
        label: 'Batch Size',
        type: 'slider' as const,
        min: 8,
        max: 128,
        step: 8,
        value: parameters.batchSize
      },
      {
        id: 'epochs',
        label: 'Epochs',
        type: 'slider' as const,
        min: 10,
        max: 500,
        step: 10,
        value: parameters.epochs
      },
      {
        id: 'optimizer',
        label: 'Optimizer',
        type: 'select' as const,
        options: [
          { value: 'adam', label: 'Adam' },
          { value: 'sgd', label: 'SGD' },
          { value: 'rmsprop', label: 'RMSprop' }
        ],
        value: parameters.optimizer
      },
      {
        id: 'activation',
        label: 'Default Activation',
        type: 'select' as const,
        options: [
          { value: 'relu', label: 'ReLU' },
          { value: 'tanh', label: 'Tanh' },
          { value: 'sigmoid', label: 'Sigmoid' },
          { value: 'linear', label: 'Linear' }
        ],
        value: parameters.activation
      }
    ],
    groups: [
      {
        title: 'Training Parameters',
        controls: [
          {
            id: 'learningRate',
            label: 'Learning Rate',
            type: 'slider' as const,
            min: 0.001,
            max: 0.1,
            step: 0.001,
            value: parameters.learningRate
          },
          {
            id: 'batchSize',
            label: 'Batch Size',
            type: 'slider' as const,
            min: 8,
            max: 128,
            step: 8,
            value: parameters.batchSize
          },
          {
            id: 'epochs',
            label: 'Epochs',
            type: 'slider' as const,
            min: 10,
            max: 500,
            step: 10,
            value: parameters.epochs
          }
        ]
      },
      {
        title: 'Model Configuration',
        controls: [
          {
            id: 'optimizer',
            label: 'Optimizer',
            type: 'select' as const,
            options: [
              { value: 'adam', label: 'Adam' },
              { value: 'sgd', label: 'SGD' },
              { value: 'rmsprop', label: 'RMSprop' }
            ],
            value: parameters.optimizer
          },
          {
            id: 'activation',
            label: 'Default Activation',
            type: 'select' as const,
            options: [
              { value: 'relu', label: 'ReLU' },
              { value: 'tanh', label: 'Tanh' },
              { value: 'sigmoid', label: 'Sigmoid' },
              { value: 'linear', label: 'Linear' }
            ],
            value: parameters.activation
          }
        ]
      },
      {
        title: 'Dataset Options',
        controls: [
          {
            id: 'generateCircle',
            label: 'Generate Circle Data',
            type: 'button' as const
          },
          {
            id: 'generateSpiral',
            label: 'Generate Spiral Data',
            type: 'button' as const
          },
          {
            id: 'generateXOR',
            label: 'Generate XOR Data',
            type: 'button' as const
          }
        ]
      }
    ]
  };

  const currentMetrics = trainingHistory.length > 0 ? {
    loss: trainingHistory[trainingHistory.length - 1]?.loss || null,
    accuracy: trainingHistory[trainingHistory.length - 1]?.accuracy || null,
    mse: null,
    mae: null,
    r2: null,
    precision: null,
    recall: null,
    f1Score: null,
    lossHistory: trainingHistory.map(h => h.loss),
    validationLossHistory: trainingHistory.map(h => h.valLoss).filter(l => l !== undefined),
    trainingProgress: isTraining ? {
      currentEpoch: currentEpoch,
      totalEpochs: typeof parameters.epochs === 'number' ? parameters.epochs : 100,
      progress: typeof parameters.epochs === 'number' ? currentEpoch / parameters.epochs : 0
    } : undefined,
    additionalMetrics: undefined
  } : {
    loss: null,
    accuracy: null,
    mse: null,
    mae: null,
    r2: null,
    precision: null,
    recall: null,
    f1Score: null,
    lossHistory: [],
    validationLossHistory: undefined,
    trainingProgress: undefined,
    additionalMetrics: undefined
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Neural Network Playground
          </h1>
          <p className="text-gray-600">
            Build and train neural networks interactively. Experiment with different architectures and see how they learn.
          </p>
        </div>

        <div className="grid grid-cols-12 gap-6">
          {/* Control Panel */}
          <div className="col-span-12 lg:col-span-3">
            <ControlPanel
              config={controlPanelConfig}
              values={parameters}
              onChange={handleParameterChange}
              className="sticky top-4"
            />

            {/* Action Buttons */}
            <div className="mt-6 space-y-3">
              <button
                onClick={handleTrainModel}
                disabled={isTraining || datasetLoading || !currentDataset}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isTraining 
                  ? `Training... (${currentEpoch}/${typeof parameters.epochs === 'number' ? parameters.epochs : 100})` 
                  : 'Train Network'
                }
              </button>

              <button
                onClick={stopTraining}
                disabled={!model}
                className="w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Reset Model
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
                  checked={showNetworkDiagram}
                  onChange={(e) => setShowNetworkDiagram(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm text-gray-600">Show Network Diagram</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showTrainingViz}
                  onChange={(e) => setShowTrainingViz(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm text-gray-600">Show Training Visualization</span>
              </label>
            </div>
          </div>

          {/* Main Content */}
          <div className={`col-span-12 ${infoSidebarOpen ? 'lg:col-span-6' : 'lg:col-span-9'}`}>
            <div className="space-y-6">
              {/* Metrics Dashboard */}
              <MetricsDashboard
                metrics={currentMetrics}
                isTraining={isTraining}
              />

              {/* Layer Controls */}
              <LayerControls
                layers={parameters.layers}
                selectedLayer={selectedLayer}
                onLayerSelect={setSelectedLayer}
                onLayerAdd={handleLayerAdd}
                onLayerRemove={handleLayerRemove}
                onLayerUpdate={handleLayerUpdate}
                isTraining={isTraining}
              />

              {/* Network Diagram */}
              {showNetworkDiagram && (
                <NetworkDiagram
                  layers={parameters.layers}
                  data={getDataPoints()}
                  predictions={getPredictions()}
                  selectedLayer={selectedLayer}
                  onLayerSelect={setSelectedLayer}
                  isTraining={isTraining}
                />
              )}

              {/* Training Visualization */}
              {showTrainingViz && (
                <TrainingVisualization
                  data={getDataPoints()}
                  predictions={getPredictions()}
                  trainingHistory={trainingHistory}
                  isTraining={isTraining}
                  currentEpoch={currentEpoch}
                />
              )}
            </div>
          </div>

          {/* Info Sidebar */}
          <div className={`col-span-12 ${infoSidebarOpen ? 'lg:col-span-3' : 'lg:col-span-0'}`}>
            <InfoSidebar
              sections={commonInfoSections.neuralNetwork}
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
