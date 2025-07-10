// Custom React hook for neural network training state management

import { useState, useCallback, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { 
  NeuralNetworkParams, 
  LayerConfig, 
  NetworkArchitecture, 
  TrainingMetrics, 
  TrainingConfig,
  ModelCompileConfig 
} from '../lib/visualize/types';

interface UseModelTrainingOptions {
  initialArchitecture?: NetworkArchitecture;
  initialParams?: NeuralNetworkParams;
  onEpochComplete?: (metrics: TrainingMetrics) => void;
  onTrainingComplete?: (model: tf.LayersModel, history: TrainingMetrics[]) => void;
  onError?: (error: string) => void;
}

export function useModelTraining(options: UseModelTrainingOptions = {}) {
  const {
    initialArchitecture,
    initialParams,
    onEpochComplete,
    onTrainingComplete,
    onError
  } = options;

  // Model and training state
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [architecture, setArchitecture] = useState<NetworkArchitecture>(
    initialArchitecture || { layers: [], connections: [] }
  );
  const [parameters, setParameters] = useState<NeuralNetworkParams>(
    initialParams || {
      layers: [],
      learningRate: 0.01,
      batchSize: 32,
      epochs: 100,
      optimizer: 'adam',
      activation: 'relu'
    }
  );

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [trainingHistory, setTrainingHistory] = useState<TrainingMetrics[]>([]);
  const [error, setError] = useState<string | undefined>();

  // Training control
  const trainingRef = useRef<{
    shouldStop: boolean;
    model?: tf.LayersModel;
  }>({ shouldStop: false });

  // Build TensorFlow model from architecture
  const buildModel = useCallback(async (): Promise<tf.LayersModel | null> => {
    try {
      if (architecture.layers.length === 0) {
        throw new Error('No layers defined in architecture');
      }

      const model = tf.sequential();
      
      // Add layers to the model
      architecture.layers.forEach((layerConfig, index) => {
        let layer: tf.layers.Layer;

        switch (layerConfig.type) {
          case 'dense':
            layer = tf.layers.dense({
              units: layerConfig.units || 1,
              activation: (layerConfig.activation || parameters.activation) as any,
              inputShape: index === 0 ? layerConfig.inputShape : undefined
            });
            break;

          case 'conv2d':
            layer = tf.layers.conv2d({
              filters: layerConfig.filters || 32,
              kernelSize: layerConfig.kernelSize || 3,
              activation: (layerConfig.activation || parameters.activation) as any,
              inputShape: index === 0 ? layerConfig.inputShape : undefined
            });
            break;

          case 'maxPooling2d':
            layer = tf.layers.maxPooling2d({
              poolSize: layerConfig.poolSize || 2
            });
            break;

          case 'dropout':
            layer = tf.layers.dropout({
              rate: layerConfig.rate || 0.2
            });
            break;

          case 'batchNormalization':
            layer = tf.layers.batchNormalization();
            break;

          default:
            throw new Error(`Unsupported layer type: ${layerConfig.type}`);
        }

        model.add(layer);
      });

      setModel(model);
      setError(undefined);
      return model;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to build model';
      setError(errorMessage);
      if (onError) onError(errorMessage);
      return null;
    }
  }, [architecture, parameters.activation, onError]);

  // Compile model
  const compileModel = useCallback(async (
    model: tf.LayersModel,
    config?: Partial<ModelCompileConfig>
  ): Promise<boolean> => {
    try {
      const compileConfig: ModelCompileConfig = {
        optimizer: config?.optimizer || parameters.optimizer,
        loss: config?.loss || 'meanSquaredError',
        metrics: config?.metrics || ['mae']
      };

      model.compile({
        optimizer: compileConfig.optimizer,
        loss: compileConfig.loss,
        metrics: compileConfig.metrics
      });

      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to compile model';
      setError(errorMessage);
      if (onError) onError(errorMessage);
      return false;
    }
  }, [parameters.optimizer, onError]);

  // Train model
  const trainModel = useCallback(async (
    trainX: tf.Tensor,
    trainY: tf.Tensor,
    config?: Partial<TrainingConfig>
  ): Promise<boolean> => {
    if (!model) {
      const errorMessage = 'No model available for training';
      setError(errorMessage);
      if (onError) onError(errorMessage);
      return false;
    }

    const trainingConfig: TrainingConfig = {
      epochs: config?.epochs || parameters.epochs,
      batchSize: config?.batchSize || parameters.batchSize,
      validationSplit: config?.validationSplit || 0.2,
      shuffle: config?.shuffle ?? true,
      callbacks: config?.callbacks || []
    };

    setIsTraining(true);
    setIsPaused(false);
    setCurrentEpoch(0);
    setTrainingHistory([]);
    trainingRef.current.shouldStop = false;
    trainingRef.current.model = model;

    try {
      // Custom callback to track training progress
      const epochCallback = new tf.CustomCallback({
        onEpochEnd: async (epoch: number, logs: any) => {
          if (trainingRef.current.shouldStop) {
            model.stopTraining = true;
            return;
          }

          const metrics: TrainingMetrics = {
            epoch: epoch + 1,
            loss: logs?.loss || 0,
            accuracy: logs?.acc || logs?.accuracy,
            valLoss: logs?.val_loss,
            valAccuracy: logs?.val_acc || logs?.val_accuracy,
            timestamp: Date.now()
          };

          setCurrentEpoch(epoch + 1);
          setTrainingHistory(prev => [...prev, metrics]);

          if (onEpochComplete) {
            onEpochComplete(metrics);
          }
        }
      });

      const callbacks = [epochCallback, ...(trainingConfig.callbacks || [])];

      const history = await model.fit(trainX, trainY, {
        epochs: trainingConfig.epochs,
        batchSize: trainingConfig.batchSize,
        validationSplit: trainingConfig.validationSplit,
        shuffle: trainingConfig.shuffle,
        callbacks
      });

      if (!trainingRef.current.shouldStop) {
        setIsTraining(false);
        if (onTrainingComplete) {
          onTrainingComplete(model, trainingHistory);
        }
      }

      return true;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Training failed';
      setError(errorMessage);
      setIsTraining(false);
      if (onError) onError(errorMessage);
      return false;
    }
  }, [model, parameters, onEpochComplete, onTrainingComplete, onError, trainingHistory]);

  // Stop training
  const stopTraining = useCallback(() => {
    trainingRef.current.shouldStop = true;
    if (trainingRef.current.model) {
      trainingRef.current.model.stopTraining = true;
    }
    setIsTraining(false);
    setIsPaused(false);
  }, []);

  // Pause/resume training (Note: TensorFlow.js doesn't support pausing, so this is a placeholder)
  const pauseTraining = useCallback(() => {
    setIsPaused(!isPaused);
    // In a real implementation, you would need to implement pause/resume logic
    // by breaking training into smaller chunks
  }, [isPaused]);

  // Add layer to architecture
  const addLayer = useCallback((layer: LayerConfig) => {
    setArchitecture(prev => ({
      ...prev,
      layers: [...prev.layers, layer]
    }));
  }, []);

  // Remove layer from architecture
  const removeLayer = useCallback((layerId: string) => {
    setArchitecture(prev => ({
      ...prev,
      layers: prev.layers.filter(layer => layer.id !== layerId),
      connections: prev.connections.filter(
        conn => conn.from !== layerId && conn.to !== layerId
      )
    }));
  }, []);

  // Update layer in architecture
  const updateLayer = useCallback((layerId: string, updates: Partial<LayerConfig>) => {
    setArchitecture(prev => ({
      ...prev,
      layers: prev.layers.map(layer =>
        layer.id === layerId ? { ...layer, ...updates } : layer
      )
    }));
  }, []);

  // Clear architecture
  const clearArchitecture = useCallback(() => {
    setArchitecture({ layers: [], connections: [] });
    setModel(null);
  }, []);

  // Update training parameters
  const updateParameters = useCallback((updates: Partial<NeuralNetworkParams>) => {
    setParameters(prev => ({ ...prev, ...updates }));
  }, []);

  // Get model summary
  const getModelSummary = useCallback((): string => {
    if (!model) return 'No model built';
    
    try {
      // Create a simple summary
      const layers = architecture.layers;
      const totalParams = layers.reduce((sum, layer) => {
        if (layer.type === 'dense' && layer.units) {
          return sum + layer.units;
        }
        return sum;
      }, 0);

      return `Model with ${layers.length} layers and ~${totalParams} parameters`;
    } catch {
      return 'Unable to generate summary';
    }
  }, [model, architecture]);

  // Predict with model
  const predict = useCallback((input: tf.Tensor): tf.Tensor | null => {
    if (!model) {
      setError('No model available for prediction');
      return null;
    }

    try {
      return model.predict(input) as tf.Tensor;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Prediction failed';
      setError(errorMessage);
      if (onError) onError(errorMessage);
      return null;
    }
  }, [model, onError]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (model) {
        model.dispose();
      }
      if (isTraining) {
        stopTraining();
      }
    };
  }, [model, isTraining, stopTraining]);

  // Auto-rebuild model when architecture changes
  useEffect(() => {
    if (architecture.layers.length > 0) {
      buildModel();
    }
  }, [architecture, buildModel]);

  return {
    // Model state
    model,
    architecture,
    parameters,
    
    // Training state
    isTraining,
    isPaused,
    currentEpoch,
    trainingHistory,
    error,
    
    // Model actions
    buildModel,
    compileModel,
    trainModel,
    predict,
    
    // Training control
    stopTraining,
    pauseTraining,
    
    // Architecture management
    addLayer,
    removeLayer,
    updateLayer,
    clearArchitecture,
    
    // Parameter management
    updateParameters,
    
    // Utilities
    getModelSummary,
    
    // Computed values
    hasModel: !!model,
    canTrain: !!model && !isTraining,
    layerCount: architecture.layers.length,
    progress: parameters.epochs > 0 ? (currentEpoch / parameters.epochs) * 100 : 0
  };
}
