// Custom React hook for managing visualization state

import { useState, useCallback, useRef, useEffect } from 'react';
import { 
  VisualizationState, 
  ModelParameters, 
  AlgorithmResult, 
  Dataset,
  ModelType 
} from '../lib/visualize/types';

interface UseVisualizationOptions {
  initialDataset?: Dataset;
  initialParameters?: ModelParameters;
  modelType: ModelType;
  onParameterChange?: (parameterId: string, value: any) => void;
  onTrainingComplete?: (result: AlgorithmResult) => void;
  onError?: (error: string) => void;
}

export function useVisualization(options: UseVisualizationOptions) {
  const {
    initialDataset,
    initialParameters = {},
    modelType,
    onParameterChange,
    onTrainingComplete,
    onError
  } = options;

  // Core state
  const [state, setState] = useState<VisualizationState>({
    isTraining: false,
    isPaused: false,
    currentEpoch: 0,
    totalEpochs: 0,
    selectedDataset: initialDataset?.id || '',
    parameters: initialParameters,
    results: undefined,
    error: undefined
  });

  // Datasets state
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [currentDataset, setCurrentDataset] = useState<Dataset | null>(initialDataset || null);

  // Training control refs
  const trainingRef = useRef<{
    shouldStop: boolean;
    cancelCallback?: () => void;
  }>({ shouldStop: false });

  // Update parameter
  const updateParameter = useCallback((parameterId: string, value: any) => {
    setState(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [parameterId]: value
      }
    }));
    
    if (onParameterChange) {
      onParameterChange(parameterId, value);
    }
  }, [onParameterChange]);

  // Update multiple parameters at once
  const updateParameters = useCallback((newParameters: Partial<ModelParameters>) => {
    setState(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        ...Object.fromEntries(
          Object.entries(newParameters).filter(([_, v]) => v !== undefined)
        ) as ModelParameters
      }
    }));
  }, []);

  // Reset parameters to defaults
  const resetParameters = useCallback(() => {
    setState(prev => ({
      ...prev,
      parameters: initialParameters
    }));
  }, [initialParameters]);

  // Set dataset
  const setDataset = useCallback((dataset: Dataset) => {
    setCurrentDataset(dataset);
    setState(prev => ({
      ...prev,
      selectedDataset: dataset.id,
      results: undefined, // Clear previous results
      error: undefined
    }));
  }, []);

  // Start training/computation
  const startTraining = useCallback(async (
    algorithm: (dataset: Dataset, params: ModelParameters, callbacks?: any) => Promise<AlgorithmResult>
  ) => {
    if (!currentDataset) {
      const error = 'No dataset selected';
      setState(prev => ({ ...prev, error }));
      if (onError) onError(error);
      return;
    }

    setState(prev => ({
      ...prev,
      isTraining: true,
      isPaused: false,
      error: undefined,
      currentEpoch: 0
    }));

    trainingRef.current.shouldStop = false;

    try {
      const callbacks = {
        onEpochComplete: (epoch: number, loss: number, weights?: number[]) => {
          if (trainingRef.current.shouldStop) return;
          
          setState(prev => ({
            ...prev,
            currentEpoch: epoch
          }));
        },
        onIterationComplete: (iteration: number, data: any) => {
          if (trainingRef.current.shouldStop) return;
          
          setState(prev => ({
            ...prev,
            currentEpoch: iteration
          }));
        }
      };

      const result = await algorithm(currentDataset, state.parameters, callbacks);
      
      if (!trainingRef.current.shouldStop) {
        setState(prev => ({
          ...prev,
          isTraining: false,
          results: result
        }));
        
        if (onTrainingComplete) {
          onTrainingComplete(result);
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Training failed';
      setState(prev => ({
        ...prev,
        isTraining: false,
        error: errorMessage
      }));
      
      if (onError) {
        onError(errorMessage);
      }
    }
  }, [currentDataset, state.parameters, onTrainingComplete, onError]);

  // Stop training
  const stopTraining = useCallback(() => {
    trainingRef.current.shouldStop = true;
    if (trainingRef.current.cancelCallback) {
      trainingRef.current.cancelCallback();
    }
    
    setState(prev => ({
      ...prev,
      isTraining: false,
      isPaused: false
    }));
  }, []);

  // Pause/resume training
  const pauseTraining = useCallback(() => {
    setState(prev => ({
      ...prev,
      isPaused: !prev.isPaused
    }));
  }, []);

  // Clear results
  const clearResults = useCallback(() => {
    setState(prev => ({
      ...prev,
      results: undefined,
      error: undefined,
      currentEpoch: 0
    }));
  }, []);

  // Set error
  const setError = useCallback((error: string | undefined) => {
    setState(prev => ({
      ...prev,
      error
    }));
  }, []);

  // Set total epochs for progress tracking
  const setTotalEpochs = useCallback((total: number) => {
    setState(prev => ({
      ...prev,
      totalEpochs: total
    }));
  }, []);

  // Calculate training progress
  const progress = state.totalEpochs > 0 
    ? Math.min((state.currentEpoch / state.totalEpochs) * 100, 100) 
    : 0;

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (state.isTraining) {
        stopTraining();
      }
    };
  }, [state.isTraining, stopTraining]);

  return {
    // State
    state,
    currentDataset,
    datasets,
    progress,
    
    // Actions
    updateParameter,
    updateParameters,
    resetParameters,
    setDataset,
    setDatasets,
    startTraining,
    stopTraining,
    pauseTraining,
    clearResults,
    setError,
    setTotalEpochs,
    
    // Computed values
    isTraining: state.isTraining,
    isPaused: state.isPaused,
    hasResults: !!state.results,
    hasError: !!state.error,
    canTrain: !!currentDataset && !state.isTraining,
    
    // Results
    results: state.results,
    error: state.error,
    parameters: state.parameters
  };
}
