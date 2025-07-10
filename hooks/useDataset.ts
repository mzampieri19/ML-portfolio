'use client';

import { useState, useCallback, useEffect } from 'react';
import { DataPoint, Dataset } from '@/lib/visualize/types';
import { 
  generateLinearDataset, 
  generateSpiralDataset, 
  generateCircleDataset,
  generatePolynomialDataset,
  generateSinusoidalDataset,
  generateBlobsDataset,
  generateXORDataset
} from '@/lib/visualize/datasets';

interface UseDatasetOptions {
  initialDataset?: Dataset;
  onDatasetChange?: (dataset: Dataset) => void;
}

interface DatasetControl {
  currentDataset: Dataset | null;
  isLoading: boolean;
  error: string | null;
  generateDataset: (type: string, options?: any) => Promise<void>;
  setCustomDataset: (dataset: Dataset) => void;
  addDataPoint: (point: DataPoint) => void;
  removeDataPoint: (index: number) => void;
  clearDataset: () => void;
  exportDataset: () => string;
  importDataset: (jsonString: string) => boolean;
}

const datasetGenerators = {
  linear: generateLinearDataset,
  spiral: generateSpiralDataset,
  circle: generateCircleDataset,
  polynomial: generatePolynomialDataset,
  sine: generateSinusoidalDataset,
  blobs: generateBlobsDataset,
  xor: generateXORDataset
};

export function useDataset(options: UseDatasetOptions = {}): DatasetControl {
  const { initialDataset, onDatasetChange } = options;
  
  const [currentDataset, setCurrentDataset] = useState<Dataset | null>(initialDataset || null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateDataset = useCallback(async (type: string, generatorOptions: any = {}) => {
    setIsLoading(true);
    setError(null);

    try {
      const generator = datasetGenerators[type as keyof typeof datasetGenerators];
      if (!generator) {
        throw new Error(`Unknown dataset type: ${type}`);
      }

      // Default options
      const defaultOptions = {
        numPoints: 100,
        noise: 0.1,
        seed: Math.random()
      };

      const finalOptions = { ...defaultOptions, ...generatorOptions };
      const dataset = generator(finalOptions);

      setCurrentDataset(dataset);
      onDatasetChange?.(dataset);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate dataset';
      setError(errorMessage);
      console.error('Dataset generation error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [onDatasetChange]);

  const setCustomDataset = useCallback((dataset: Dataset) => {
    setCurrentDataset(dataset);
    setError(null);
    onDatasetChange?.(dataset);
  }, [onDatasetChange]);

  const addDataPoint = useCallback((point: DataPoint) => {
    setCurrentDataset(prev => {
      if (!prev) return null;
      
      const newDataset = {
        ...prev,
        points: [...prev.points, point]
      };
      
      onDatasetChange?.(newDataset);
      return newDataset;
    });
  }, [onDatasetChange]);

  const removeDataPoint = useCallback((index: number) => {
    setCurrentDataset(prev => {
      if (!prev || index < 0 || index >= prev.points.length) return prev;
      
      const newDataset = {
        ...prev,
        points: prev.points.filter((_: DataPoint, i: number) => i !== index)
      };
      
      onDatasetChange?.(newDataset);
      return newDataset;
    });
  }, [onDatasetChange]);

  const clearDataset = useCallback(() => {
    setCurrentDataset(null);
    setError(null);
    onDatasetChange?.(null as any);
  }, [onDatasetChange]);

  const exportDataset = useCallback((): string => {
    if (!currentDataset) {
      throw new Error('No dataset to export');
    }
    return JSON.stringify(currentDataset, null, 2);
  }, [currentDataset]);

  const importDataset = useCallback((jsonString: string): boolean => {
    try {
      const dataset = JSON.parse(jsonString) as Dataset;
      
      // Basic validation
      if (!dataset.points || !Array.isArray(dataset.points)) {
        throw new Error('Invalid dataset format: missing or invalid points array');
      }

      // Validate data points
      for (const point of dataset.points) {
        if (typeof point.x !== 'number' || typeof point.y !== 'number') {
          throw new Error('Invalid dataset format: data points must have numeric x and y values');
        }
      }

      setCustomDataset(dataset);
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to import dataset';
      setError(errorMessage);
      return false;
    }
  }, [setCustomDataset]);

  // Clear error when dataset changes
  useEffect(() => {
    if (currentDataset && error) {
      setError(null);
    }
  }, [currentDataset, error]);

  return {
    currentDataset,
    isLoading,
    error,
    generateDataset,
    setCustomDataset,
    addDataPoint,
    removeDataPoint,
    clearDataset,
    exportDataset,
    importDataset
  };
}

export default useDataset;
