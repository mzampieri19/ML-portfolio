'use client';

import { useState, useCallback, useEffect } from 'react';
import { ModelParameters } from '@/lib/visualize/types';

interface UseParameterControlOptions<T extends ModelParameters> {
  initialParameters: T;
  onParameterChange?: (parameterId: keyof T, value: any) => void;
  onParametersReset?: (parameters: T) => void;
  debounceMs?: number;
}

interface ParameterControl<T extends ModelParameters> {
  parameters: T;
  updateParameter: (parameterId: keyof T, value: any) => void;
  updateParameters: (newParameters: Partial<T>) => void;
  resetParameters: () => void;
  isModified: boolean;
  hasChanges: boolean;
}

export function useParameterControl<T extends ModelParameters>(
  options: UseParameterControlOptions<T>
): ParameterControl<T> {
  const {
    initialParameters,
    onParameterChange,
    onParametersReset,
    debounceMs = 300
  } = options;

  const [parameters, setParameters] = useState<T>(initialParameters);
  const [isModified, setIsModified] = useState(false);
  const [debounceTimer, setDebounceTimer] = useState<NodeJS.Timeout | null>(null);

  // Check if current parameters differ from initial
  const hasChanges = JSON.stringify(parameters) !== JSON.stringify(initialParameters);

  const updateParameter = useCallback((parameterId: keyof T, value: any) => {
    setParameters(prev => {
      const newParameters = { ...prev, [parameterId]: value };
      
      // Clear existing timer
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }

      // Set new debounced timer
      const timer = setTimeout(() => {
        onParameterChange?.(parameterId, value);
      }, debounceMs);
      
      setDebounceTimer(timer);
      setIsModified(true);
      
      return newParameters;
    });
  }, [onParameterChange, debounceMs, debounceTimer]);

  const updateParameters = useCallback((newParameters: Partial<T>) => {
    setParameters(prev => {
      const updated = { ...prev, ...newParameters };
      setIsModified(true);
      return updated;
    });
  }, []);

  const resetParameters = useCallback(() => {
    setParameters(initialParameters);
    setIsModified(false);
    onParametersReset?.(initialParameters);
    
    // Clear any pending debounced calls
    if (debounceTimer) {
      clearTimeout(debounceTimer);
      setDebounceTimer(null);
    }
  }, [initialParameters, onParametersReset, debounceTimer]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }
    };
  }, [debounceTimer]);

  return {
    parameters,
    updateParameter,
    updateParameters,
    resetParameters,
    isModified,
    hasChanges
  };
}

export default useParameterControl;
