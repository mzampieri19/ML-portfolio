// Utility functions for the Interactive Model Visualization Section

import { DataPoint, Point2D, Dataset, VisualizationConfig } from './types';

// ============================================================================
// MATHEMATICAL UTILITIES
// ============================================================================

/**
 * Generate a range of numbers from start to end with specified step
 */
export function range(start: number, end: number, step: number = 1): number[] {
  const result: number[] = [];
  for (let i = start; i < end; i += step) {
    result.push(i);
  }
  return result;
}

/**
 * Linear interpolation between two values
 */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Map a value from one range to another
 */
export function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

/**
 * Calculate Euclidean distance between two points
 */
export function distance(p1: Point2D, p2: Point2D): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate mean of an array of numbers
 */
export function mean(values: number[]): number {
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

/**
 * Calculate standard deviation of an array of numbers
 */
export function standardDeviation(values: number[]): number {
  const avg = mean(values);
  const squareDiffs = values.map(value => Math.pow(value - avg, 2));
  return Math.sqrt(mean(squareDiffs));
}

/**
 * Calculate R-squared (coefficient of determination)
 */
export function rSquared(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  
  const actualMean = mean(actual);
  const totalSumSquares = actual.reduce((sum, val) => sum + Math.pow(val - actualMean, 2), 0);
  const residualSumSquares = actual.reduce((sum, val, i) => sum + Math.pow(val - predicted[i], 2), 0);
  
  return 1 - (residualSumSquares / totalSumSquares);
}

/**
 * Calculate Mean Squared Error
 */
export function meanSquaredError(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  
  const squaredErrors = actual.map((val, i) => Math.pow(val - predicted[i], 2));
  return mean(squaredErrors);
}

/**
 * Calculate Mean Absolute Error
 */
export function meanAbsoluteError(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  
  const absoluteErrors = actual.map((val, i) => Math.abs(val - predicted[i]));
  return mean(absoluteErrors);
}

// ============================================================================
// DATA MANIPULATION UTILITIES
// ============================================================================

/**
 * Normalize data to [0, 1] range
 */
export function normalize(data: number[]): { normalized: number[], min: number, max: number } {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;
  
  if (range === 0) {
    return { normalized: data.map(() => 0), min, max };
  }
  
  const normalized = data.map(val => (val - min) / range);
  return { normalized, min, max };
}

/**
 * Standardize data to have mean 0 and standard deviation 1
 */
export function standardize(data: number[]): { standardized: number[], mean: number, std: number } {
  const dataMean = mean(data);
  const dataStd = standardDeviation(data);
  
  if (dataStd === 0) {
    return { standardized: data.map(() => 0), mean: dataMean, std: dataStd };
  }
  
  const standardized = data.map(val => (val - dataMean) / dataStd);
  return { standardized, mean: dataMean, std: dataStd };
}

/**
 * Shuffle an array using Fisher-Yates algorithm
 */
export function shuffle<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Split dataset into training and testing sets
 */
export function trainTestSplit<T>(
  data: T[],
  testSize: number = 0.2,
  randomSeed?: number
): { train: T[], test: T[] } {
  if (randomSeed !== undefined) {
    // Simple seedable random function
    let seed = randomSeed;
    Math.random = () => {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    };
  }
  
  const shuffledData = shuffle(data);
  const splitIndex = Math.floor(data.length * (1 - testSize));
  
  return {
    train: shuffledData.slice(0, splitIndex),
    test: shuffledData.slice(splitIndex)
  };
}

// ============================================================================
// VISUALIZATION UTILITIES
// ============================================================================

/**
 * Generate a color palette for visualizations
 */
export function generateColorPalette(numColors: number): string[] {
  const colors: string[] = [];
  for (let i = 0; i < numColors; i++) {
    const hue = (i * 360) / numColors;
    colors.push(`hsl(${hue}, 70%, 50%)`);
  }
  return colors;
}

/**
 * Convert RGB to hex color
 */
export function rgbToHex(r: number, g: number, b: number): string {
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/**
 * Convert hex to RGB color
 */
export function hexToRgb(hex: string): { r: number, g: number, b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

/**
 * Get bounds of a dataset
 */
export function getDataBounds(points: DataPoint[]): {
  minX: number,
  maxX: number,
  minY: number,
  maxY: number
} {
  if (points.length === 0) {
    return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  }
  
  const xValues = points.map(p => p.x);
  const yValues = points.map(p => p.y);
  
  return {
    minX: Math.min(...xValues),
    maxX: Math.max(...xValues),
    minY: Math.min(...yValues),
    maxY: Math.max(...yValues)
  };
}

/**
 * Create a grid of points for contour plotting
 */
export function createGrid(
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  resolution: number = 50
): Point2D[] {
  const grid: Point2D[] = [];
  const xStep = (xMax - xMin) / resolution;
  const yStep = (yMax - yMin) / resolution;
  
  for (let i = 0; i <= resolution; i++) {
    for (let j = 0; j <= resolution; j++) {
      grid.push({
        x: xMin + i * xStep,
        y: yMin + j * yStep
      });
    }
  }
  
  return grid;
}

// ============================================================================
// ANIMATION UTILITIES
// ============================================================================

/**
 * Easing functions for smooth animations
 */
export const easing = {
  linear: (t: number): number => t,
  easeIn: (t: number): number => t * t,
  easeOut: (t: number): number => t * (2 - t),
  easeInOut: (t: number): number => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeInCubic: (t: number): number => t * t * t,
  easeOutCubic: (t: number): number => (--t) * t * t + 1,
  easeInOutCubic: (t: number): number => 
    t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1
};

/**
 * Animate a value from start to end over duration
 */
export function animateValue(
  start: number,
  end: number,
  duration: number,
  easingFn: (t: number) => number = easing.easeInOut,
  onUpdate: (value: number) => void,
  onComplete?: () => void
): () => void {
  const startTime = performance.now();
  
  function update() {
    const elapsed = performance.now() - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easedProgress = easingFn(progress);
    const currentValue = lerp(start, end, easedProgress);
    
    onUpdate(currentValue);
    
    if (progress < 1) {
      requestAnimationFrame(update);
    } else if (onComplete) {
      onComplete();
    }
  }
  
  const animationId = requestAnimationFrame(update);
  
  // Return cancel function
  return () => cancelAnimationFrame(animationId);
}

// ============================================================================
// FORMATTING UTILITIES
// ============================================================================

/**
 * Format a number for display
 */
export function formatNumber(
  value: number,
  decimals: number = 2,
  notation: 'standard' | 'scientific' | 'engineering' = 'standard'
): string {
  if (notation === 'scientific') {
    return value.toExponential(decimals);
  } else if (notation === 'engineering') {
    const exponent = Math.floor(Math.log10(Math.abs(value)) / 3) * 3;
    const mantissa = value / Math.pow(10, exponent);
    return `${mantissa.toFixed(decimals)}e${exponent}`;
  } else {
    return value.toFixed(decimals);
  }
}

/**
 * Format time duration for display
 */
export function formatDuration(milliseconds: number): string {
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

/**
 * Validate dataset structure
 */
export function validateDataset(dataset: Dataset): string[] {
  const errors: string[] = [];
  
  if (!dataset.id || dataset.id.trim() === '') {
    errors.push('Dataset must have a valid ID');
  }
  
  if (!dataset.name || dataset.name.trim() === '') {
    errors.push('Dataset must have a name');
  }
  
  if (!dataset.points || dataset.points.length === 0) {
    errors.push('Dataset must contain at least one data point');
  }
  
  if (dataset.points) {
    dataset.points.forEach((point, index) => {
      if (typeof point.x !== 'number' || typeof point.y !== 'number') {
        errors.push(`Data point at index ${index} has invalid coordinates`);
      }
      if (!point.id || point.id.trim() === '') {
        errors.push(`Data point at index ${index} must have a valid ID`);
      }
    });
  }
  
  return errors;
}

/**
 * Validate parameter ranges
 */
export function validateParameter(
  value: number,
  min: number,
  max: number,
  name: string
): string | null {
  if (typeof value !== 'number' || isNaN(value)) {
    return `${name} must be a valid number`;
  }
  
  if (value < min || value > max) {
    return `${name} must be between ${min} and ${max}`;
  }
  
  return null;
}

// ============================================================================
// PERFORMANCE UTILITIES
// ============================================================================

/**
 * Debounce function to limit function calls
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): T {
  let timeoutId: NodeJS.Timeout;
  
  return ((...args: any[]) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  }) as T;
}

/**
 * Throttle function to limit function call frequency
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): T {
  let inThrottle: boolean;
  
  return ((...args: any[]) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }) as T;
}

/**
 * Simple performance measurement
 */
export function measurePerformance<T>(
  name: string,
  func: () => T
): { result: T, duration: number } {
  const start = performance.now();
  const result = func();
  const duration = performance.now() - start;
  
  console.log(`${name}: ${duration.toFixed(2)}ms`);
  
  return { result, duration };
}

// ============================================================================
// EXPORT UTILITIES
// ============================================================================

/**
 * Download data as JSON file
 */
export function downloadJSON(data: any, filename: string): void {
  const jsonString = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  URL.revokeObjectURL(url);
}

/**
 * Download canvas as image
 */
export function downloadCanvasAsImage(
  canvas: HTMLCanvasElement,
  filename: string,
  format: 'png' | 'jpeg' = 'png'
): void {
  const link = document.createElement('a');
  link.download = filename;
  link.href = canvas.toDataURL(`image/${format}`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
