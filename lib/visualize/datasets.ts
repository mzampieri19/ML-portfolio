// Dataset generators for the Interactive Model Visualization Section

import { 
  Dataset, 
  DataPoint, 
  SpiralDatasetConfig, 
  CircleDatasetConfig, 
  RegressionDatasetConfig,
  DatasetGeneratorConfig 
} from './types';
import { generateColorPalette } from './utils';

// ============================================================================
// SEEDED RANDOM NUMBER GENERATOR
// ============================================================================

class SeededRandom {
  private seed: number;

  constructor(seed: number = Date.now()) {
    this.seed = seed;
  }

  next(): number {
    this.seed = (this.seed * 9301 + 49297) % 233280;
    return this.seed / 233280;
  }

  normal(mean: number = 0, stdDev: number = 1): number {
    // Box-Muller transformation
    const u1 = this.next();
    const u2 = this.next();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0 * stdDev + mean;
  }
}

// ============================================================================
// CLASSIFICATION DATASETS
// ============================================================================

/**
 * Generate spiral dataset for classification
 */
export function generateSpiralDataset(config: SpiralDatasetConfig): Dataset {
  const { numSamples, numClasses, spiralTightness, noise, randomSeed } = config;
  const random = new SeededRandom(randomSeed);
  const points: DataPoint[] = [];
  const colors = generateColorPalette(numClasses);
  
  const samplesPerClass = Math.floor(numSamples / numClasses);
  
  for (let classIndex = 0; classIndex < numClasses; classIndex++) {
    for (let i = 0; i < samplesPerClass; i++) {
      const t = (i / samplesPerClass) * spiralTightness;
      const angle = (classIndex * 2 * Math.PI) / numClasses + t;
      
      const radius = t;
      const x = radius * Math.cos(angle) + random.normal(0, noise);
      const y = radius * Math.sin(angle) + random.normal(0, noise);
      
      points.push({
        id: `spiral_${classIndex}_${i}`,
        x,
        y,
        label: classIndex,
        color: colors[classIndex]
      });
    }
  }
  
  return {
    id: 'spiral',
    name: 'Spiral Dataset',
    description: `${numClasses}-class spiral classification dataset with ${numSamples} samples`,
    points,
    features: ['x', 'y'],
    target: 'class',
    type: 'classification'
  };
}

/**
 * Generate concentric circles dataset for classification
 */
export function generateCircleDataset(config: CircleDatasetConfig): Dataset {
  const { numSamples, radius, innerRadius = 0, noise, randomSeed } = config;
  const random = new SeededRandom(randomSeed);
  const points: DataPoint[] = [];
  const colors = ['#ff6b6b', '#4ecdc4'];
  
  const outerSamples = Math.floor(numSamples / 2);
  const innerSamples = numSamples - outerSamples;
  
  // Outer circle (class 1)
  for (let i = 0; i < outerSamples; i++) {
    const angle = random.next() * 2 * Math.PI;
    const r = radius + random.normal(0, noise);
    const x = r * Math.cos(angle);
    const y = r * Math.sin(angle);
    
    points.push({
      id: `circle_outer_${i}`,
      x,
      y,
      label: 1,
      color: colors[1]
    });
  }
  
  // Inner circle (class 0)
  for (let i = 0; i < innerSamples; i++) {
    const angle = random.next() * 2 * Math.PI;
    const r = innerRadius + random.next() * (radius - innerRadius - 0.5) + random.normal(0, noise);
    const x = r * Math.cos(angle);
    const y = r * Math.sin(angle);
    
    points.push({
      id: `circle_inner_${i}`,
      x,
      y,
      label: 0,
      color: colors[0]
    });
  }
  
  return {
    id: 'circles',
    name: 'Concentric Circles',
    description: 'Two-class concentric circles classification dataset',
    points,
    features: ['x', 'y'],
    target: 'class',
    type: 'classification'
  };
}

/**
 * Generate XOR dataset for classification
 */
export function generateXORDataset(config: DatasetGeneratorConfig): Dataset {
  const { numSamples, noise, randomSeed } = config;
  const random = new SeededRandom(randomSeed);
  const points: DataPoint[] = [];
  const colors = ['#ff6b6b', '#4ecdc4'];
  
  for (let i = 0; i < numSamples; i++) {
    const x = random.next() * 2 - 1; // Range [-1, 1]
    const y = random.next() * 2 - 1; // Range [-1, 1]
    
    // XOR logic: class 1 if (x > 0) XOR (y > 0), class 0 otherwise
    const label = (x > 0) !== (y > 0) ? 1 : 0;
    
    points.push({
      id: `xor_${i}`,
      x: x + random.normal(0, noise),
      y: y + random.normal(0, noise),
      label,
      color: colors[label]
    });
  }
  
  return {
    id: 'xor',
    name: 'XOR Dataset',
    description: 'XOR classification problem dataset',
    points,
    features: ['x', 'y'],
    target: 'class',
    type: 'classification'
  };
}

/**
 * Generate gaussian blobs for classification
 */
export function generateBlobsDataset(config: {
  numSamples: number;
  numClasses: number;
  clusterStd: number;
  randomSeed?: number;
}): Dataset {
  const { numSamples, numClasses, clusterStd, randomSeed } = config;
  const random = new SeededRandom(randomSeed);
  const points: DataPoint[] = [];
  const colors = generateColorPalette(numClasses);
  
  const samplesPerClass = Math.floor(numSamples / numClasses);
  
  // Generate cluster centers
  const centers: { x: number, y: number }[] = [];
  for (let i = 0; i < numClasses; i++) {
    centers.push({
      x: random.normal(0, 2),
      y: random.normal(0, 2)
    });
  }
  
  for (let classIndex = 0; classIndex < numClasses; classIndex++) {
    const center = centers[classIndex];
    
    for (let i = 0; i < samplesPerClass; i++) {
      const x = center.x + random.normal(0, clusterStd);
      const y = center.y + random.normal(0, clusterStd);
      
      points.push({
        id: `blob_${classIndex}_${i}`,
        x,
        y,
        label: classIndex,
        color: colors[classIndex]
      });
    }
  }
  
  return {
    id: 'blobs',
    name: 'Gaussian Blobs',
    description: `${numClasses}-class gaussian blob classification dataset`,
    points,
    features: ['x', 'y'],
    target: 'class',
    type: 'classification'
  };
}

// ============================================================================
// REGRESSION DATASETS
// ============================================================================

/**
 * Generate linear regression dataset
 */
export function generateLinearDataset(config: RegressionDatasetConfig): Dataset {
  const { numSamples, coefficients, intercept, noise, randomSeed } = config;
  const random = new SeededRandom(randomSeed);
  const points: DataPoint[] = [];
  
  for (let i = 0; i < numSamples; i++) {
    const x = (i / (numSamples - 1)) * 4 - 2; // Range [-2, 2]
    let y = intercept;
    
    // Calculate polynomial value
    for (let j = 0; j < coefficients.length; j++) {
      y += coefficients[j] * Math.pow(x, j + 1);
    }
    
    y += random.normal(0, noise);
    
    points.push({
      id: `linear_${i}`,
      x,
      y
    });
  }
  
  return {
    id: 'linear',
    name: 'Linear Regression Dataset',
    description: 'Dataset for linear regression with configurable noise',
    points,
    features: ['x'],
    target: 'y',
    type: 'regression'
  };
}

/**
 * Generate polynomial regression dataset
 */
export function generatePolynomialDataset(config: RegressionDatasetConfig): Dataset {
  const { numSamples, coefficients, intercept, noise, randomSeed } = config;
  const random = new SeededRandom(randomSeed);
  const points: DataPoint[] = [];
  
  for (let i = 0; i < numSamples; i++) {
    const x = (i / (numSamples - 1)) * 4 - 2; // Range [-2, 2]
    let y = intercept;
    
    // Calculate polynomial value
    for (let j = 0; j < coefficients.length; j++) {
      y += coefficients[j] * Math.pow(x, j + 1);
    }
    
    y += random.normal(0, noise);
    
    points.push({
      id: `poly_${i}`,
      x,
      y
    });
  }
  
  return {
    id: 'polynomial',
    name: 'Polynomial Regression Dataset',
    description: `Polynomial regression dataset (degree ${coefficients.length})`,
    points,
    features: ['x'],
    target: 'y',
    type: 'regression'
  };
}

/**
 * Generate sinusoidal dataset for regression
 */
export function generateSinusoidalDataset(config: {
  numSamples: number;
  amplitude: number;
  frequency: number;
  phase: number;
  noise: number;
  randomSeed?: number;
}): Dataset {
  const { numSamples, amplitude, frequency, phase, noise, randomSeed } = config;
  const random = new SeededRandom(randomSeed);
  const points: DataPoint[] = [];
  
  for (let i = 0; i < numSamples; i++) {
    const x = (i / (numSamples - 1)) * 4 * Math.PI; // Range [0, 4π]
    const y = amplitude * Math.sin(frequency * x + phase) + random.normal(0, noise);
    
    points.push({
      id: `sin_${i}`,
      x,
      y
    });
  }
  
  return {
    id: 'sinusoidal',
    name: 'Sinusoidal Dataset',
    description: 'Sinusoidal regression dataset with configurable frequency and noise',
    points,
    features: ['x'],
    target: 'y',
    type: 'regression'
  };
}

// ============================================================================
// CLUSTERING DATASETS
// ============================================================================

/**
 * Generate random dataset for clustering
 */
export function generateRandomClustersDataset(config: {
  numSamples: number;
  numClusters: number;
  clusterStd: number;
  randomSeed?: number;
}): Dataset {
  const { numSamples, numClusters, clusterStd, randomSeed } = config;
  const random = new SeededRandom(randomSeed);
  const points: DataPoint[] = [];
  
  // Generate cluster centers
  const centers: { x: number, y: number }[] = [];
  for (let i = 0; i < numClusters; i++) {
    centers.push({
      x: random.normal(0, 3),
      y: random.normal(0, 3)
    });
  }
  
  const samplesPerCluster = Math.floor(numSamples / numClusters);
  
  for (let clusterIndex = 0; clusterIndex < numClusters; clusterIndex++) {
    const center = centers[clusterIndex];
    
    for (let i = 0; i < samplesPerCluster; i++) {
      const x = center.x + random.normal(0, clusterStd);
      const y = center.y + random.normal(0, clusterStd);
      
      points.push({
        id: `cluster_${clusterIndex}_${i}`,
        x,
        y
      });
    }
  }
  
  // Add remaining samples to random clusters
  const remainingSamples = numSamples - (samplesPerCluster * numClusters);
  for (let i = 0; i < remainingSamples; i++) {
    const clusterIndex = Math.floor(random.next() * numClusters);
    const center = centers[clusterIndex];
    const x = center.x + random.normal(0, clusterStd);
    const y = center.y + random.normal(0, clusterStd);
    
    points.push({
      id: `cluster_extra_${i}`,
      x,
      y
    });
  }
  
  return {
    id: 'random_clusters',
    name: 'Random Clusters',
    description: `${numClusters} random clusters for clustering algorithms`,
    points,
    features: ['x', 'y'],
    type: 'clustering'
  };
}

/**
 * Generate elongated clusters dataset
 */
export function generateElongatedClustersDataset(config: DatasetGeneratorConfig): Dataset {
  const { numSamples, randomSeed } = config;
  const random = new SeededRandom(randomSeed);
  const points: DataPoint[] = [];
  
  const samplesPerCluster = Math.floor(numSamples / 3);
  
  // Horizontal elongated cluster
  for (let i = 0; i < samplesPerCluster; i++) {
    const x = random.normal(0, 2);
    const y = random.normal(0, 0.3);
    points.push({
      id: `elongated_h_${i}`,
      x,
      y
    });
  }
  
  // Vertical elongated cluster
  for (let i = 0; i < samplesPerCluster; i++) {
    const x = random.normal(3, 0.3);
    const y = random.normal(0, 2);
    points.push({
      id: `elongated_v_${i}`,
      x,
      y
    });
  }
  
  // Diagonal elongated cluster
  for (let i = 0; i < samplesPerCluster; i++) {
    const t = random.normal(0, 1);
    const x = -2 + t * 0.8 + random.normal(0, 0.2);
    const y = -2 + t * 0.8 + random.normal(0, 0.2);
    points.push({
      id: `elongated_d_${i}`,
      x,
      y
    });
  }
  
  return {
    id: 'elongated_clusters',
    name: 'Elongated Clusters',
    description: 'Three elongated clusters with different orientations',
    points,
    features: ['x', 'y'],
    type: 'clustering'
  };
}

// ============================================================================
// PRESET DATASETS
// ============================================================================

/**
 * Get predefined datasets
 */
export function getPresetDatasets(): Dataset[] {
  return [
    // Classification datasets
    generateSpiralDataset({
      numSamples: 200,
      numClasses: 2,
      spiralTightness: 3,
      noise: 0.1,
      randomSeed: 42,
      dimensions: 2
    }),
    generateCircleDataset({
      numSamples: 200,
      radius: 2,
      innerRadius: 0.5,
      noise: 0.1,
      randomSeed: 42,
      dimensions: 2
    }),
    generateXORDataset({
      numSamples: 200,
      noise: 0.1,
      randomSeed: 42,
      dimensions: 2
    }),
    generateBlobsDataset({
      numSamples: 200,
      numClasses: 3,
      clusterStd: 0.5,
      randomSeed: 42
    }),
    
    // Regression datasets
    generateLinearDataset({
      numSamples: 100,
      coefficients: [1],
      intercept: 0,
      polynomialDegree: 1,
      noise: 0.3,
      randomSeed: 42,
      dimensions: 1
    }),
    generatePolynomialDataset({
      numSamples: 100,
      coefficients: [1, -0.5, 0.2],
      intercept: 0,
      polynomialDegree: 3,
      noise: 0.2,
      randomSeed: 42,
      dimensions: 1
    }),
    generateSinusoidalDataset({
      numSamples: 100,
      amplitude: 1,
      frequency: 1,
      phase: 0,
      noise: 0.1,
      randomSeed: 42
    }),
    
    // Clustering datasets
    generateRandomClustersDataset({
      numSamples: 200,
      numClusters: 4,
      clusterStd: 0.7,
      randomSeed: 42
    }),
    generateElongatedClustersDataset({
      numSamples: 150,
      noise: 0.1,
      randomSeed: 42,
      dimensions: 2
    })
  ];
}

/**
 * Get dataset by ID
 */
export function getDatasetById(id: string): Dataset | null {
  const presets = getPresetDatasets();
  return presets.find(dataset => dataset.id === id) || null;
}

/**
 * Get datasets by type
 */
export function getDatasetsByType(type: 'classification' | 'regression' | 'clustering'): Dataset[] {
  const presets = getPresetDatasets();
  return presets.filter(dataset => dataset.type === type);
}
