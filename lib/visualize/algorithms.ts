// Core algorithm implementations for the Interactive Model Visualization Section

import * as tf from '@tensorflow/tfjs';
import { 
  DataPoint, 
  LinearRegressionParams, 
  ClusteringParams, 
  RegressionResult, 
  ClusteringResult, 
  Point2D 
} from './types';
import { mean, distance } from './utils';

// ============================================================================
// LINEAR REGRESSION ALGORITHMS
// ============================================================================

/**
 * Polynomial feature transformation
 */
function polynomialFeatures(x: number[], degree: number): number[][] {
  const features: number[][] = [];
  
  for (let i = 0; i < x.length; i++) {
    const row: number[] = [];
    for (let d = 1; d <= degree; d++) {
      row.push(Math.pow(x[i], d));
    }
    features.push(row);
  }
  
  return features;
}

/**
 * Linear regression using normal equation
 */
export async function linearRegressionNormal(
  points: DataPoint[], 
  params: LinearRegressionParams
): Promise<RegressionResult> {
  const { polynomialDegree, regularizationStrength, regularizationType } = params;
  
  const x = points.map(p => p.x);
  const y = points.map(p => p.y);
  
  // Create polynomial features
  const X = polynomialFeatures(x, polynomialDegree);
  
  // Add bias term (intercept)
  const XWithBias = X.map(row => [1, ...row]);
  
  // Convert to tensors
  const XTensor = tf.tensor2d(XWithBias);
  const yTensor = tf.tensor1d(y);
  
  let weights: tf.Tensor;
  
  if (regularizationType === 'ridge' && regularizationStrength > 0) {
    // Ridge regression: Use gradient descent for ridge regression to avoid matrix inversion issues
    const learningRate = 0.01;
    const maxIterations = 1000;
    
    // Initialize weights
    weights = tf.randomNormal([XTensor.shape[1]]);
    
    for (let i = 0; i < maxIterations; i++) {
      const predictions = XTensor.matMul(weights.expandDims(1)).squeeze();
      const errors = predictions.sub(yTensor);
      const loss = errors.square().mean();
      
      // Ridge regularization: add λ * ||w||²
      const regularizationTerm = weights.square().sum().mul(regularizationStrength);
      const totalLoss = loss.add(regularizationTerm);
      
      // Calculate gradients
      const dataGradients = XTensor.transpose().matMul(errors.expandDims(1)).squeeze().div(XTensor.shape[0]);
      const regGradients = weights.mul(2 * regularizationStrength);
      const gradients = dataGradients.add(regGradients);
      
      // Update weights
      const newWeights = weights.sub(gradients.mul(learningRate));
      weights.dispose();
      weights = newWeights;
      
      // Clean up
      predictions.dispose();
      errors.dispose();
      loss.dispose();
      regularizationTerm.dispose();
      totalLoss.dispose();
      dataGradients.dispose();
      regGradients.dispose();
      gradients.dispose();
      
      // Early stopping check (optional)
      if (i % 100 === 0) {
        const lossData = await totalLoss.data();
        if (lossData[0] < 1e-6) break;
      }
    }
  } else {
    // Ordinary least squares: Use simple gradient descent to avoid matrix inversion
    const learningRate = 0.01;
    const maxIterations = 1000;
    
    // Initialize weights
    weights = tf.randomNormal([XTensor.shape[1]]);
    
    for (let i = 0; i < maxIterations; i++) {
      const predictions = XTensor.matMul(weights.expandDims(1)).squeeze();
      const errors = predictions.sub(yTensor);
      const loss = errors.square().mean();
      
      // Calculate gradients
      const gradients = XTensor.transpose().matMul(errors.expandDims(1)).squeeze().div(XTensor.shape[0]);
      
      // Update weights
      const newWeights = weights.sub(gradients.mul(learningRate));
      weights.dispose();
      weights = newWeights;
      
      // Clean up
      predictions.dispose();
      errors.dispose();
      loss.dispose();
      gradients.dispose();
      
      // Early stopping check
      if (i % 100 === 0) {
        const lossData = await loss.data();
        if (lossData[0] < 1e-6) break;
      }
    }
  }
  
  // Make predictions
  const predictions = XTensor.matMul(weights.expandDims(1)).squeeze();
  
  // Calculate metrics
  const predArray = await predictions.data();
  const weightsArray = await weights.data();
  
  const residuals = y.map((actual, i) => actual - predArray[i]);
  const yMean = mean(y);
  const totalSumSquares = y.reduce((sum, val) => sum + Math.pow(val - yMean, 2), 0);
  const residualSumSquares = residuals.reduce((sum, val) => sum + Math.pow(val, 2), 0);
  const rSquared = 1 - (residualSumSquares / totalSumSquares);
  
  // Clean up tensors
  XTensor.dispose();
  yTensor.dispose();
  weights.dispose();
  predictions.dispose();
  
  return {
    predictions: Array.from(predArray),
    coefficients: Array.from(weightsArray).slice(1), // Remove bias term
    intercept: weightsArray[0],
    residuals,
    rSquared,
    parameters: params,
    metrics: {
      trainMetrics: [],
      r2Score: rSquared,
      mse: mean(residuals.map(r => r * r)),
      mae: mean(residuals.map(r => Math.abs(r)))
    }
  };
}

/**
 * Linear regression using gradient descent
 */
export async function linearRegressionGradientDescent(
  points: DataPoint[], 
  params: LinearRegressionParams,
  onEpochComplete?: (epoch: number, loss: number, weights: number[]) => void
): Promise<RegressionResult> {
  const { polynomialDegree, learningRate, iterations } = params;
  
  const x = points.map(p => p.x);
  const y = points.map(p => p.y);
  
  // Create polynomial features
  const X = polynomialFeatures(x, polynomialDegree);
  const XWithBias = X.map(row => [1, ...row]);
  
  // Convert to tensors
  const XTensor = tf.tensor2d(XWithBias);
  const yTensor = tf.tensor1d(y);
  
  // Initialize weights
  let weights = tf.randomNormal([XTensor.shape[1]]);
  
  const convergenceHistory: number[] = [];
  
  // Gradient descent
  for (let i = 0; i < iterations; i++) {
    const predictions = XTensor.matMul(weights.expandDims(1)).squeeze();
    const errors = predictions.sub(yTensor);
    const loss = errors.square().mean();
    
    // Calculate gradients
    const gradients = XTensor.transpose().matMul(errors.expandDims(1)).squeeze().div(XTensor.shape[0]);
    
    // Update weights
    weights = weights.sub(gradients.mul(learningRate));
    
    const lossValue = await loss.data();
    convergenceHistory.push(lossValue[0]);
    
    if (onEpochComplete) {
      const weightsArray = await weights.data();
      onEpochComplete(i, lossValue[0], Array.from(weightsArray));
    }
    
    // Clean up intermediate tensors
    predictions.dispose();
    errors.dispose();
    loss.dispose();
    gradients.dispose();
  }
  
  // Final predictions
  const finalPredictions = XTensor.matMul(weights.expandDims(1)).squeeze();
  const predArray = await finalPredictions.data();
  const weightsArray = await weights.data();
  
  const residuals = y.map((actual, i) => actual - predArray[i]);
  const yMean = mean(y);
  const totalSumSquares = y.reduce((sum, val) => sum + Math.pow(val - yMean, 2), 0);
  const residualSumSquares = residuals.reduce((sum, val) => sum + Math.pow(val, 2), 0);
  const rSquared = 1 - (residualSumSquares / totalSumSquares);
  
  // Clean up tensors
  XTensor.dispose();
  yTensor.dispose();
  weights.dispose();
  finalPredictions.dispose();
  
  return {
    predictions: Array.from(predArray),
    coefficients: Array.from(weightsArray).slice(1),
    intercept: weightsArray[0],
    residuals,
    rSquared,
    convergenceHistory,
    parameters: params,
    metrics: {
      trainMetrics: [],
      r2Score: rSquared,
      mse: mean(residuals.map(r => r * r)),
      mae: mean(residuals.map(r => Math.abs(r)))
    }
  };
}

// ============================================================================
// CLUSTERING ALGORITHMS
// ============================================================================

/**
 * K-Means clustering algorithm
 */
export function kMeansClustering(
  points: DataPoint[], 
  params: ClusteringParams,
  onIterationComplete?: (iteration: number, centroids: Point2D[], assignments: number[]) => void
): ClusteringResult {
  const { numClusters } = params;
  const maxIterations = 100;
  const tolerance = 1e-6;
  
  // Initialize centroids randomly
  let centroids: Point2D[] = [];
  for (let i = 0; i < numClusters; i++) {
    const randomPoint = points[Math.floor(Math.random() * points.length)];
    centroids.push({
      x: randomPoint.x + (Math.random() - 0.5) * 0.1,
      y: randomPoint.y + (Math.random() - 0.5) * 0.1
    });
  }
  
  let assignments: number[] = new Array(points.length).fill(0);
  let previousCentroids: Point2D[] = [];
  
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    // Assign points to nearest centroid
    for (let i = 0; i < points.length; i++) {
      let minDistance = Infinity;
      let nearestCentroid = 0;
      
      for (let j = 0; j < centroids.length; j++) {
        const dist = distance(points[i], centroids[j]);
        if (dist < minDistance) {
          minDistance = dist;
          nearestCentroid = j;
        }
      }
      
      assignments[i] = nearestCentroid;
    }
    
    // Update centroids
    previousCentroids = centroids.map(c => ({ ...c }));
    
    for (let j = 0; j < numClusters; j++) {
      const clusterPoints = points.filter((_, i) => assignments[i] === j);
      
      if (clusterPoints.length > 0) {
        centroids[j] = {
          x: mean(clusterPoints.map(p => p.x)),
          y: mean(clusterPoints.map(p => p.y))
        };
      }
    }
    
    if (onIterationComplete) {
      onIterationComplete(iteration, [...centroids], [...assignments]);
    }
    
    // Check for convergence
    const centroidMovement = centroids.reduce((sum, centroid, i) => {
      return sum + distance(centroid, previousCentroids[i]);
    }, 0);
    
    if (centroidMovement < tolerance) {
      break;
    }
  }
  
  // Calculate inertia (within-cluster sum of squares)
  let inertia = 0;
  for (let i = 0; i < points.length; i++) {
    const centroid = centroids[assignments[i]];
    inertia += Math.pow(distance(points[i], centroid), 2);
  }
  
  // Calculate cluster sizes
  const clusterSizes = new Array(numClusters).fill(0);
  assignments.forEach(assignment => clusterSizes[assignment]++);
  
  return {
    labels: assignments,
    centroids,
    inertia,
    clusterSizes,
    predictions: assignments,
    parameters: params,
    metrics: {
      trainMetrics: []
    }
  };
}

/**
 * DBSCAN clustering algorithm
 */
export function dbscanClustering(
  points: DataPoint[], 
  params: ClusteringParams
): ClusteringResult {
  const { epsilon, minSamples } = params;
  const labels: number[] = new Array(points.length).fill(-1); // -1 means noise
  let clusterId = 0;
  
  for (let i = 0; i < points.length; i++) {
    if (labels[i] !== -1) continue; // Already processed
    
    const neighbors = getNeighbors(points, i, epsilon);
    
    if (neighbors.length < minSamples) {
      labels[i] = -1; // Mark as noise
    } else {
      expandCluster(points, labels, i, neighbors, clusterId, epsilon, minSamples);
      clusterId++;
    }
  }
  
  // Calculate centroids for each cluster
  const centroids: Point2D[] = [];
  for (let i = 0; i < clusterId; i++) {
    const clusterPoints = points.filter((_, idx) => labels[idx] === i);
    if (clusterPoints.length > 0) {
      centroids.push({
        x: mean(clusterPoints.map(p => p.x)),
        y: mean(clusterPoints.map(p => p.y))
      });
    }
  }
  
  // Calculate cluster sizes
  const clusterSizes = new Array(clusterId).fill(0);
  labels.forEach(label => {
    if (label >= 0) clusterSizes[label]++;
  });
  
  return {
    labels,
    centroids,
    clusterSizes,
    predictions: labels,
    parameters: params,
    metrics: {
      trainMetrics: []
    }
  };
}

function getNeighbors(points: DataPoint[], pointIndex: number, epsilon: number): number[] {
  const neighbors: number[] = [];
  const point = points[pointIndex];
  
  for (let i = 0; i < points.length; i++) {
    if (distance(point, points[i]) <= epsilon) {
      neighbors.push(i);
    }
  }
  
  return neighbors;
}

function expandCluster(
  points: DataPoint[],
  labels: number[],
  pointIndex: number,
  neighbors: number[],
  clusterId: number,
  epsilon: number,
  minSamples: number
): void {
  labels[pointIndex] = clusterId;
  
  let i = 0;
  while (i < neighbors.length) {
    const neighborIndex = neighbors[i];
    
    if (labels[neighborIndex] === -1) {
      labels[neighborIndex] = clusterId;
    }
    
    if (labels[neighborIndex] !== -1) {
      i++;
      continue;
    }
    
    labels[neighborIndex] = clusterId;
    
    const newNeighbors = getNeighbors(points, neighborIndex, epsilon);
    if (newNeighbors.length >= minSamples) {
      neighbors.push(...newNeighbors);
    }
    
    i++;
  }
}

// ============================================================================
// UTILITY FUNCTIONS FOR ALGORITHMS
// ============================================================================

/**
 * Calculate silhouette score for clustering
 */
export function calculateSilhouetteScore(points: DataPoint[], labels: number[]): number {
  const silhouetteScores: number[] = [];
  
  for (let i = 0; i < points.length; i++) {
    const label = labels[i];
    if (label === -1) continue; // Skip noise points
    
    // Calculate a(i) - average distance to points in same cluster
    const sameClusterPoints = points.filter((_, idx) => labels[idx] === label && idx !== i);
    const a = sameClusterPoints.length > 0 
      ? mean(sameClusterPoints.map(p => distance(points[i], p)))
      : 0;
    
    // Calculate b(i) - minimum average distance to points in other clusters
    const otherClusters = [...new Set(labels.filter(l => l !== label && l !== -1))];
    let b = Infinity;
    
    for (const otherLabel of otherClusters) {
      const otherClusterPoints = points.filter((_, idx) => labels[idx] === otherLabel);
      if (otherClusterPoints.length > 0) {
        const avgDistance = mean(otherClusterPoints.map(p => distance(points[i], p)));
        b = Math.min(b, avgDistance);
      }
    }
    
    if (b === Infinity) b = 0;
    
    // Calculate silhouette score for point i
    const silhouette = a === 0 && b === 0 ? 0 : (b - a) / Math.max(a, b);
    silhouetteScores.push(silhouette);
  }
  
  return silhouetteScores.length > 0 ? mean(silhouetteScores) : 0;
}

/**
 * Calculate elbow method scores for K-means
 */
export function calculateElbowScores(
  points: DataPoint[], 
  maxK: number = 10
): { k: number, score: number }[] {
  const scores: { k: number, score: number }[] = [];
  
  for (let k = 1; k <= maxK; k++) {
    const result = kMeansClustering(points, {
      algorithm: 'kmeans',
      numClusters: k,
      epsilon: 0,
      minSamples: 0,
      linkage: 'ward'
    });
    
    scores.push({
      k,
      score: result.inertia || 0
    });
  }
  
  return scores;
}

/**
 * Generate prediction grid for decision boundary visualization
 */
export function generatePredictionGrid(
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

/**
 * Predict values for a grid of points using linear regression
 */
export function predictOnGrid(
  grid: Point2D[],
  coefficients: number[],
  intercept: number,
  polynomialDegree: number
): number[] {
  return grid.map(point => {
    let prediction = intercept;
    for (let d = 1; d <= polynomialDegree; d++) {
      if (coefficients[d - 1] !== undefined) {
        prediction += coefficients[d - 1] * Math.pow(point.x, d);
      }
    }
    return prediction;
  });
}
