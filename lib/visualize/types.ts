// TypeScript interfaces and types for the Interactive Model Visualization Section

import * as tf from '@tensorflow/tfjs';

// ============================================================================
// CORE VISUALIZATION TYPES
// ============================================================================

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D extends Point2D {
  z: number;
}

export interface DataPoint extends Point2D {
  id: string;
  label?: number;
  color?: string;
  prediction?: number;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  points: DataPoint[];
  features: string[];
  target?: string;
  type: 'classification' | 'regression' | 'clustering';
}

// ============================================================================
// MODEL PARAMETER TYPES
// ============================================================================

export interface ModelParameters {
  [key: string]: number | string | boolean;
}

export interface LinearRegressionParams extends ModelParameters {
  polynomialDegree: number;
  regularizationType: 'none' | 'ridge' | 'lasso';
  regularizationStrength: number;
  learningRate: number;
  iterations: number;
}

export interface NeuralNetworkParams extends ModelParameters {
  layers: LayerConfig[];
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: 'sgd' | 'adam' | 'rmsprop';
  activation: 'relu' | 'tanh' | 'sigmoid' | 'linear';
  [key: string]: any; // Index signature for compatibility with ModelParameters
}

export interface DecisionTreeParams extends ModelParameters {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  criterion: 'gini' | 'entropy';
}

export interface ClusteringParams extends ModelParameters {
  algorithm: 'kmeans' | 'dbscan' | 'hierarchical';
  numClusters: number; // for k-means
  epsilon: number; // for DBSCAN
  minSamples: number; // for DBSCAN
  linkage: 'ward' | 'complete' | 'average' | 'single'; // for hierarchical
}

// ============================================================================
// NEURAL NETWORK SPECIFIC TYPES
// ============================================================================

export interface LayerConfig {
  id: string;
  type: 'dense' | 'conv2d' | 'maxPooling2d' | 'dropout' | 'batchNormalization';
  units?: number; // for dense layers
  activation?: string;
  inputShape?: number[];
  filters?: number; // for conv2d
  kernelSize?: number | [number, number]; // for conv2d
  poolSize?: number | [number, number]; // for pooling
  rate?: number; // for dropout
  position: Point2D;
}

export interface NetworkArchitecture {
  layers: LayerConfig[];
  connections: Connection[];
}

export interface Connection {
  from: string; // layer id
  to: string; // layer id
}

// ============================================================================
// TRAINING AND METRICS TYPES
// ============================================================================

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy?: number;
  valLoss?: number;
  valAccuracy?: number;
  timestamp: number;
}

export interface TrainingHistory {
  epoch: number;
  loss: number;
  accuracy?: number;
  valLoss?: number;
  valAccuracy?: number;
}

export interface TrainingProgress {
  currentEpoch: number;
  totalEpochs: number;
  progress: number; // 0-100
}

export interface ModelMetrics {
  // Core metrics
  accuracy: number | null;
  loss: number | null;

  // Regression metrics
  mse: number | null; // Mean Squared Error
  mae: number | null; // Mean Absolute Error
  r2: number | null; // R-squared

  // Classification metrics
  precision: number | null;
  recall: number | null;
  f1Score: number | null;

  // Training data
  lossHistory?: number[];
  validationLossHistory?: number[];
  trainingProgress?: TrainingProgress;

  // Additional custom metrics
  additionalMetrics?: { [key: string]: number };
}

export interface ModelPerformance {
  trainMetrics: TrainingMetrics[];
  testMetrics?: TrainingMetrics;
  confusion?: number[][];
  predictions?: number[];
  residuals?: number[];
  r2Score?: number;
  mse?: number;
  mae?: number;
}

// ============================================================================
// VISUALIZATION COMPONENT TYPES
// ============================================================================

export interface VisualizationConfig {
  type: VisualizationType;
  width: number;
  height: number;
  margin: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  showGrid: boolean;
  showAxes: boolean;
  interactive: boolean;
  theme: 'light' | 'dark';
  xAxisLabel?: string;
  yAxisLabel?: string;
  showPredictionLine?: boolean;
}

export interface PlotData {
  x: number[];
  y: number[];
  labels?: string[];
  colors?: string[];
  type: 'scatter' | 'line' | 'bar' | 'heatmap';
}

export interface AnimationConfig {
  duration: number;
  easing: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out';
  delay?: number;
}

// ============================================================================
// CONTROL PANEL TYPES
// ============================================================================

export interface ControlConfig {
  id: string;
  label: string;
  type: 'slider' | 'select' | 'checkbox' | 'input';
  value: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string | number; label: string }[];
  description?: string;
  category?: string;
}

export interface ControlPanelConfig {
  title: string;
  controls: ControlConfig[];
  categories?: string[];
  collapsible?: boolean;
}

// ============================================================================
// ALGORITHM IMPLEMENTATION TYPES
// ============================================================================

export interface AlgorithmResult {
  predictions: number[];
  model?: tf.LayersModel | any;
  parameters: ModelParameters;
  metrics: ModelPerformance;
  convergenceHistory?: number[];
  weights?: number[];
  biases?: number[];
}

export interface RegressionResult extends AlgorithmResult {
  coefficients: number[];
  intercept: number;
  residuals: number[];
  rSquared: number;
  predictions: number[];
}

export interface ClassificationResult extends AlgorithmResult {
  probabilities: number[][];
  classes: string[] | number[];
  confusionMatrix: number[][];
  accuracy: number;
  precision: number[];
  recall: number[];
  f1Score: number[];
}

export interface ClusteringResult extends AlgorithmResult {
  labels: number[];
  centroids: Point2D[];
  inertia?: number;
  silhouetteScore?: number;
  clusterSizes: number[];
}

// ============================================================================
// DATASET GENERATOR TYPES
// ============================================================================

export interface DatasetGeneratorConfig {
  numSamples: number;
  noise: number;
  randomSeed?: number;
  dimensions: number;
}

export interface SpiralDatasetConfig extends DatasetGeneratorConfig {
  numClasses: number;
  spiralTightness: number;
}

export interface CircleDatasetConfig extends DatasetGeneratorConfig {
  radius: number;
  innerRadius?: number;
}

export interface RegressionDatasetConfig extends DatasetGeneratorConfig {
  coefficients: number[];
  intercept: number;
  polynomialDegree: number;
}

// ============================================================================
// UI STATE TYPES
// ============================================================================

export interface VisualizationState {
  isTraining: boolean;
  isPaused: boolean;
  currentEpoch: number;
  totalEpochs: number;
  selectedDataset: string;
  parameters: ModelParameters;
  results?: AlgorithmResult;
  error?: string;
}

export interface UIState {
  sidebarOpen: boolean;
  controlPanelCollapsed: boolean;
  fullscreenMode: boolean;
  selectedTool?: string;
  zoom: number;
  pan: Point2D;
}

// ============================================================================
// EVENT TYPES
// ============================================================================

export interface ParameterChangeEvent {
  parameterId: string;
  value: number | string | boolean;
  previousValue: number | string | boolean;
}

export interface DataPointSelectEvent {
  point: DataPoint;
  index: number;
  event: MouseEvent;
}

export interface ModelUpdateEvent {
  type: 'parameter_change' | 'training_complete' | 'epoch_complete';
  data: any;
  timestamp: number;
}

// ============================================================================
// EXPORT TYPES
// ============================================================================

export interface ExportConfig {
  format: 'png' | 'svg' | 'pdf' | 'json';
  includeData: boolean;
  includeParameters: boolean;
  includeMetrics: boolean;
  resolution?: number;
}

export interface SavedConfiguration {
  id: string;
  name: string;
  modelType: string;
  parameters: ModelParameters;
  dataset: Dataset;
  results?: AlgorithmResult;
  timestamp: number;
  version: string;
}

// ============================================================================
// TENSORFLOW.JS INTEGRATION TYPES
// ============================================================================

export interface TensorFlowConfig {
  backend: 'cpu' | 'webgl' | 'wasm';
  enableDebugMode: boolean;
  logLevel: 'verbose' | 'info' | 'warn' | 'error';
}

export interface ModelCompileConfig {
  optimizer: string;
  loss: string;
  metrics?: string[];
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  validationSplit?: number;
  shuffle?: boolean;
  callbacks?: any[];
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

export type ModelType = 'linear-regression' | 'neural-network' | 'decision-tree' | 'clustering' | 'cnn';

export type VisualizationType = 'scatter' | 'line' | 'heatmap' | 'network' | 'tree' | 'surface';

export type InteractionMode = 'select' | 'pan' | 'zoom' | 'draw' | 'edit';

// Generic utility types
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type RequireOnly<T, K extends keyof T> = Partial<T> & Required<Pick<T, K>>;

// Event handler types
export type EventHandler<T = any> = (event: T) => void;

export type AsyncEventHandler<T = any> = (event: T) => Promise<void>;
