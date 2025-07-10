import TopicPageBuilder from '../../components/TopicPageBuilder';
import { BarChart3, Target, CheckCircle, TrendingUp, AlertCircle, Book, ExternalLink, Search, BookOpen, Code } from 'lucide-react';

export const metadata = {
  title: 'Model Evaluation Metrics - ML Portfolio',
  description: 'Metrics and techniques for assessing machine learning model performance across different tasks',
};

const evaluationTopicData = {
  title: "Model Evaluation Metrics",
  header: {
    category: "Fundamentals",
    difficulty: "Beginner" as const,
    readTime: "11 min read",
      description: "Comprehensive guide to metrics and techniques for assessing machine learning model performance across classification, regression, and other tasks",
      relatedProjects: ["image-classifier", "tamid-image-classifier", "real-salary", "clustering-exploration"],
      gradientFrom: "from-blue-50 to-indigo-50",
      gradientTo: "dark:from-blue-900/20 dark:to-indigo-900/20",
      borderColor: "border-blue-200 dark:border-blue-800"
    },
    tags: {
      items: ["Evaluation", "Performance", "Metrics", "Validation"],
      colorScheme: "blue" as const
    },
    blocks: [
      {
        type: 'section' as const,
        props: { title: 'Why Evaluation Matters' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Model evaluation is crucial for understanding how well your machine learning model performs on unseen data. Proper evaluation helps you compare different models, tune hyperparameters, and ensure your model will generalize well to real-world scenarios. The choice of evaluation metrics depends on your specific problem type and business objectives.'
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Classification Metrics' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Classification tasks involve predicting discrete class labels. Here are the most important metrics:'
          },
          {
            type: 'section' as const,
            props: { title: 'Basic Classification Metrics' },
            children: [
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  code: `import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Basic classification metrics
def calculate_metrics(y_true, y_pred):
    """
    Calculate basic classification metrics
    """
    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision: TP / (TP + FP)
    precision = precision_score(y_true, y_pred, average='weighted')
    
    # Recall (Sensitivity): TP / (TP + FN)
    recall = recall_score(y_true, y_pred, average='weighted')
    
    # F1 Score: 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Create and display confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    print("Confusion Matrix:")
    print("True\\Predicted", end="\\t")
    for name in class_names:
        print(f"{name}\\t", end="")
    print()
    
    for i, true_name in enumerate(class_names):
        print(f"{true_name}\\t\\t", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]}\\t", end="")
        print()
    
    return cm

# Example usage
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2])
class_names = ['Class A', 'Class B', 'Class C']

metrics = calculate_metrics(y_true, y_pred)
print("Classification Metrics:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

cm = plot_confusion_matrix(y_true, y_pred, class_names)`
                }
              }
            ]
          },
          {
            type: 'section' as const,
            props: { title: 'Advanced Classification Metrics' },
            children: [
              {
                type: 'paragraph' as const,
                content: 'For more detailed evaluation, especially with imbalanced datasets:'
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  code: `from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import average_precision_score, log_loss
import matplotlib.pyplot as plt

# ROC AUC and PR AUC
def advanced_classification_metrics(y_true, y_pred_proba, y_pred):
    """
    Calculate advanced classification metrics
    """
    # For binary classification
    if len(np.unique(y_true)) == 2:
        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        # Precision-Recall AUC
        pr_auc = average_precision_score(y_true, y_pred_proba[:, 1])
        
        # Log Loss
        logloss = log_loss(y_true, y_pred_proba)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'log_loss': logloss
        }
    
    # For multiclass
    else:
        # Multi-class ROC AUC (one-vs-rest)
        roc_auc_ovr = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        
        # Multi-class ROC AUC (one-vs-one)
        roc_auc_ovo = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')
        
        # Log Loss
        logloss = log_loss(y_true, y_pred_proba)
        
        return {
            'roc_auc_ovr': roc_auc_ovr,
            'roc_auc_ovo': roc_auc_ovo,
            'log_loss': logloss
        }

# Custom metrics for specific use cases
def specificity_score(y_true, y_pred):
    """
    Calculate specificity (True Negative Rate)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def balanced_accuracy(y_true, y_pred):
    """
    Balanced accuracy for imbalanced datasets
    """
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)

# Matthews Correlation Coefficient
def mcc_score(y_true, y_pred):
    """
    Matthews Correlation Coefficient - good for imbalanced datasets
    """
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(y_true, y_pred)`
                }
              }
            ]
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Regression Metrics' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Regression tasks involve predicting continuous values. Common metrics include:'
          },
          {
            type: 'codeBlock' as const,
            props: {
              language: 'python',
              code: `from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error

def regression_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics
    """
    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # R² Score (Coefficient of Determination)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Median Absolute Error
    median_ae = median_absolute_error(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'median_ae': median_ae
    }

# Custom regression metrics
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """
    Mean Absolute Scaled Error - good for time series
    """
    n = len(y_train)
    mae = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    return mae / mae_naive

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Symmetric MAPE - handles zero values better
    """
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

# Example usage
y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
y_pred = np.array([2.5, 0.0, 2.1, 7.8, 4.0])

reg_metrics = regression_metrics(y_true, y_pred)
print("Regression Metrics:")
for metric, value in reg_metrics.items():
    print(f"{metric.upper()}: {value:.4f}")

# Residual analysis
def residual_analysis(y_true, y_pred):
    """
    Analyze residuals for regression models
    """
    residuals = y_true - y_pred
    
    print(f"Residual Statistics:")
    print(f"Mean: {np.mean(residuals):.4f}")
    print(f"Std: {np.std(residuals):.4f}")
    print(f"Min: {np.min(residuals):.4f}")
    print(f"Max: {np.max(residuals):.4f}")
    
    return residuals`
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Cross-Validation' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Cross-validation provides a more robust evaluation by testing on multiple train/test splits:'
          },
          {
            type: 'codeBlock' as const,
            props: {
              language: 'python',
              code: `from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

def comprehensive_cross_validation(model, X, y, task_type='classification'):
    """
    Perform comprehensive cross-validation
    """
    if task_type == 'classification':
        # Use stratified k-fold for classification
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    else:
        # Use regular k-fold for regression
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    # Calculate statistics
    results = {}
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results[metric] = {
            'test_mean': np.mean(test_scores),
            'test_std': np.std(test_scores),
            'train_mean': np.mean(train_scores),
            'train_std': np.std(train_scores)
        }
    
    return results

# Learning curves
def plot_learning_curves(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plot learning curves to diagnose overfitting/underfitting
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    print("Learning Curve Data:")
    print("Train Size\\tTrain Score\\tVal Score")
    for i, size in enumerate(train_sizes):
        print(f"{size:.2f}\\t\\t{train_mean[i]:.4f}\\t\\t{val_mean[i]:.4f}")
    
    return train_sizes, train_mean, train_std, val_mean, val_std

# Time series cross-validation
def time_series_cv(model, X, y, n_splits=5):
    """
    Time series cross-validation with forward chaining
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.array(scores)`
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Model Comparison and Selection' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Statistical methods for comparing multiple models:'
          },
          {
            type: 'codeBlock' as const,
            props: {
              language: 'python',
              code: `from scipy import stats
import pandas as pd

def compare_models(model_results):
    """
    Compare multiple models using statistical tests
    
    Args:
        model_results: dict of model_name -> list of cv scores
    """
    # Create DataFrame for easy comparison
    df = pd.DataFrame(model_results)
    
    print("Model Comparison:")
    print("=" * 50)
    print(f"{'Model':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 50)
    
    for model_name, scores in model_results.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        print(f"{model_name:<20} {mean_score:<10.4f} {std_score:<10.4f} {min_score:<10.4f} {max_score:<10.4f}")
    
    # Perform pairwise t-tests
    print("\\nPairwise t-tests (p-values):")
    print("=" * 30)
    
    model_names = list(model_results.keys())
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                t_stat, p_value = stats.ttest_rel(model_results[model1], model_results[model2])
                print(f"{model1} vs {model2}: p = {p_value:.4f}")
    
    return df

# Example usage
model_results = {
    'Random Forest': [0.85, 0.87, 0.82, 0.88, 0.86],
    'SVM': [0.83, 0.85, 0.81, 0.84, 0.82],
    'Logistic Regression': [0.79, 0.81, 0.78, 0.82, 0.80]
}

comparison_df = compare_models(model_results)

# McNemar's test for classifier comparison
def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    McNemar's test for comparing two classifiers
    """
    from statsmodels.stats.contingency_tables import mcnemar
    
    # Create contingency table
    correct1 = (y_true == y_pred1)
    correct2 = (y_true == y_pred2)
    
    both_correct = np.sum(correct1 & correct2)
    model1_correct = np.sum(correct1 & ~correct2)
    model2_correct = np.sum(~correct1 & correct2)
    both_wrong = np.sum(~correct1 & ~correct2)
    
    contingency_table = np.array([[both_correct, model1_correct],
                                  [model2_correct, both_wrong]])
    
    result = mcnemar(contingency_table, exact=True)
    return result.pvalue`
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Best Practices' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Guidelines for effective model evaluation:'
          },
          {
            type: 'list' as const,
            props: {
              items: [
                'Always use a held-out test set that is never used during model development',
                'Choose metrics that align with your business objectives',
                'Use multiple metrics to get a comprehensive view of model performance',
                'Consider class imbalance when selecting classification metrics',
                'Validate on data that represents your target distribution',
                'Document your evaluation methodology for reproducibility',
                'Be aware of data leakage and temporal dependencies'
              ]
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Metric Selection Guide' },
        children: [
          {
            type: 'features' as const,
            props: {
              features: [
                {
                  title: 'Balanced Classification',
                  description: 'Use accuracy, precision, recall, and F1-score for balanced datasets'
                },
                {
                  title: 'Imbalanced Classification',
                  description: 'Use precision-recall AUC, F1-score, and Matthews correlation coefficient'
                },
                {
                  title: 'Regression',
                  description: 'Use RMSE for penalty on large errors, MAE for robust evaluation, R² for explained variance'
                },
                {
                  title: 'Ranking/Recommendation',
                  description: 'Use precision@k, recall@k, NDCG, and MAP for ranked results'
                }
              ]
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: {
          title: "Learning Resources",
          background: true
        },
        children: [
          {
            type: 'paragraph' as const,
            content: "Comprehensive resources to deepen your understanding of evaluation metrics:"
          },
          {
            type: 'twoColumn' as const,
            props: {
              left: [
                {
                  type: 'heading' as const,
                  props: { level: 3 },
                  content: "Essential Papers"
                },
                {
                  type: 'features' as const,
                  props: {
                    features: [
                      {
                        icon: <Search className="w-6 h-6" />,
                        title: "A Survey on Evaluation of Large Language Models",
                        description: "Chang et al. (2023) - Comprehensive evaluation survey"
                      },
                      {
                        icon: <Search className="w-6 h-6" />,
                        title: "Beyond Accuracy: Behavioral Testing of NLP Models",
                        description: "Ribeiro et al. (2020) - Beyond traditional metrics"
                      },
                      {
                        icon: <Search className="w-6 h-6" />,
                        title: "Measuring Catastrophic Forgetting in Neural Networks",
                        description: "Kirkpatrick et al. (2017) - Evaluation in continual learning"
                      }
                    ],
                    columns: 1
                  }
                }
              ],
              right: [
                {
                  type: 'heading' as const,
                  props: { level: 3 },
                  content: "Practical Resources"
                },
                {
                  type: 'features' as const,
                  props: {
                    features: [
                      {
                        icon: <BookOpen className="w-6 h-6" />,
                        title: "Scikit-learn Metrics Documentation",
                        description: "Comprehensive metric implementations and usage"
                      },
                      {
                        icon: <Code className="w-6 h-6" />,
                        title: "Hugging Face Evaluate Library",
                        description: "Modern evaluation toolkit for ML models"
                      },
                      {
                        icon: <ExternalLink className="w-6 h-6" />,
                        title: "Papers with Code Evaluation Guide",
                        description: "Best practices and benchmarks for model evaluation"
                      }
                    ],
                    columns: 1
                  }
                }
              ]
            }
          }
        ]
      }
    ]
  };

export default function EvaluationMetricsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-yellow-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-yellow-600 dark:hover:text-yellow-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-yellow-600 dark:hover:text-yellow-400 transition-colors">
                Projects
              </a>
              <a href="/topics" className="text-yellow-600 dark:text-yellow-400 font-medium">
                Topics
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <TopicPageBuilder {...evaluationTopicData} />
      </article>
    </div>
  );
}
