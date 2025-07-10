import { Database, Filter, BarChart3, Target, TrendingUp, Zap, Code, Settings, Eye, ExternalLink, CheckCircle, AlertTriangle } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Data Preprocessing - ML Portfolio Topics',
  description: 'Learn about data preprocessing - techniques for cleaning, transforming, and preparing data for machine learning',
};

const dataPreprocessingPageData = {
  title: "Data Preprocessing",
  header: {
    date: "Data Science",
    readTime: "7 min read",
    description: "Essential techniques for cleaning, transforming, and preparing data for machine learning models",
    gradientFrom: "from-yellow-50 to-orange-50",
    gradientTo: "dark:from-yellow-900/20 dark:to-orange-900/20",
    borderColor: "border-yellow-200 dark:border-yellow-800",
    difficulty: "Beginner" as const,
    category: "Data Science",
    relatedProjects: ["image-classifier", "tamid-image-classifier", "real-salary", "clustering-exploration"]
  },
  tags: {
    items: ['Data Science', 'Feature Engineering', 'Data Cleaning', 'Normalization', 'Exploratory Data Analysis'],
    colorScheme: 'yellow' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What is Data Preprocessing?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Data preprocessing is the process of preparing raw data for machine learning algorithms. It involves cleaning, transforming, and organizing data to improve model performance and ensure reliable results. Quality preprocessing often determines the success of a machine learning project."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Why It Matters",
            icon: <Database className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Real-world data is often messy, incomplete, and inconsistent. Proper preprocessing can dramatically improve model accuracy, reduce training time, and prevent overfitting."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Core Preprocessing Steps"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Filter className="w-6 h-6" />,
                title: "Data Cleaning",
                description: "Remove duplicates, handle missing values, and correct inconsistencies"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Exploratory Data Analysis",
                description: "Understand data distribution, relationships, and patterns"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Feature Engineering",
                description: "Create new features and transform existing ones for better model performance"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Normalization/Scaling",
                description: "Standardize feature ranges to prevent bias toward larger values"
              }
            ],
            columns: 2
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Handling Missing Data",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Missing data is one of the most common challenges in real-world datasets. The strategy you choose depends on the nature and amount of missing data."
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Deletion Methods"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <AlertTriangle className="w-6 h-6" />,
                      title: "Listwise Deletion",
                      description: "Remove entire rows with any missing values"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Pairwise Deletion",
                      description: "Use available data for each analysis, excluding missing pairs"
                    },
                    {
                      icon: <Filter className="w-6 h-6" />,
                      title: "Feature Deletion",
                      description: "Remove features with excessive missing values"
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
                content: "Imputation Methods"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Statistical Imputation",
                      description: "Fill with mean, median, or mode values"
                    },
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Forward/Backward Fill",
                      description: "Use previous or next values for time series data"
                    },
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "Model-Based Imputation",
                      description: "Predict missing values using other features"
                    }
                  ],
                  columns: 1
                }
              }
            ]
          }
        }
      ]
    },
    {
      type: 'codeBlock' as const,
      props: {
        language: "python",
        filename: "missing_data_handling.py",
        code: `import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Sample dataset with missing values
data = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 35, np.nan, 28],
    'income': [50000, 60000, 55000, np.nan, 70000, 65000, np.nan],
    'education': ['Bachelor', 'Master', np.nan, 'PhD', 'Bachelor', 'Master', 'High School']
})

# 1. Check missing data
print("Missing data summary:")
print(data.isnull().sum())
print(f"Missing data percentage:")
print((data.isnull().sum() / len(data)) * 100)

# 2. Simple imputation strategies
# Mean imputation for numerical features
mean_imputer = SimpleImputer(strategy='mean')
data['age_mean_imputed'] = mean_imputer.fit_transform(data[['age']])

# Median imputation (more robust to outliers)
median_imputer = SimpleImputer(strategy='median')
data['income_median_imputed'] = median_imputer.fit_transform(data[['income']])

# Mode imputation for categorical features
mode_imputer = SimpleImputer(strategy='most_frequent')
data['education_mode_imputed'] = mode_imputer.fit_transform(data[['education']])

# 3. Advanced imputation techniques
# KNN Imputation
numerical_features = ['age', 'income']
knn_imputer = KNNImputer(n_neighbors=3)
data[['age_knn', 'income_knn']] = knn_imputer.fit_transform(data[numerical_features])

# Iterative Imputation (MICE)
iterative_imputer = IterativeImputer(random_state=42)
data[['age_iterative', 'income_iterative']] = iterative_imputer.fit_transform(data[numerical_features])

print("\\nAfter imputation:")
print(data.head())`
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Feature Scaling and Normalization"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Different features often have different scales, which can bias machine learning algorithms toward features with larger values. Scaling ensures all features contribute equally to the model."
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Target className="w-6 h-6" />,
                title: "Min-Max Scaling",
                description: "Scale features to a fixed range, typically [0, 1]"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Standardization (Z-score)",
                description: "Transform features to have mean=0 and standard deviation=1"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Robust Scaling",
                description: "Use median and interquartile range, robust to outliers"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "Unit Vector Scaling",
                description: "Scale samples individually to have unit norm"
              }
            ],
            columns: 2
          }
        }
      ]
    },
    {
      type: 'codeBlock' as const,
      props: {
        language: "python",
        filename: "feature_scaling.py",
        code: `from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import pandas as pd
import numpy as np

# Sample dataset
data = pd.DataFrame({
    'age': [25, 30, 35, 45, 28, 50, 33],
    'income': [50000, 60000, 70000, 120000, 55000, 150000, 65000],
    'score': [85, 92, 78, 95, 88, 90, 87]
})

print("Original data:")
print(data.describe())

# 1. Min-Max Scaling (0 to 1)
min_max_scaler = MinMaxScaler()
data_minmax = pd.DataFrame(
    min_max_scaler.fit_transform(data),
    columns=[f'{col}_minmax' for col in data.columns]
)

# 2. Standardization (Z-score normalization)
standard_scaler = StandardScaler()
data_standard = pd.DataFrame(
    standard_scaler.fit_transform(data),
    columns=[f'{col}_standard' for col in data.columns]
)

# 3. Robust Scaling
robust_scaler = RobustScaler()
data_robust = pd.DataFrame(
    robust_scaler.fit_transform(data),
    columns=[f'{col}_robust' for col in data.columns]
)

# 4. Unit Vector Scaling (L2 normalization)
normalizer = Normalizer(norm='l2')
data_normalized = pd.DataFrame(
    normalizer.fit_transform(data),
    columns=[f'{col}_normalized' for col in data.columns]
)

# Combine all scaling methods for comparison
comparison = pd.concat([data, data_minmax, data_standard, data_robust, data_normalized], axis=1)
print("\\nScaled data comparison:")
print(comparison.round(3))`
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Categorical Data Encoding",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Machine learning algorithms typically work with numerical data, so categorical variables need to be encoded appropriately. The choice of encoding method depends on the nature of the categorical variable."
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Ordinal Encoding"
              },
              {
                type: 'paragraph' as const,
                content: "For categorical variables with inherent order (like education levels: High School < Bachelor < Master < PhD)."
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Label Encoding",
                      description: "Assign integers based on alphabetical or custom order"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Custom Mapping",
                      description: "Manually define the order and corresponding values"
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
                content: "Nominal Encoding"
              },
              {
                type: 'paragraph' as const,
                content: "For categorical variables without inherent order (like colors: red, blue, green)."
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "One-Hot Encoding",
                      description: "Create binary columns for each category"
                    },
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "Target Encoding",
                      description: "Replace categories with target variable statistics"
                    }
                  ],
                  columns: 1
                }
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Outlier Detection and Treatment"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Outliers can significantly impact model performance. Detecting and handling them appropriately is crucial for robust machine learning models."
        },
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Statistical Methods",
                date: "Step 1",
                description: "Use Z-score, IQR, or modified Z-score to identify outliers based on statistical properties"
              },
              {
                title: "Visualization",
                date: "Step 2",
                description: "Create box plots, scatter plots, and histograms to visually identify unusual data points"
              },
              {
                title: "Domain Knowledge",
                date: "Step 3",
                description: "Apply subject matter expertise to determine if outliers are errors or valid extreme values"
              },
              {
                title: "Treatment Strategy",
                date: "Step 4",
                description: "Remove, transform, or cap outliers based on their nature and impact on the model"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Best Practices",
        background: true
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Understand Your Data",
                description: "Always perform exploratory data analysis before preprocessing"
              },
              {
                icon: <CheckCircle className="w-6 h-6" />,
                title: "Document Everything",
                description: "Keep track of all preprocessing steps for reproducibility"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Avoid Data Leakage",
                description: "Apply preprocessing steps separately to training and test sets"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Iterative Process",
                description: "Continuously refine preprocessing based on model performance"
              }
            ],
            columns: 2
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "References and Further Learning"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Explore these resources to deepen your understanding of data preprocessing:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Essential Libraries"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Pandas",
                      description: "Comprehensive data manipulation and analysis library"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Scikit-learn Preprocessing",
                      description: "Robust preprocessing utilities and transformers"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "NumPy",
                      description: "Foundation for numerical computing and array operations"
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
                content: "Learning Resources"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Python for Data Analysis",
                      description: "Comprehensive book by Wes McKinney (Pandas creator)"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Kaggle Learn: Data Cleaning",
                      description: "Free hands-on course with practical examples"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Feature Engineering for Machine Learning",
                      description: "O'Reilly book on advanced preprocessing techniques"
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
  ],
  navigation: {
    colorScheme: 'yellow' as const
  }
};

export default function DataPreprocessingPage() {
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
        <TopicPageBuilder {...dataPreprocessingPageData} />
      </article>
    </div>
  );
}
