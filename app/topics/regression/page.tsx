import { TrendingUp, BarChart3, Target, LineChart, Brain, Calculator, Search, BookOpen, ExternalLink } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Regression Analysis - ML Portfolio',
  description: 'Statistical method for modeling relationships between variables and making predictions',
};

const regressionData = {
  title: "Regression Analysis",
  header: {
    category: "Fundamentals",
    difficulty: "Beginner" as const,
    readTime: "7 min read",
    description: "Statistical method for modeling relationships between variables and making predictions, forming the foundation of many machine learning algorithms",
    relatedProjects: ["Real Salary Prediction"],
    gradientFrom: "from-blue-50 to-green-50",
    gradientTo: "dark:from-blue-900/20 dark:to-green-900/20",
    borderColor: "border-blue-200 dark:border-blue-800"
  },
  tags: {
    items: ['Supervised Learning', 'Prediction', 'Statistics', 'Linear Models'],
    colorScheme: 'blue' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What is Regression?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Regression analysis is a statistical method that examines the relationship between a dependent variable (target) and one or more independent variables (features). It's used to understand how the value of the dependent variable changes when any one of the independent variables is varied."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Core Purpose",
            icon: <Target className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "Predict continuous numerical outcomes",
                  "Understand relationships between variables",
                  "Quantify the impact of different factors",
                  "Make data-driven decisions"
                ]
              }
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Types of Regression"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <LineChart className="w-6 h-6" />,
                title: "Linear Regression",
                description: "Models linear relationship between variables. Simple yet powerful for many real-world problems.",
                color: "blue"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Polynomial Regression",
                description: "Captures non-linear relationships by using polynomial features of the input variables.",
                color: "green"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Multiple Regression",
                description: "Uses multiple independent variables to predict the dependent variable.",
                color: "purple"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Linear Regression Implementation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's how to implement linear regression from scratch and using scikit-learn:"
        },
        {
          type: 'codeBlock' as const,
          content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.squeeze() + 1 + 0.1 * np.random.randn(100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1: From Scratch
class LinearRegressionScratch:
    def __init__(self):
        self.slope = None
        self.intercept = None
    
    def fit(self, X, y):
        # Calculate slope and intercept using normal equation
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        numerator = np.sum((X.squeeze() - X_mean) * (y - y_mean))
        denominator = np.sum((X.squeeze() - X_mean) ** 2)
        
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * X_mean
    
    def predict(self, X):
        return self.slope * X.squeeze() + self.intercept

# Method 2: Using scikit-learn
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

# Custom implementation
custom_model = LinearRegressionScratch()
custom_model.fit(X_train, y_train)

# Make predictions
y_pred_sklearn = sklearn_model.predict(X_test)
y_pred_custom = custom_model.predict(X_test)

print(f"Scikit-learn - MSE: {mean_squared_error(y_test, y_pred_sklearn):.4f}")
print(f"Custom - MSE: {mean_squared_error(y_test, y_pred_custom):.4f}")`,
          props: {
            language: 'python',
            title: 'Linear Regression Implementation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Multiple Linear Regression"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "When dealing with multiple features, we extend linear regression to multiple dimensions:"
        },
        {
          type: 'codeBlock' as const,
          content: `import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create sample dataset with multiple features
np.random.seed(42)
n_samples = 1000

# Generate correlated features
age = np.random.normal(35, 10, n_samples)
experience = age - 22 + np.random.normal(0, 2, n_samples)
education = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.3, 0.2])

# Create target variable (salary) with realistic relationships
salary = (
    30000 +  # base salary
    1000 * experience +  # experience factor
    5000 * education +   # education factor
    500 * age +          # age factor
    np.random.normal(0, 5000, n_samples)  # noise
)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'experience': experience,
    'education': education,
    'salary': salary
})

# Prepare features and target
X = df[['age', 'experience', 'education']]
y = df['salary']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_names = ['age', 'experience', 'education']
coefficients = model.coef_

print("\\nFeature Importance:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.2f}")`,
          props: {
            language: 'python',
            title: 'Multiple Linear Regression'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Polynomial Regression"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "For non-linear relationships, polynomial regression can capture more complex patterns:"
        },
        {
          type: 'codeBlock' as const,
          content: `from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X.squeeze()**3 - X.squeeze()**2 + 2 * X.squeeze() + np.random.normal(0, 1, 100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial regression models with different degrees
degrees = [1, 2, 3, 4, 5]
models = {}

for degree in degrees:
    # Create pipeline with polynomial features and linear regression
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    poly_model.fit(X_train, y_train)
    models[degree] = poly_model

# Evaluate models
print("Polynomial Regression Results:")
print("-" * 40)

for degree, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Degree {degree}: MSE = {mse:.4f}, R² = {r2:.4f}")

# Visualize results
plt.figure(figsize=(15, 10))

for i, (degree, model) in enumerate(models.items(), 1):
    plt.subplot(2, 3, i)
    
    # Generate smooth curve for plotting
    X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
    plt.plot(X_plot, y_plot, 'r-', label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()`,
          props: {
            language: 'python',
            title: 'Polynomial Regression'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Model Evaluation and Assumptions"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Key Metrics",
            icon: <Calculator className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "R² Score: Proportion of variance explained by the model",
                  "Mean Squared Error (MSE): Average of the squares of errors",
                  "Mean Absolute Error (MAE): Average of absolute differences",
                  "Root Mean Squared Error (RMSE): Square root of MSE"
                ]
              },
            }
          ]
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'warning' as const,
            title: "Linear Regression Assumptions",
            icon: <Brain className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "Linearity: Relationship between variables is linear",
                  "Independence: Observations are independent",
                  "Homoscedasticity: Constant variance of residuals",
                  "Normality: Residuals are normally distributed"
                ]
              },
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Advanced Regression Techniques"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "When basic linear regression isn't sufficient, several advanced techniques can help:"
        },
        {
          type: 'codeBlock' as const,
          content: `from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

# Generate data with many features (some irrelevant)
np.random.seed(42)
n_samples, n_features = 100, 20
X = np.random.randn(n_samples, n_features)

# Only first 5 features are relevant
true_coef = np.zeros(n_features)
true_coef[:5] = [1.5, -2.0, 0.5, -1.0, 3.0]

y = X @ true_coef + 0.1 * np.random.randn(n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare different regularization techniques
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}

for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                               scoring='neg_mean_squared_error')
    
    # Test score
    test_score = model.score(X_test, y_test)
    
    results[name] = {
        'CV Score': -cv_scores.mean(),
        'Test R²': test_score,
        'Non-zero coefs': np.sum(np.abs(model.coef_) > 1e-4)
    }

# Display results
print("Regularization Comparison:")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()`,
          props: {
            language: 'python',
            title: 'Regularization Techniques'
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
          content: "Comprehensive resources to deepen your understanding of regression analysis:"
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
                      title: "Regression Shrinkage and Selection via the Lasso",
                      description: "Tibshirani (1996) - Introduction of LASSO regularization"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Regularization and Variable Selection via the Elastic Net",
                      description: "Zou & Hastie (2005) - Elastic net regularization method"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
                      description: "Hoerl & Kennard (1970) - Foundational ridge regression paper"
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
                      title: "Scikit-learn Linear Models",
                      description: "Comprehensive guide to regression models in scikit-learn"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "An Introduction to Statistical Learning",
                      description: "Free textbook with excellent regression chapters"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Linear Regression in Python",
                      description: "Practical tutorial for implementing regression in Python"
                    },
                    {
                      icon: <Calculator className="w-6 h-6" />,
                      title: "Understanding Bias-Variance Tradeoff",
                      description: "Explanation of bias-variance tradeoff in regression models"
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
    colorScheme: 'blue' as const
  }
};

export default function RegressionPage() {
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
        <TopicPageBuilder {...regressionData} />
      </article>
    </div>
  );
}