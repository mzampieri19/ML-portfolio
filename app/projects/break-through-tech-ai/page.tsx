import { BookOpen, Award, Target, Brain, BarChart3, Code, Database, Cpu } from 'lucide-react';
import PageBuilder from '../../components/PageBuilder';

export const metadata = {
  title: 'Break Through Tech AI 2024 - ML Portfolio',
  description: 'A collection of various projects created in the BTTAI program 2024 cohort',
};

const bttaiPageData = {
  title: "Break Through Tech AI 2024",
  header: {
    date: "Summer 2024",
    readTime: "8 min read",
    description: "Comprehensive collection of machine learning projects from the BTTAI 2024 cohort",
    githubUrl: "https://github.com/mzampieri19/Break-Through-Tech-AI-2024",
    gradientFrom: "from-indigo-50 to-purple-50",
    gradientTo: "dark:from-indigo-900/20 dark:to-purple-900/20",
    borderColor: "border-indigo-200 dark:border-indigo-800",
    collaborators: "Program Cohort"
  },
  tags: {
    items: ['Introduction', 'Data Science', 'CNN', 'Regression', 'Random Forest'],
    colorScheme: 'indigo' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "Program Overview",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The Break Through Tech AI (BTTAI) 2024 cohort was an introductory course to machine learning and AI at MIT through eCornell, completed in Spring 2024. This comprehensive program provided hands-on experience with fundamental ML concepts, data preparation techniques, and various modeling approaches."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Program Highlights",
            icon: <Award className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                "Introductory course to ML and AI at MIT through eCornell",
                "Hands-on labs and assignments with real datasets",
                "Comprehensive coverage from data preparation to modeling",
                "Multiple ML algorithms and techniques explored",
                "Practical Python programming and Jupyter notebooks" 
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
        title: "Technical Skills Developed"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Code className="w-6 h-6" />,
                title: "Python Programming",
                description: "Proficiency in pandas, numpy, matplotlib, and seaborn"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Machine Learning",
                description: "KNN, Logistic Regression, Random Forest, and Decision Trees"
              },
              {
                icon: <Cpu className="w-6 h-6" />,
                title: "Deep Learning",
                description: "CNN implementation for image classification (MNIST)"
              },
              {
                icon: <Database className="w-6 h-6" />,
                title: "Data Preprocessing",
                description: "Missing data handling, feature engineering, categorical encoding"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Model Evaluation",
                description: "Cross-validation, grid search, and performance metrics"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Statistical Analysis",
                description: "Correlation analysis, descriptive statistics, data visualization"
              }
            ],
            columns: 3
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Lab Projects Collection"
      },
      children: [
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Lab 1: Business Understanding",
                date: "Foundation",
                description: "Python Tools and Machine Learning Strategy - understanding Python tools and formulating business plans for ML models"
              },
              {
                title: "Lab 2: Data Preparation",
                date: "Preprocessing",
                description: "Deep dive into data preprocessing techniques essential for ML, including handling missing values and feature engineering"
              },
              {
                title: "Lab 3: K-Nearest Neighbors",
                date: "First Modeling",
                description: "Implementation and analysis of KNN algorithm for classification tasks, including optimization and evaluation"
              },
              {
                title: "Lab 4: Logistic Regression",
                date: "Classification",
                description: "Training and analysis of logistic regression models with comprehensive performance evaluation"
              },
              {
                title: "Lab 5: Evaluation & Deployment",
                date: "Optimization",
                description: "Hyperparameter optimization with grid search techniques to maximize model accuracy"
              },
              {
                title: "Lab 6: Model Comparison",
                date: "Analysis",
                description: "Comprehensive comparison of various ML algorithms to analyze performance differences"
              },
              {
                title: "Lab 7: Convolutional Neural Networks",
                date: "Deep Learning",
                description: "Implementation of CNN for predicting handwritten numbers using neural networks"
              },
              {
                title: "Lab 8: World Happiness Project",
                date: "End-to-End",
                description: "Comprehensive ML project using World Happiness Report data with logistic regression and grid search"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'twoColumn' as const,
      props: {
        ratio: '1:1' as const,
        left: [
          {
            type: 'heading' as const,
            props: { level: 3 },
            content: "Data Sources"
          },
          {
            type: 'list' as const,
            props: {
              items: [
                "Census Data: Demographic analysis and classification tasks",
                "AirBNB Listings: Pricing and recommendation analysis",
                "World Happiness Report: Comprehensive happiness analysis",
                "MNIST Digit Dataset: Standard dataset for CNN implementation",
                "Book Reviews: Text analysis and sentiment classification",
                "Cell2Cell Telecom: Customer churn prediction dataset"
              ]
            }
          }
        ],
        right: [
          {
            type: 'heading' as const,
            props: { level: 3 },
            content: "Core Libraries & Tools"
          },
          {
            type: 'highlight' as const,
            props: {
              variant: 'success' as const,
              title: "Technical Stack"
            },
            children: [
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Data Analysis: pandas, numpy",
                    "Visualization: matplotlib, seaborn",
                    "Machine Learning: scikit-learn",
                    "Deep Learning: TensorFlow/Keras",
                    "Development: Jupyter notebooks"
                  ]
                }
              }
            ]
          }
        ]
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Assignment Highlights",
        background: true
      },
      children: [
        {
          type: 'heading' as const,
          props: { level: 3 },
          content: "Advanced Techniques Covered"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <BookOpen className="w-6 h-6" />,
                title: "Text Analysis & NLP",
                description: "Book review sentiment analysis using logistic regression and feature extraction from text data"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Neural Network Analysis",
                description: "Advanced text processing and sentiment classification with deep learning approaches"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Decision Tree Optimization",
                description: "Hyperparameter tuning for decision trees and model selection with performance comparison"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Ensemble Learning",
                description: "Random Forest implementation, GBDT techniques, and feature importance analysis"
              }
            ],
            columns: 2
          }
        },
        {
          type: 'heading' as const,
          props: { level: 3 },
          content: "Real-World Applications"
        },
        {
          type: 'list' as const,
          props: {
            items: [
              "Customer Churn Prediction: Using telecom data to predict customer retention",
              "Income Classification: Census data analysis for demographic insights",
              "Image Recognition: Handwritten digit classification with CNNs",
              "Happiness Analysis: World Happiness Report data for policy insights",
              "Sentiment Analysis: Amazon book review classification"
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Program Structure & Approach"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The BTTAI program utilized a progressive learning approach that built comprehensive expertise step by step:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <BookOpen className="w-6 h-6" />,
                title: "Sequential Labs",
                description: "Each lab built upon previous knowledge for comprehensive understanding"
              },
              {
                icon: <Code className="w-6 h-6" />,
                title: "Hands-on Implementation",
                description: "Practical coding exercises in Jupyter notebooks with real datasets"
              },
              {
                icon: <Award className="w-6 h-6" />,
                title: "Graded Assessments",
                description: "Automatic grading and self-check systems for immediate feedback"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Comprehensive Coverage",
                description: "From basic Python programming to advanced neural network implementation"
              }
            ],
            columns: 4
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Key Achievements",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "This comprehensive program provided a solid foundation for entry into the machine learning and AI field:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Target className="w-6 h-6" />,
                title: "Solid Foundation",
                description: "Strong base in ML fundamentals and Python programming"
              },
              {
                icon: <Database className="w-6 h-6" />,
                title: "Practical Experience",
                description: "Real-world dataset analysis and model building experience"
              },
              {
                icon: <Award className="w-6 h-6" />,
                title: "Academic Rigor",
                description: "MIT-level coursework through eCornell platform"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Industry Readiness",
                description: "Preparation for entry-level ML/AI positions with comprehensive portfolio"
              }
            ],
            columns: 2
          }
        },
        {
          type: 'paragraph' as const,
          content: "This collection represents the culmination of intensive learning and hands-on experience in the field of artificial intelligence and machine learning, providing a comprehensive foundation for future advanced studies and professional development."
        }
      ]
    }
  ],
  navigation: {
    colorScheme: 'indigo' as const
  }
};

export default function BreakThroughTechAIPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">
                Projects
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <PageBuilder {...bttaiPageData} />
      </article>
    </div>
  );
}
