import { TrendingDown, Target, Brain, BarChart3, Code, Zap, ArrowDown, Search, ExternalLink, BookOpen } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Gradient Descent - ML Portfolio',
  description: 'Fundamental optimization algorithm used to minimize loss functions in machine learning',
};

const gradientDescentData = {
  title: "Gradient Descent",
  header: {
    category: "Fundamentals",
    difficulty: "Beginner" as const,
    readTime: "6 min read",
    description: "The fundamental optimization algorithm that powers machine learning, helping models learn by iteratively finding the minimum of a loss function",
    relatedProjects: ["Image Classifier", "Custom GPT LLM", "DQN Flappy Bird"],
    gradientFrom: "from-green-50 to-blue-50",
    gradientTo: "dark:from-green-900/20 dark:to-blue-900/20",
    borderColor: "border-green-200 dark:border-green-800"
  },
  tags: {
    items: ['Optimization', 'Mathematics', 'Training', 'Fundamentals'],
    colorScheme: 'green' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What is Gradient Descent?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Gradient descent is like hiking down a mountain in the fog, you can't see the bottom, but you can feel which direction slopes downward most steeply. The algorithm repeatedly takes steps in the direction of steepest decrease (negative gradient) until it reaches a minimum."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Why It Matters",
            icon: <Brain className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Foundation of neural network training, that enables automatic parameter optimization."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "How Gradient Descent Works"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The algorithm follows a simple iterative process: it first calculates the gradient by finding the partial derivatives of the loss function with respect to each parameter. Then, it determines the step direction by moving in the opposite direction of the gradient, which is the direction of steepest descent. Next, it updates the parameters by a small amount, known as the learning rate, in that direction. This process is repeated until the algorithm converges or reaches the maximum number of iterations."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Mathematical Formula",
            icon: <Target className="w-6 h-6" />
          },
          children: [
            {
              type: 'math' as const,
              props: {block: true},
              content: "θ = θ - α∇J(θ)\n\n",
            },
            {
              type: 'paragraph' as const,
              content: "Where:\n θ = parameters (weights)\n α = learning rate\n ∇J(θ) = gradient of loss function"
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Implementation Example"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's a basic implementation of gradient descent for linear regression:"
        },
        {
          type: 'codeBlock' as const,
          content: `import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    Basic gradient descent for linear regression
    """
    m = len(y)  # number of training examples
    
    # Initialize parameters
    theta_0 = 0  # bias
    theta_1 = 0  # slope
    
    # Cost function history
    cost_history = []
    
    for epoch in range(epochs):
        # Forward pass: predictions
        y_pred = theta_0 + theta_1 * X
        
        # Calculate cost (Mean Squared Error)
        cost = (1/(2*m)) * np.sum((y_pred - y)**2)
        cost_history.append(cost)
        
        # Calculate gradients
        dJ_dtheta_0 = (1/m) * np.sum(y_pred - y)
        dJ_dtheta_1 = (1/m) * np.sum((y_pred - y) * X)
        
        # Update parameters
        theta_0 = theta_0 - learning_rate * dJ_dtheta_0
        theta_1 = theta_1 - learning_rate * dJ_dtheta_1
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return theta_0, theta_1, cost_history

# Example usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 7, 11, 15, 19])  # y = 4x - 1

theta_0, theta_1, costs = gradient_descent(X, y, learning_rate=0.01, epochs=1000)
print(f"Final parameters: theta_0={theta_0:.2f}, theta_1={theta_1:.2f}")`,
          props: {
            language: 'python',
            title: 'Basic Gradient Descent Implementation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Types of Gradient Descent"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Batch Gradient Descent",
                description: "Uses entire dataset for each update. Stable convergence but slow for large datasets.",
                color: "blue"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "Stochastic Gradient Descent (SGD)",
                description: "Uses one sample at a time. Faster updates, more noise in convergence.",
                color: "yellow"
              },
              {
                icon: <TrendingDown className="w-6 h-6" />,
                title: "Mini-batch Gradient Descent",
                description: "Uses small batches of data. Balance between stability and speed.",
                color: "green"
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
          content: "Comprehensive resources to deepen your understanding of gradient descent optimization:"
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
                      title: "An Overview of Gradient Descent Optimization Algorithms",
                      description: "Ruder (2016) - Comprehensive survey of gradient descent variants"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Adam: A Method for Stochastic Optimization",
                      description: "Kingma & Ba (2014) - The popular Adam optimizer"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "On the Convergence of Adam and Beyond",
                      description: "Reddi et al. (2018) - AdamW and optimizer improvements"
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
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Distill.pub Gradient Descent Visualization",
                      description: "Interactive explanations of optimization landscapes"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "PyTorch Optimization Tutorial",
                      description: "Practical optimizer usage and implementation guide"
                    },
                    {
                      icon: <BookOpen className="w-6 h-6" />,
                      title: "CS231n Optimization Notes",
                      description: "Stanford course materials on optimization"
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
    colorScheme: 'green' as const
  }
};

export default function GradientDescentPage() {
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
          <TopicPageBuilder {...gradientDescentData} />
        </article>
      </div>
    );
}
