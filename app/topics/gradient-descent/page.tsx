import { TrendingDown, Target, Brain, BarChart3, Code, Zap, ArrowDown } from 'lucide-react';
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
          content: "Gradient descent is like hiking down a mountain in the fog - you can't see the bottom, but you can feel which direction slopes downward most steeply. The algorithm repeatedly takes steps in the direction of steepest decrease (negative gradient) until it reaches a minimum."
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
              type: 'list' as const,
              content: "Foundation of neural network training\nEnables automatic parameter optimization\nCore to understanding all ML optimization\nMathematical basis for learning algorithms"
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
          content: "The algorithm follows a simple iterative process:"
        },
        {
          type: 'list' as const,
          content: "Calculate the Gradient: Find the partial derivatives of the loss function with respect to each parameter\nDetermine Step Direction: Move in the opposite direction of the gradient (steepest descent)\nUpdate Parameters: Adjust parameters by a small amount (learning rate) in that direction\nRepeat: Continue until convergence or maximum iterations"
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
              type: 'paragraph' as const,
              content: "θ = θ - α∇J(θ)\n\nWhere:\n- θ = parameters (weights)\n- α = learning rate\n- ∇J(θ) = gradient of loss function"
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
            items: [
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
    }
  ],
  navigation: {
    colorScheme: 'green' as const
  }
};

export default function GradientDescentPage() {
  return <TopicPageBuilder {...gradientDescentData} />;
}
