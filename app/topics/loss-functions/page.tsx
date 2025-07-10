import TopicPageBuilder from '../../components/TopicPageBuilder';
import { Calculator, Target, TrendingDown, BarChart3, AlertTriangle, Search, BookOpen, Code, ExternalLink } from 'lucide-react';

export const metadata = {
  title: 'Loss Functions - ML Portfolio',
  description: 'Mathematical functions that measure the difference between predicted and actual values in machine learning models',
};

const LossFunctionData = {
  title: "Loss Functions",
  header: {
    category: "Fundamentals",
    difficulty: "Beginner" as const,
      readTime: "10 min read",
      description: "Mathematical functions that measure the difference between predicted and actual values, providing the foundation for training machine learning models",
      relatedProjects: ["image-classifier", "custom-gpt-llm", "dqn-flappy-bird", "real-salary"],
      gradientFrom: "from-red-50 to-orange-50",
      gradientTo: "dark:from-red-900/20 dark:to-orange-900/20",
      borderColor: "border-red-200 dark:border-red-800"
    },
    tags: {
      items: ["Training", "Optimization", "Mathematics", "Evaluation"],
      colorScheme: "red" as const
    },
    blocks: [
      {
        type: 'section' as const,
        props: { title: 'What are Loss Functions?' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Loss functions are mathematical functions that quantify the difference between predicted and actual values in machine learning models. They serve as the objective function that optimization algorithms like gradient descent minimize during training. The choice of loss function depends on the type of problem (regression, classification, etc.) and can significantly impact model performance.'
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Key Properties' },
        children: [
          {
            type: 'features' as const,
            props: {
              features: [
                {
                  title: 'Differentiability',
                  description: 'Loss functions must be differentiable to enable gradient-based optimization algorithms'
                },
                {
                  title: 'Convexity',
                  description: 'Convex loss functions guarantee global minima, making optimization more reliable'
                },
                {
                  title: 'Robustness',
                  description: 'Some loss functions are more robust to outliers and noisy data than others'
                }
              ]
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Common Loss Functions' },
        children: [
          {
            type: 'section' as const,
            props: { title: 'Regression Loss Functions' },
            children: [
              {
                type: 'paragraph' as const,
                content: 'For regression tasks, loss functions measure the difference between predicted and actual continuous values.'
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  code: `import torch
import torch.nn.functional as F
import numpy as np

# Mean Squared Error (MSE)
def mse_loss(y_pred, y_true):
    """
    L2 loss - sensitive to outliers
    """
    return torch.mean((y_pred - y_true) ** 2)

# Mean Absolute Error (MAE)
def mae_loss(y_pred, y_true):
    """
    L1 loss - more robust to outliers
    """
    return torch.mean(torch.abs(y_pred - y_true))

# Huber Loss (Smooth L1)
def huber_loss(y_pred, y_true, delta=1.0):
    """
    Combines MSE and MAE - robust to outliers
    """
    residual = torch.abs(y_pred - y_true)
    condition = residual < delta
    squared_loss = 0.5 * (residual ** 2)
    linear_loss = delta * residual - 0.5 * (delta ** 2)
    return torch.mean(torch.where(condition, squared_loss, linear_loss))

# Example usage
y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_pred = torch.tensor([1.1, 2.2, 2.8, 4.5])

print(f"MSE Loss: {mse_loss(y_pred, y_true):.4f}")
print(f"MAE Loss: {mae_loss(y_pred, y_true):.4f}")
print(f"Huber Loss: {huber_loss(y_pred, y_true):.4f}")`
                }
              }
            ]
          },
          {
            type: 'section' as const,
            props: { title: 'Classification Loss Functions' },
            children: [
              {
                type: 'paragraph' as const,
                content: 'For classification tasks, loss functions measure the difference between predicted class probabilities and actual class labels.'
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  code: `# Binary Cross-Entropy
def binary_cross_entropy(y_pred, y_true):
    """
    For binary classification
    """
    epsilon = 1e-15  # Small value to prevent log(0)
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

# Categorical Cross-Entropy
def categorical_cross_entropy(y_pred, y_true):
    """
    For multi-class classification
    """
    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    return -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=1))

# Focal Loss
def focal_loss(y_pred, y_true, alpha=1.0, gamma=2.0):
    """
    Addresses class imbalance by focusing on hard examples
    """
    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    # Calculate cross entropy
    ce_loss = -y_true * torch.log(y_pred)
    
    # Calculate focal term
    p_t = torch.where(y_true == 1, y_pred, 1 - y_pred)
    focal_term = (1 - p_t) ** gamma
    
    # Apply alpha weighting
    alpha_t = torch.where(y_true == 1, alpha, 1 - alpha)
    
    return torch.mean(alpha_t * focal_term * ce_loss)

# Example usage
# Binary classification
y_true_binary = torch.tensor([1.0, 0.0, 1.0, 0.0])
y_pred_binary = torch.tensor([0.8, 0.2, 0.9, 0.1])

print(f"Binary Cross-Entropy: {binary_cross_entropy(y_pred_binary, y_true_binary):.4f}")`
                }
              }
            ]
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Advanced Loss Functions' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Modern deep learning applications often use specialized loss functions designed for specific tasks:'
          },
          {
            type: 'list' as const,
            props: {
              items: [
                'Contrastive Loss: For learning embeddings that bring similar samples closer and push dissimilar ones apart',
                'Triplet Loss: For learning embeddings using anchor, positive, and negative samples',
                'Dice Loss: For image segmentation tasks, especially with imbalanced classes',
                'GAN Loss: For training generative adversarial networks with minimax objectives',
                'Perceptual Loss: For image generation tasks using pre-trained networks to compare high-level features'
              ]
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Choosing the Right Loss Function' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'The choice of loss function depends on several factors:'
          },
          {
            type: 'features' as const,
            props: {
              features: [
                {
                  title: 'Problem Type',
                  description: 'Regression vs classification vs structured prediction tasks require different loss functions'
                },
                {
                  title: 'Data Distribution',
                  description: 'Outliers, class imbalance, and noise levels affect which loss function performs best'
                },
                {
                  title: 'Optimization Properties',
                  description: 'Some loss functions are easier to optimize and converge faster than others'
                }
              ]
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Implementation Tips' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Best practices for implementing and using loss functions:'
          },
          {
            type: 'codeBlock' as const,
            props: {
              language: 'python',
              code: `# Custom loss function class
class CustomLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, y_pred, y_true):
        # Combine multiple loss terms
        mse_term = F.mse_loss(y_pred, y_true)
        mae_term = F.l1_loss(y_pred, y_true)
        
        # Weighted combination
        return self.alpha * mse_term + (1 - self.alpha) * mae_term

# Numerical stability considerations
def stable_cross_entropy(logits, targets):
    """
    Numerically stable cross-entropy using log-sum-exp trick
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return F.nll_loss(log_probs, targets)

# Loss function with regularization
def loss_with_regularization(model, y_pred, y_true, lambda_reg=0.01):
    """
    Add L2 regularization to the loss
    """
    data_loss = F.mse_loss(y_pred, y_true)
    
    # L2 regularization
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.norm(param, 2)
    
    return data_loss + lambda_reg * reg_loss`
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
            content: "Comprehensive resources to deepen your understanding of loss functions:"
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
                        title: "Focal Loss for Dense Object Detection",
                        description: "Lin et al. (2017) - Focal loss for addressing class imbalance"
                      },
                      {
                        icon: <Search className="w-6 h-6" />,
                        title: "Label Smoothing Regularization",
                        description: "Szegedy et al. (2016) - Label smoothing technique"
                      },
                      {
                        icon: <Search className="w-6 h-6" />,
                        title: "When Does Label Smoothing Help?",
                        description: "Müller et al. (2019) - Analysis of label smoothing"
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
                        title: "PyTorch Loss Functions Documentation",
                        description: "Comprehensive loss function library and usage"
                      },
                      {
                        icon: <ExternalLink className="w-6 h-6" />,
                        title: "Loss Function Visualization Tool",
                        description: "Interactive loss landscape exploration"
                      },
                      {
                        icon: <Code className="w-6 h-6" />,
                        title: "Deep Learning Book Chapter 5",
                        description: "Theoretical foundations of loss functions"
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

  export default function LossFunctionPage() {
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
          <TopicPageBuilder {...LossFunctionData} />
        </article>
      </div>
    );
}
