import TopicPageBuilder from '../../components/TopicPageBuilder';
import { Shield, Zap, TrendingUp, Settings, CheckCircle, Search, Brain, Code, ExternalLink } from 'lucide-react';

export const metadata = {
  title: 'Regularization Techniques - ML Portfolio',
  description: 'Methods to prevent overfitting and improve model generalization in machine learning',
};

const regularizationData = {
    title: "Regularization Techniques",
    header: {
      category: "Fundamentals",
      difficulty: "Intermediate" as const,
      readTime: "12 min read",
      description: "Methods to prevent overfitting and improve model generalization by adding constraints or penalties to the learning process",
      relatedProjects: ["image-classifier", "custom-gpt-llm", "real-salary"],
      gradientFrom: "from-green-50 to-emerald-50",
      gradientTo: "dark:from-green-900/20 dark:to-emerald-900/20",
      borderColor: "border-green-200 dark:border-green-800"
    },
    tags: {
      items: ["Overfitting", "Generalization", "Dropout", "Batch Normalization"],
      colorScheme: "green" as const
    },
    blocks: [
      {
        type: 'section' as const,
        props: { title: 'What is Regularization?' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Regularization is a set of techniques used to prevent overfitting in machine learning models by adding constraints, penalties, or noise to the learning process. The goal is to improve model generalization - ensuring that the model performs well on unseen data, not just the training data. Regularization helps find the right balance between fitting the training data and maintaining simplicity.'
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'The Overfitting Problem' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Overfitting occurs when a model learns the training data too well, including its noise and specific patterns that don\'t generalize. This results in:'
          },
          {
            type: 'list' as const,
            props: {
              items: [
                'High accuracy on training data but poor performance on test data',
                'Complex models that capture noise rather than underlying patterns',
                'Inability to generalize to new, unseen examples',
                'High variance in model predictions'
              ]
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Classical Regularization Methods' },
        children: [
          {
            type: 'section' as const,
            props: { title: 'L1 and L2 Regularization' },
            children: [
              {
                type: 'paragraph' as const,
                content: 'L1 (Lasso) and L2 (Ridge) regularization add penalty terms to the loss function based on the magnitude of model parameters.'
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# L2 Regularization (Ridge)
def l2_regularization(model, lambda_reg=0.01):
    """
    Add L2 penalty to encourage smaller weights
    """
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param, 2) ** 2
    return lambda_reg * l2_loss

# L1 Regularization (Lasso)
def l1_regularization(model, lambda_reg=0.01):
    """
    Add L1 penalty to encourage sparsity
    """
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.norm(param, 1)
    return lambda_reg * l1_loss

# Elastic Net (L1 + L2)
def elastic_net_regularization(model, l1_lambda=0.01, l2_lambda=0.01):
    """
    Combine L1 and L2 regularization
    """
    l1_loss = l1_regularization(model, l1_lambda)
    l2_loss = l2_regularization(model, l2_lambda)
    return l1_loss + l2_loss

# Example: Linear model with regularization
class RegularizedLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)
    
    def loss_with_regularization(self, y_pred, y_true, l1_lambda=0.01, l2_lambda=0.01):
        # Data loss
        data_loss = F.mse_loss(y_pred, y_true)
        
        # Regularization loss
        reg_loss = elastic_net_regularization(self, l1_lambda, l2_lambda)
        
        return data_loss + reg_loss`
                }
              }
            ]
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Neural Network Regularization' },
        children: [
          {
            type: 'section' as const,
            props: { title: 'Dropout' },
            children: [
              {
                type: 'paragraph' as const,
                content: 'Dropout randomly sets a fraction of input units to zero during training, preventing co-adaptation of neurons.'
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  code: `# Dropout implementation
class DropoutLayer(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        if self.training:
            # Create random mask
            mask = torch.rand(x.shape) > self.dropout_rate
            # Scale by (1 - dropout_rate) to maintain expected value
            return x * mask.float() / (1 - self.dropout_rate)
        else:
            return x

# Using PyTorch's built-in dropout
class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Different dropout variants
class DropoutVariants(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        
        # Standard dropout
        self.dropout = nn.Dropout(0.5)
        
        # Dropout2d for convolutional layers
        self.dropout2d = nn.Dropout2d(0.5)
        
        # Alpha dropout for SELU activation
        self.alpha_dropout = nn.AlphaDropout(0.5)
        
    def forward(self, x):
        x = self.fc(x)
        return self.dropout(x)`
                }
              }
            ]
          },
          {
            type: 'section' as const,
            props: { title: 'Batch Normalization' },
            children: [
              {
                type: 'paragraph' as const,
                content: 'Batch normalization normalizes inputs to each layer, reducing internal covariate shift and acting as a regularizer.'
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  code: `# Batch Normalization implementation
class BatchNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Use running statistics during inference
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        return self.gamma * x_normalized + self.beta

# Network with batch normalization
class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x`
                }
              }
            ]
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Advanced Regularization Techniques' },
        children: [
          {
            type: 'features' as const,
            props: {
              features: [
                {
                  title: 'Early Stopping',
                  description: 'Monitor validation loss during training and stop when it starts increasing'
                },
                {
                  title: 'Data Augmentation',
                  description: 'Artificially increase dataset size by applying transformations to training data'
                },
                {
                  title: 'Weight Decay',
                  description: 'Built-in L2 regularization in optimizers that decays weights during training'
                },
                {
                  title: 'Label Smoothing',
                  description: 'Soften one-hot labels to prevent overconfident predictions'
                },
                {
                  title: 'Spectral Normalization',
                  description: 'Normalize weight matrices by their spectral norm to control Lipschitz constant'
                }
              ]
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Regularization in Practice' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Practical implementation of regularization techniques:'
          },
          {
            type: 'codeBlock' as const,
            props: {
              language: 'python',
              code: `# Complete training loop with regularization
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_regularization(model, train_loader, val_loader, epochs=100):
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            
            # Loss with additional regularization
            loss = F.cross_entropy(output, target)
            
            # Add custom regularization if needed
            if hasattr(model, 'custom_regularization'):
                loss += model.custom_regularization()
            
            loss.backward()
            
            # Gradient clipping (another form of regularization)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += F.cross_entropy(output, target).item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

# Label smoothing implementation
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Convert to one-hot
        one_hot = torch.zeros_like(pred)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        smooth_target = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        # Calculate loss
        log_pred = F.log_softmax(pred, dim=1)
        return -torch.sum(smooth_target * log_pred, dim=1).mean()`
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Choosing Regularization Techniques' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Guidelines for selecting appropriate regularization methods:'
          },
          {
            type: 'list' as const,
            props: {
              items: [
                'Start with simple techniques like L2 regularization and dropout',
                'Use batch normalization for deep networks to stabilize training',
                'Apply data augmentation when you have limited training data',
                'Monitor validation metrics to detect overfitting early',
                'Combine multiple techniques for better results',
                'Tune regularization strength using validation data'
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
            content: "Comprehensive resources to deepen your understanding of regularization:"
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
                        title: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
                        description: "Srivastava et al. (2014) - Introduction of dropout regularization"
                      },
                      {
                        icon: <Search className="w-6 h-6" />,
                        title: "Batch Normalization: Accelerating Deep Network Training",
                        description: "Ioffe & Szegedy (2015) - Batch normalization technique"
                      },
                      {
                        icon: <Search className="w-6 h-6" />,
                        title: "Early Stopping - But When?",
                        description: "Prechelt (1998) - Early stopping strategies"
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
                        icon: <Brain className="w-6 h-6" />,
                        title: "Deep Learning Book Chapter 7",
                        description: "Comprehensive regularization coverage"
                      },
                      {
                        icon: <Code className="w-6 h-6" />,
                        title: "PyTorch Regularization Techniques",
                        description: "Practical implementations"
                      },
                      {
                        icon: <ExternalLink className="w-6 h-6" />,
                        title: "Regularization in Practice Guide",
                        description: "Applied techniques and tips"
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

  export default function RegularizationPage() {
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
          <TopicPageBuilder {...regularizationData} />
        </article>
      </div>
    );
  }