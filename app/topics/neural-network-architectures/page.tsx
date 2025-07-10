import { Brain, Layers, Cpu, Target, TrendingUp, Zap, Code, Settings, Eye, ExternalLink, GitBranch, Network } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Neural Network Architectures - ML Portfolio Topics',
  description: 'Learn about different neural network architectures and their applications in deep learning',
};

const neuralNetworkArchitecturesPageData = {
  title: "Neural Network Architectures",
  header: {
    date: "Deep Learning",
    readTime: "9 min read",
    description: "Different structures and designs of neural networks optimized for various machine learning tasks",
    gradientFrom: "from-blue-50 to-indigo-50",
    gradientTo: "dark:from-blue-900/20 dark:to-indigo-900/20",
    borderColor: "border-blue-200 dark:border-blue-800",
    difficulty: "Intermediate" as const,
    category: "Deep Learning",
    relatedProjects: ["dqn-flappy-bird", "custom-diffusion-model", "custom-gpt-llm", "image-classifier"]
  },
  tags: {
    items: ['Deep Learning', 'Architecture Design', 'Neural Networks', 'CNN', 'RNN', 'Transformers'],
    colorScheme: 'blue' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What are Neural Network Architectures?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Neural network architectures define how neurons are organized and connected in artificial neural networks. Different architectures are optimized for different types of data and tasks, from simple feedforward networks to complex transformer models."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Architecture Importance",
            icon: <Brain className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "The choice of architecture can dramatically impact model performance, training efficiency, and the types of patterns the network can learn."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Feedforward Neural Networks"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The simplest type of neural network where connections between nodes do not form cycles. Information flows in one direction from input to output through hidden layers."
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Perceptron"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Single Layer",
                      description: "Simplest neural network with no hidden layers"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "Linear Classification",
                      description: "Can only solve linearly separable problems"
                    },
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "Binary Output",
                      description: "Originally designed for binary classification tasks"
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
                content: "Multi-Layer Perceptron (MLP)"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Layers className="w-6 h-6" />,
                      title: "Hidden Layers",
                      description: "One or more hidden layers enable non-linear learning"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Universal Approximator",
                      description: "Can approximate any continuous function"
                    },
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Backpropagation",
                      description: "Trained using gradient descent and backpropagation"
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
        filename: "feedforward_network.py",
        code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(FeedforwardNN, self).__init__()
        
        # Create a list to store all layers
        layers = []
        
        # Input layer to first hidden layer
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Example usage
model = FeedforwardNN(
    input_size=784,      # 28x28 MNIST images flattened
    hidden_sizes=[512, 256, 128],  # Three hidden layers
    output_size=10,      # 10 classes for MNIST
    dropout_rate=0.3
)

# Print model architecture
print(model)

# Example forward pass
batch_size = 32
input_tensor = torch.randn(batch_size, 784)
output = model(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")`
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Convolutional Neural Networks (CNNs)",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "CNNs are specialized neural networks designed for processing grid-like data such as images. They use convolutional layers to detect spatial features and patterns."
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Convolutional Layers",
                description: "Apply filters to detect features like edges, shapes, and textures"
              },
              {
                icon: <Layers className="w-6 h-6" />,
                title: "Pooling Layers",
                description: "Reduce spatial dimensions while preserving important information"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Parameter Sharing",
                description: "Same filter weights used across different image regions"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Translation Invariance",
                description: "Can recognize features regardless of their position in the image"
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
        title: "Recurrent Neural Networks (RNNs)"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "RNNs are designed for sequential data by maintaining internal memory. They can process variable-length sequences and capture temporal dependencies."
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Vanilla RNN"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <GitBranch className="w-6 h-6" />,
                      title: "Hidden State",
                      description: "Maintains memory of previous inputs in the sequence"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "Simple Architecture",
                      description: "Basic recurrent connection with tanh activation"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Vanishing Gradients",
                      description: "Difficulty learning long-term dependencies"
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
                content: "LSTM & GRU"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Gating Mechanisms",
                      description: "Control information flow to prevent vanishing gradients"
                    },
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Long-term Memory",
                      description: "Better at capturing long-range dependencies"
                    },
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "Selective Updates",
                      description: "Gates decide what information to keep or forget"
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
        filename: "rnn_architectures.py",
        code: `import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Use last time step
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Example usage
input_size = 100
hidden_size = 128
output_size = 10
seq_length = 20
batch_size = 32

# Create models
vanilla_rnn = VanillaRNN(input_size, hidden_size, output_size)
lstm_model = LSTMModel(input_size, hidden_size, output_size)
gru_model = GRUModel(input_size, hidden_size, output_size)

# Test input
x = torch.randn(batch_size, seq_length, input_size)

print(f"Input shape: {x.shape}")
print(f"Vanilla RNN output: {vanilla_rnn(x).shape}")
print(f"LSTM output: {lstm_model(x).shape}")
print(f"GRU output: {gru_model(x).shape}")`
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Transformer Architecture",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Transformers revolutionized deep learning by replacing recurrent layers with self-attention mechanisms. They can process sequences in parallel and capture long-range dependencies effectively."
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Self-Attention",
                description: "Each position attends to all positions in the sequence"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "Parallel Processing",
                description: "Process entire sequences simultaneously, not sequentially"
              },
              {
                icon: <Layers className="w-6 h-6" />,
                title: "Multi-Head Attention",
                description: "Multiple attention mechanisms capture different relationships"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Positional Encoding",
                description: "Inject sequence order information without recurrence"
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
        title: "Specialized Architectures"
      },
      children: [
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Autoencoders",
                date: "Unsupervised",
                description: "Learn compressed representations by reconstructing input data"
              },
              {
                title: "Generative Adversarial Networks (GANs)",
                date: "Generative",
                description: "Two networks compete: generator creates data, discriminator evaluates it"
              },
              {
                title: "U-Net",
                date: "Computer Vision",
                description: "Encoder-decoder with skip connections for image segmentation"
              },
              {
                title: "ResNet",
                date: "Computer Vision",
                description: "Residual connections enable training very deep networks"
              },
              {
                title: "Graph Neural Networks",
                date: "Graph Data",
                description: "Process graph-structured data with node and edge relationships"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Architecture Selection Guidelines",
        background: true
      },
      children: [
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Data Type Considerations"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Images",
                      description: "CNNs for spatial patterns, Vision Transformers for global context"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "Sequences",
                      description: "RNNs/LSTMs for short sequences, Transformers for long sequences"
                    },
                    {
                      icon: <Network className="w-6 h-6" />,
                      title: "Graphs",
                      description: "Graph Neural Networks for node/edge relationships"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Tabular",
                      description: "MLPs often sufficient, sometimes gradient boosting preferred"
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
                content: "Task Considerations"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Classification",
                      description: "Add softmax/sigmoid output layer to base architecture"
                    },
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Regression",
                      description: "Linear output layer without activation function"
                    },
                    {
                      icon: <Cpu className="w-6 h-6" />,
                      title: "Generation",
                      description: "Autoregressive models, VAEs, GANs, or Diffusion models"
                    },
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "Reinforcement Learning",
                      description: "Policy networks, value networks, actor-critic architectures"
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
        title: "References and Further Learning"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Explore these resources to deepen your understanding of neural network architectures:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Foundational Papers"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Attention Is All You Need",
                      description: "Original Transformer paper (Vaswani et al., 2017)"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Deep Residual Learning",
                      description: "ResNet architecture paper (He et al., 2016)"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Long Short-Term Memory",
                      description: "LSTM paper (Hochreiter & Schmidhuber, 1997)"
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
                      title: "Deep Learning by Ian Goodfellow",
                      description: "Comprehensive textbook on neural network theory"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "CS231n: CNNs for Visual Recognition",
                      description: "Stanford course on convolutional neural networks"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "The Illustrated Transformer",
                      description: "Visual guide to understanding Transformer architecture"
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

export default function NeuralNetworkArchitecturesPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-cyan-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-cyan-600 dark:hover:text-cyan-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-cyan-600 dark:hover:text-cyan-400 transition-colors">
                Projects
              </a>
              <a href="/topics" className="text-cyan-600 dark:text-cyan-400 font-medium">
                Topics
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <TopicPageBuilder {...neuralNetworkArchitecturesPageData} />
      </article>
    </div>
  );
}