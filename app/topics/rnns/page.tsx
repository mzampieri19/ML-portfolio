import { Repeat, Brain, Clock, Zap, ArrowRight, Network, Search, BookOpen, Code, ExternalLink } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Recurrent Neural Networks (RNNs) - ML Portfolio',
  description: 'Neural networks designed for sequential data with memory capabilities',
};

const rnnData = {
  title: "Recurrent Neural Networks (RNNs)",
  header: {
    category: "Deep Learning",
    difficulty: "Intermediate" as const,
    readTime: "8 min read",
    description: "Neural networks designed for sequential data with memory capabilities, enabling processing of variable-length sequences and time series data",
    relatedProjects: ["Custom GPT LLM"],
    gradientFrom: "from-purple-50 to-blue-50",
    gradientTo: "dark:from-purple-900/20 dark:to-blue-900/20",
    borderColor: "border-purple-200 dark:border-purple-800"
  },
  tags: {
    items: ['Sequential Data', 'NLP', 'Time Series', 'Memory'],
    colorScheme: 'purple' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What are RNNs?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Recurrent Neural Networks (RNNs) are a class of neural networks designed to work with sequential data. Unlike feedforward networks, RNNs have loops that allow information to persist, giving them a form of memory that makes them ideal for tasks involving sequences."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Key Innovation",
            icon: <Brain className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "RNNs can process sequences of variable length by maintaining hidden states that capture information from previous time steps, making them perfect for natural language processing, time series analysis, and any task where context matters."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "How RNNs Work"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "An RNN processes sequences step by step, maintaining a hidden state that gets updated at each time step:"
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "RNN Formula",
            icon: <Clock className="w-6 h-6" />
          },
          children: [
            {
              type: 'math' as const,
              props: {block: true},
              content: "h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)\ny_t = W_hy * h_t + b_y\n\n"
            }, 
            {
              type: 'paragraph' as const,
              content: "Where:\n- \(h_t\) is the hidden state at time \(t\)\n- \(x_t\) is the input at time \(t\)\n- \(W_hh\), \(W_xh\), and \(W_hy\) are weight matrices\n- \(b_h\) and \(b_y\) are bias vectors"
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Basic RNN Implementation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's how to implement a simple RNN from scratch and using PyTorch:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Define layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward pass through RNN
        out, hidden = self.rnn(x, h0)
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

# Create a simple sequence prediction task
def generate_sine_data(seq_length, num_samples):
    """Generate sine wave data for sequence prediction"""
    X, y = [], []
    
    for _ in range(num_samples):
        # Random starting point
        start = np.random.uniform(0, 2*np.pi)
        
        # Generate sequence
        sequence = np.sin(np.linspace(start, start + 2*np.pi, seq_length + 1))
        
        X.append(sequence[:-1])  # Input sequence
        y.append(sequence[-1])   # Target (next value)
    
    return np.array(X), np.array(y)

# Generate training data
seq_length = 10
num_samples = 1000

X_train, y_train = generate_sine_data(seq_length, num_samples)
X_test, y_test = generate_sine_data(seq_length, 200)

# Convert to tensors
X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # Add feature dimension
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test)

# Initialize model
model = SimpleRNN(input_size=1, hidden_size=32, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Evaluate model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs.squeeze(), y_test)
    print(f'Test Loss: {test_loss.item():.6f}')`,
          props: {
            language: 'python',
            title: 'Simple RNN Implementation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "RNN Variants and Applications"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Repeat className="w-6 h-6" />,
                title: "Vanilla RNN",
                description: "Basic RNN with simple recurrent connections. Good for short sequences but suffers from vanishing gradients.",
                color: "blue"
              },
              {
                icon: <Network className="w-6 h-6" />,
                title: "LSTM",
                description: "Long Short-Term Memory networks solve vanishing gradient problem with gating mechanisms.",
                color: "green"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "GRU",
                description: "Gated Recurrent Units are simpler than LSTM but often perform similarly well.",
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
        title: "Text Generation with RNN"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's an example of using RNNs for character-level text generation:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.nn as nn
import torch.nn.functional as F
import string

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, 
                          dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # Embed characters
        embedded = self.embedding(x)
        
        # Pass through RNN
        if hidden is None:
            output, hidden = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded, hidden)
        
        # Apply linear layer to get logits
        output = self.fc(output)
        
        return output, hidden

class TextGenerator:
    def __init__(self, text):
        # Create character mappings
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in text]
    
    def get_batches(self, seq_length, batch_size):
        """Create training batches"""
        n_batches = len(self.data) // (seq_length * batch_size)
        
        # Trim data to fit evenly into batches
        data_len = n_batches * seq_length * batch_size
        data = self.data[:data_len]
        
        # Reshape into batches
        data = np.array(data).reshape(batch_size, -1)
        
        for i in range(0, data.shape[1] - seq_length, seq_length):
            x = data[:, i:i+seq_length]
            y = data[:, i+1:i+seq_length+1]
            yield torch.LongTensor(x), torch.LongTensor(y)
    
    def generate_text(self, model, start_string, length=100, temperature=0.8):
        """Generate text using the trained model"""
        model.eval()
        
        # Convert start string to tensor
        chars = [self.char_to_idx[ch] for ch in start_string]
        input_tensor = torch.LongTensor(chars).unsqueeze(0)
        
        hidden = None
        generated = start_string
        
        with torch.no_grad():
            for _ in range(length):
                output, hidden = model(input_tensor, hidden)
                
                # Apply temperature scaling
                output = output[:, -1, :] / temperature
                probabilities = F.softmax(output, dim=1)
                
                # Sample from the distribution
                next_char_idx = torch.multinomial(probabilities, 1).item()
                next_char = self.idx_to_char[next_char_idx]
                
                generated += next_char
                input_tensor = torch.LongTensor([[next_char_idx]])
        
        return generated

# Example usage with sample text
sample_text = """
The quick brown fox jumps over the lazy dog. 
Machine learning is fascinating and powerful.
Neural networks can learn complex patterns from data.
"""

# Initialize text generator
text_gen = TextGenerator(sample_text)

# Create and train model
model = CharRNN(text_gen.vocab_size, hidden_size=128, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop (simplified)
seq_length = 25
batch_size = 1
num_epochs = 50

for epoch in range(num_epochs):
    for batch_x, batch_y in text_gen.get_batches(seq_length, batch_size):
        optimizer.zero_grad()
        
        output, _ = model(batch_x)
        loss = criterion(output.view(-1, text_gen.vocab_size), batch_y.view(-1))
        
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Generate text
generated = text_gen.generate_text(model, "The ", length=100)
print("Generated text:", generated)`,
          props: {
            language: 'python',
            title: 'Character-Level Text Generation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Common Challenges"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'warning' as const,
            title: "Vanishing Gradient Problem",
            icon: <ArrowRight className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "In long sequences, gradients can become very small during backpropagation, making it difficult to learn long-term dependencies. Solutions include LSTM, GRU, and gradient clipping."
            }
          ]
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'error' as const,
            title: "Exploding Gradients",
            icon: <Zap className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Gradients can also become very large, causing unstable training. Gradient clipping and proper initialization help mitigate this issue."
            }
          ]
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
          content: "Comprehensive resources to deepen your understanding of recurrent neural networks:"
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
                      title: "Finding Structure in Time",
                      description: "Elman (1990) - Early work on RNNs and temporal structure"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "On the Difficulty of Training Recurrent Neural Networks",
                      description: "Pascanu et al. (2013) - Challenges in training RNNs"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Empirical Evaluation of Gated Recurrent Neural Networks",
                      description: "Chung et al. (2014) - GRU vs LSTM comparison"
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
                      title: "The Unreasonable Effectiveness of RNNs",
                      description: "Karpathy's famous blog post on RNN capabilities"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "PyTorch RNN Tutorial",
                      description: "Implementation from scratch with practical examples"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "TensorFlow RNN Guide",
                      description: "Production-ready RNN implementations"
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
    colorScheme: 'purple' as const
  }
};

export default function RNNPage() {
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
        <TopicPageBuilder {...rnnData} />
      </article>
    </div>
  );
}