import { Database, Shield, Brain, Clock, Zap, Target, Search, Code, BookOpen, Video } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Long Short-Term Memory (LSTM) - ML Portfolio',
  description: 'Advanced RNN architecture that can learn long-term dependencies through gating mechanisms',
};

const lstmData = {
  title: "Long Short-Term Memory (LSTM)",
  header: {
    category: "Deep Learning",
    difficulty: "Intermediate" as const,
    readTime: "9 min read",
    description: "Advanced RNN architecture that solves the vanishing gradient problem through sophisticated gating mechanisms, enabling learning of long-term dependencies",
    relatedProjects: ["Custom GPT LLM"],
    gradientFrom: "from-indigo-50 to-purple-50",
    gradientTo: "dark:from-indigo-900/20 dark:to-purple-900/20",
    borderColor: "border-indigo-200 dark:border-indigo-800"
  },
  tags: {
    items: ['Sequential Data', 'Memory', 'NLP', 'Vanishing Gradient'],
    colorScheme: 'purple' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What is LSTM?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Long Short-Term Memory (LSTM) networks are a special kind of RNN capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997) and solve the vanishing gradient problem that plagues traditional RNNs."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Key Innovation",
            icon: <Database className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "LSTMs use a cell state and three gates (forget, input, output) to carefully regulate information flow, allowing them to remember important information for long periods while forgetting irrelevant details."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "LSTM Architecture"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The LSTM cell has three main components that control information flow:"
        },
        {
          type: 'features' as const,
          props: {
            items: [
              {
                icon: <Shield className="w-6 h-6" />,
                title: "Forget Gate",
                description: "Decides what information to discard from the cell state. Looks at previous hidden state and current input.",
                color: "red"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Input Gate",
                description: "Determines which new information to store in the cell state. Creates candidate values to be added.",
                color: "green"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Output Gate",
                description: "Controls what parts of the cell state to output as the hidden state for the next time step.",
                color: "blue"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "LSTM Mathematical Formulation"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "LSTM Equations",
            icon: <Clock className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Forget Gate:"
            }, 
            {
              type: 'math' as const,
              props: {block: true},
              content: "f_t = σ(W_f · [h_{t-1}, x_t] + b_f)"
            }, 
            {
              type: 'paragraph' as const,
              content: "Input Gate:"
            },
            {
              type: 'math' as const, 
              props: {block: true},
              content: "i_t = σ(W_i · [h_{t-1}, x_t] + b_i"
            },
            {
              type: 'paragraph' as const,
              content: "Candidate:"
            },
            {
              type: 'math' as const,
              props: {block: true},
              content: " C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)"
            },
            {
              type: 'paragraph' as const,
              content: "Cell State:"
            },
            {
              type: 'math' as const,
              props: {block: true},
              content: "C_t = f_t * C_{t-1} + i_t * C̃_t"
            },
            {
              type: 'paragraph' as const,
              content: "Output Gate:"
            },
            {
              type: 'math' as const,
              props: {block: true},
              content: "o_t = σ(W_o · [h_{t-1}, x_t] + b_o)"
            },
            {
              type: 'paragraph' as const,
              content: "Hidden State:"
            },
            {
              type: 'math' as const,
              props: {block: true},
              content: "h_t = o_t * tanh(C_t)"
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "LSTM Implementation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's how to implement LSTM from scratch and using PyTorch:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LSTMCell(nn.Module):
    """Custom LSTM Cell implementation"""
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Forget gate
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Input gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Candidate gate
        self.W_C = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Output gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, hidden_state):
        h_prev, C_prev = hidden_state
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Forget gate
        f_t = torch.sigmoid(self.W_f(combined))
        
        # Input gate
        i_t = torch.sigmoid(self.W_i(combined))
        
        # Candidate values
        C_tilde = torch.tanh(self.W_C(combined))
        
        # Update cell state
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Output gate
        o_t = torch.sigmoid(self.W_o(combined))
        
        # Update hidden state
        h_t = o_t * torch.tanh(C_t)
        
        return h_t, (h_t, C_t)

class CustomLSTM(nn.Module):
    """Multi-layer LSTM using custom LSTM cells"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create LSTM layers
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden states
        hidden_states = [
            (torch.zeros(batch_size, self.hidden_size),
             torch.zeros(batch_size, self.hidden_size))
            for _ in range(self.num_layers)
        ]
        
        outputs = []
        
        for t in range(seq_length):
            layer_input = x[:, t, :]
            
            for layer in range(self.num_layers):
                layer_input, hidden_states[layer] = self.lstm_cells[layer](
                    layer_input, hidden_states[layer]
                )
            
            outputs.append(layer_input)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        # Apply final linear layer
        outputs = self.fc(outputs)
        
        return outputs

# Compare with PyTorch LSTM
class PyTorchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Generate synthetic time series data
def generate_time_series(n_samples, seq_length):
    """Generate sine wave with noise"""
    X, y = [], []
    
    for _ in range(n_samples):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        
        # Generate sequence
        t = np.linspace(0, 4*np.pi, seq_length + 1)
        series = np.sin(freq * t + phase) + 0.1 * np.random.randn(seq_length + 1)
        
        X.append(series[:-1])
        y.append(series[1:])  # Predict next step
    
    return np.array(X), np.array(y)

# Generate data
seq_length = 20
X_train, y_train = generate_time_series(1000, seq_length)
X_test, y_test = generate_time_series(200, seq_length)

# Convert to tensors
X_train = torch.FloatTensor(X_train).unsqueeze(-1)
y_train = torch.FloatTensor(y_train).unsqueeze(-1)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test).unsqueeze(-1)

# Initialize models
custom_model = CustomLSTM(1, 32, 2, 1)
pytorch_model = PyTorchLSTM(1, 32, 2, 1)

print(f"Custom LSTM parameters: {sum(p.numel() for p in custom_model.parameters())}")
print(f"PyTorch LSTM parameters: {sum(p.numel() for p in pytorch_model.parameters())}")`,
          props: {
            language: 'python',
            title: 'Custom LSTM Implementation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Practical LSTM for Sentiment Analysis"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's a practical example using LSTM for sentiment analysis:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import re

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) 
                  for word in text.split()[:self.max_length]]
        
        # Pad sequence
        if len(indices) < self.max_length:
            indices += [self.vocab['<PAD>']] * (self.max_length - len(indices))
        
        return torch.LongTensor(indices), torch.LongTensor([label])

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 num_classes, dropout=0.3):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the final hidden state (concatenated for bidirectional)
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Dropout and classification
        output = self.dropout(final_hidden)
        output = self.fc(output)
        
        return output

# Sample data preparation
def preprocess_text(text):
    """Simple text preprocessing"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    return text

# Sample dataset (in practice, use real sentiment data)
sample_texts = [
    "I love this movie, it's fantastic!",
    "This film is terrible, I hate it.",
    "Great acting and amazing storyline.",
    "Boring and predictable plot.",
    "Excellent cinematography and direction.",
    "Worst movie I've ever seen.",
    "Beautiful and touching story.",
    "Complete waste of time."
]

sample_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative

# Preprocess texts
processed_texts = [preprocess_text(text) for text in sample_texts]

# Build vocabulary
all_words = set()
for text in processed_texts:
    all_words.update(text.split())

vocab = {'<PAD>': 0, '<UNK>': 1}
vocab.update({word: i+2 for i, word in enumerate(sorted(all_words))})

# Create dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    processed_texts, sample_labels, test_size=0.3, random_state=42
)

train_dataset = SentimentDataset(train_texts, train_labels, vocab)
test_dataset = SentimentDataset(test_texts, test_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# Initialize model
model = SentimentLSTM(
    vocab_size=len(vocab),
    embedding_dim=50,
    hidden_dim=32,
    num_layers=2,
    num_classes=2
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_texts, batch_labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(batch_texts)
        loss = criterion(outputs, batch_labels.squeeze())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')

# Test the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_texts, batch_labels in test_loader:
        outputs = model(batch_texts)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels.squeeze()).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')`,
          props: {
            language: 'python',
            title: 'LSTM for Sentiment Analysis'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "LSTM vs Traditional RNN"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "LSTM Advantages",
            icon: <Zap className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "Solves vanishing gradient problem",
                  "Can learn long-term dependencies",
                  "Better at selective memory",
                  "More stable training"
                ]
              },
            }
          ]
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'warning' as const,
            title: "Trade-offs",
            icon: <Database className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "More complex architecture",
                  "Higher computational cost",
                  "More parameters to train",
                  "Slower than simple RNNs"
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
        title: "Learning Resources",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Comprehensive resources to deepen your understanding of LSTM (Long Short-Term Memory):"
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
                      title: "Long Short-Term Memory",
                      description: "Hochreiter & Schmidhuber (1997) - Original LSTM paper"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "LSTM: A Search Space Odyssey",
                      description: "Greff et al. (2017) - Comprehensive analysis of LSTM variants"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "An Empirical Exploration of Recurrent Network Architectures",
                      description: "Jozefowicz et al. (2015) - Systematic evaluation of RNN architectures"
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
                      title: "Understanding LSTM Networks",
                      description: "Christopher Olah's intuitive explanation of LSTM architecture"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "PyTorch LSTM Tutorial",
                      description: "Hands-on implementation guide for LSTM in PyTorch"
                    },
                    {
                      icon: <BookOpen className="w-6 h-6" />,
                      title: "Keras LSTM Documentation",
                      description: "Official Keras LSTM layer documentation with examples"
                    },
                    {
                      icon: <Video className="w-6 h-6" />,
                      title: "LSTM for Time Series Forecasting",
                      description: "Practical guide to using LSTMs for time series prediction"
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

export default function LSTMPage() {
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
        <TopicPageBuilder {...lstmData} />
      </article>
    </div>
  );
}
