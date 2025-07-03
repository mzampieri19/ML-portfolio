import { Brain, Code, Cpu, Database, GitBranch, Settings, Target, TrendingUp, Zap, BookOpen, Play, Download, CheckCircle } from 'lucide-react';
import PageBuilder from '../../components/PageBuilder';
import CodeBlock from '../../components/CodeBlock';

export const metadata = {
  title: 'Custom GPT LLM - ML Portfolio',
  description: 'GPT LLM from scratch, includes encoding, decoding, data extraction, and training',
};

const customGptPageData = {
  title: "Custom GPT LLM",
  header: {
    date: "Summer 2025",
    readTime: "10 min read",
    description: "Building a complete GPT language model from scratch with custom tokenization, attention mechanisms, and training pipeline",
    githubUrl: "https://github.com/mzampieri19/Custom-GPT-LLM",
    gradientFrom: "from-blue-50 to-indigo-50",
    gradientTo: "dark:from-blue-900/20 dark:to-indigo-900/20",
    borderColor: "border-blue-200 dark:border-blue-800"
  },
  tags: {
    items: ['LLM', 'GPT', 'From Scratch', 'Neural Network', 'GenAI', 'Transformer', 'Attention', 'PyTorch'],
    colorScheme: 'blue' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "Project Summary",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "This project demonstrates how to build, train, and use a custom Generative Pre-trained Transformer (GPT) language model from scratch using PyTorch. Inspired by the freeCodeCamp.org tutorial, the project walks through all steps required to preprocess data, define the model architecture, train the model, and generate new text samples."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Apple Silicon Optimized",
            icon: <Cpu className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "The project is designed to run efficiently on Apple Silicon (M1/M2/M3) using the Metal backend (mps), making it accessible for development on modern MacBooks."
            }
          ]
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Custom Architecture",
                description: "Multi-layer decoder-only transformer built from scratch"
              },
              {
                icon: <Database className="w-6 h-6" />,
                title: "Data Pipeline",
                description: "Complete data preprocessing and batch preparation"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Training System",
                description: "Custom training loop with loss estimation and model checkpointing"
              },
              {
                icon: <Code className="w-6 h-6" />,
                title: "Text Generation",
                description: "Sampling-based text generation with temperature control"
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
        title: "Model Versions & Progressive Training Results"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Each model version showed progressive improvement in text quality and coherence:"
        },
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "model-01",
                date: "Proof of Concept",
                description: "Trained for a few thousand epochs on minimal dataset. Mostly incoherent output but demonstrated basic grammar and sentence structure."
              },
              {
                title: "model-02", 
                date: "Extended Training",
                description: "Same dataset as model-01 but trained for ~50,000 epochs. Noticeably improved grammar and structure, though still generated non-existent words."
              },
              {
                title: "model-03",
                date: "Larger Dataset", 
                description: "Trained on ~300,000 lines of classic literature from Project Gutenberg. Strong grammar and human-like sentence structure with occasional non-existent words."
              },
              {
                title: "model-04",
                date: "Scale Up",
                description: "Common Crawl dataset with 1 million lines (~4GB). Training split into sections, completed 5,000 epochs on Google Colab with best performance yet."
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Project Structure",
        background: true
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <BookOpen className="w-6 h-6" />,
                title: "gpt-v1.ipynb",
                description: "Main Jupyter notebook containing all code for data prep, model, training, etc."
              },
              {
                icon: <Play className="w-6 h-6" />,
                title: "training.py",
                description: "A Python script to run the training code"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "chatbot.py", 
                description: "A simple terminal-based application of the LLM"
              },
              {
                icon: <Database className="w-6 h-6" />,
                title: "data_extract.py",
                description: "Script to extract workable data from the JSON file"
              },
              {
                icon: <Code className="w-6 h-6" />,
                title: "vocab.txt",
                description: "Text file containing the vocabulary (unique characters) for encoding/decoding"
              },
              {
                icon: <Download className="w-6 h-6" />,
                title: "model-01.pkl",
                description: "Saved PyTorch model checkpoint (after training)"
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
        title: "Step-by-Step Implementation Guide"
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
                content: "Data Preparation"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Input Files"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "extracted_train_data.txt – The main text corpus for training"
                  },
                  {
                    type: 'paragraph' as const,
                    content: "extracted_val_data.txt – A separate text corpus for validation"
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Vocabulary Extraction"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Read all unique characters from your data and save them to vocab.txt. This ensures the model can encode/decode every character in your dataset."
                  }
                ]
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Encoding & Decoding"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'warning' as const,
                  title: "Character Mapping"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Each character is mapped to a unique integer using dictionaries (string_to_int and int_to_string). Functions encode and decode convert between text and integer sequences."
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'error' as const,
                  title: "Batch Preparation"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "The function get_random_chunk(split) reads a random chunk from training or validation files. get_batch(split) prepares batches of input (x) and target (y) tensors for training."
                  }
                ]
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Model Architecture",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The model is built from scratch using PyTorch, including multi-head self-attention, feed-forward layers, layer normalization and residual connections:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Target className="w-6 h-6" />,
                title: "Multi-Head Attention",
                description: "Custom implementation of scaled dot-product attention with multiple heads for parallel processing"
              },
              {
                icon: <GitBranch className="w-6 h-6" />,
                title: "Feed-Forward Networks",
                description: "Position-wise fully connected layers with ReLU activation and dropout"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Layer Normalization",
                description: "Applied before attention and feed-forward blocks for training stability"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "Residual Connections",
                description: "Skip connections around each sub-layer to enable deep network training"
              }
            ],
            columns: 2
          }
        }
      ]
    },
    {
      type: 'custom' as const,
      component: (
        <CodeBlock language="python">
{`class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x`}
        </CodeBlock>
      )
    },
    {
      type: 'custom' as const,
      component: (
        <CodeBlock language="python">
{`class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, index, max_new_tokens, temperature=1.0, top_p=0.9):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :]
            next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)
            index = torch.cat((index, next_token), dim=1)
        return index`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Training Pipeline"
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
                content: "Device Selection"
              },
              {
                type: 'paragraph' as const,
                content: "Uses Apple Silicon GPU via torch.device('mps' if torch.backends.mps.is_available() else 'cpu') for optimal performance on M1/M2/M3 chips."
              },
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Training Loop"
              },
              {
                type: 'paragraph' as const,
                content: "Runs for specified iterations with periodic evaluation, uses AdamW optimizer and learning rate scheduler for stable training."
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Loss Estimation"
              },
              {
                type: 'paragraph' as const,
                content: "The estimate_loss() function computes average loss over several batches for both train and validation splits to monitor progress."
              },
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Model Persistence"
              },
              {
                type: 'paragraph' as const,
                content: "Models are saved using Python's pickle module and can be reloaded for further training or inference."
              }
            ]
          }
        }
      ]
    },
    {
      type: 'custom' as const,
      component: (
        <CodeBlock language="python">
{`# Training configuration
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

# Training loop
for iter in range(max_iters):
    # Periodic evaluation
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f'step: {iter}, train loss: {losses["train"]:.3f}, val loss: {losses["val"]:.3f}')
    
    # Forward pass
    xb, yb = get_batch('train')
    logits, loss = m.forward(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

print(f"Iteration {iter}, Loss: {loss.item()}")

# Save trained model
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
    
# Loss estimation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Results and Performance",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The custom GPT model demonstrates strong performance across various text generation tasks:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Coherent Text Generation",
                description: "Maintains context and coherence over long passages with proper grammar and flow"
              },
              {
                icon: <Code className="w-6 h-6" />,
                title: "Diverse Writing Styles", 
                description: "Adapts to different genres and tones based on training data characteristics"
              },
              {
                icon: <Database className="w-6 h-6" />,
                title: "Factual Knowledge",
                description: "Demonstrates learned knowledge from training data with accurate information recall"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Creative Writing",
                description: "Capable of creative storytelling and poetry generation with imagination"
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
        title: "Key Technical Insights"
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
                content: "Architecture Decisions"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "Pre-normalization",
                      description: "Layer normalization before attention and feed-forward blocks"
                    },
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "GELU Activation",
                      description: "Chose GELU over ReLU for smoother gradients"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Learned Positional Encoding",
                      description: "More flexible than fixed sinusoidal encoding"
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
                content: "Training Optimizations"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Learning Rate Schedule",
                      description: "Cosine annealing with warmup for optimal convergence"
                    },
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "Weight Decay",
                      description: "Applied to all parameters except biases and layer norms"
                    },
                    {
                      icon: <Cpu className="w-6 h-6" />,
                      title: "Mixed Precision",
                      description: "Automatic mixed precision for faster training"
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
        title: "How to Replicate This Project",
        background: true
      },
      children: [
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Clone Repository",
                date: "Step 1",
                description: "Clone or download the repository from GitHub to your local machine"
              },
              {
                title: "Prepare Data",
                date: "Step 2",
                description: "Place training and validation text files as extracted_train_data.txt and extracted_val_data.txt. Ensure vocab.txt contains all unique characters"
              },
              {
                title: "Setup Environment",
                date: "Step 3",
                description: "Use Python 3.9+ and install dependencies: pip install torch torchvision"
              },
              {
                title: "Run Training",
                date: "Step 4",
                description: "Open gpt-v1.ipynb in VS Code or Jupyter, step through each cell and run the training loop"
              },
              {
                title: "Generate Text",
                date: "Step 5",
                description: "Use the trained model to generate new text samples with the provided generation code"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Tips for Best Results"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Database className="w-6 h-6" />,
                title: "Data Quality",
                description: "Clean and preprocess your data thoroughly for optimal training results"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Hyperparameter Tuning",
                description: "Tune n_embd, n_head, n_layer, learning_rate, and dropout for your specific dataset"
              },
              {
                icon: <Cpu className="w-6 h-6" />,
                title: "Compute Resources",
                description: "Larger models and more data require significant compute and memory resources"
              },
              {
                icon: <CheckCircle className="w-6 h-6" />,
                title: "Validation Monitoring",
                description: "Always monitor validation loss to detect and prevent overfitting"
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
        title: "Conclusion",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "This custom GPT implementation provides a comprehensive understanding of transformer-based language models. Building every component from scratch - tokenization, attention mechanisms, training pipeline, and generation strategies - offers invaluable insights into the inner workings of modern LLMs."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Key Achievements",
            icon: <CheckCircle className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "This foundation enables further exploration of advanced language model techniques and provides a solid base for research and development in natural language processing, demonstrating deep understanding of transformer architecture and practical experience with large-scale model training."
            }
          ]
        }
      ]
    }
  ],
  navigation: {
    colorScheme: 'blue' as const
  }
};

export default function CustomGptLlmPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Projects
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <PageBuilder {...customGptPageData} />
      </article>
    </div>
  );
}
