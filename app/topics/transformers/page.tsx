import { Layers, Zap, Brain, Target, Eye, ArrowRightLeft, Search, ExternalLink, Video } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';
import { blockquote } from 'framer-motion/client';

export const metadata = {
  title: 'Transformer Architecture - ML Portfolio',
  description: 'Attention-based neural network architecture that revolutionized NLP and beyond',
};

const transformerData = {
  title: "Transformer Architecture",
  header: {
    category: "Natural Language Processing",
    difficulty: "Advanced" as const,
    readTime: "12 min read",
    description: "Revolutionary attention-based neural network architecture that has transformed natural language processing and enabled breakthrough models like GPT and BERT",
    relatedProjects: ["Custom GPT LLM"],
    gradientFrom: "from-emerald-50 to-teal-50",
    gradientTo: "dark:from-emerald-900/20 dark:to-teal-900/20",
    borderColor: "border-emerald-200 dark:border-emerald-800"
  },
  tags: {
    items: ['Attention', 'NLP', 'Self-Attention', 'Transformers'],
    colorScheme: 'green' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What are Transformers?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Transformers are a neural network architecture introduced in the paper 'Attention Is All You Need' (Vaswani et al., 2017). They revolutionized NLP by replacing recurrent layers with self-attention mechanisms, enabling parallel processing and better capture of long-range dependencies."
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
              content: "Transformers eliminate the need for recurrence and convolutions entirely, relying solely on attention mechanisms to draw global dependencies between input and output."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Transformer Architecture Components"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            items: [
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Self-Attention",
                description: "Allows each position to attend to all positions in the previous layer, capturing relationships across the entire sequence.",
                color: "blue"
              },
              {
                icon: <Layers className="w-6 h-6" />,
                title: "Multi-Head Attention",
                description: "Runs multiple attention mechanisms in parallel, allowing the model to focus on different aspects simultaneously.",
                color: "green"
              },
              {
                icon: <ArrowRightLeft className="w-6 h-6" />,
                title: "Encoder-Decoder",
                description: "Original architecture with encoder for understanding input and decoder for generating output.",
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
        title: "Self-Attention Mechanism"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Attention Formula",
            icon: <Target className="w-6 h-6" />
          },
          children: [
            {
              type: 'math' as const,
              props: {block: true},
              content: "Attention(Q, K, V) = softmax(QK^T / √d_k)V\n\n"
            },
            {
              type: 'paragraph' as const,
              content: "Where:\n- Q = Queries matrix\n- K = Keys matrix  \n- V = Values matrix\n- d_k = dimension of key vectors."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Transformer Implementation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's a simplified implementation of the Transformer architecture:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        return context, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_length, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        return x

# Example usage
def create_transformer_model():
    model = TransformerEncoder(
        vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_length=1000,
        dropout=0.1
    )
    return model

# Create and test model
model = create_transformer_model()
batch_size, seq_length = 2, 10
input_ids = torch.randint(0, 10000, (batch_size, seq_length))

output = model(input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {output.shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")`,
          props: {
            language: 'python',
            title: 'Transformer Implementation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Training a Simple Transformer"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's how to train a Transformer for a language modeling task:"
        },
        {
          type: 'codeBlock' as const,
          content: `class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_length, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers, d_ff, max_length, dropout
        )
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        encoder_output = self.encoder(x, mask)
        logits = self.output_projection(encoder_output)
        return logits

def create_causal_mask(seq_length):
    """Create causal mask for decoder (prevents looking at future tokens)"""
    mask = torch.tril(torch.ones(seq_length, seq_length))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

def train_transformer(model, dataloader, num_epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding token
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    
    # Learning rate scheduler (warmup + decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 4000 ** -1.5)
    )
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch.to(device)  # Assuming batch contains input_ids
            
            # Create causal mask for language modeling
            seq_length = input_ids.size(1)
            mask = create_causal_mask(seq_length).to(device)
            
            # Prepare input and targets
            inputs = input_ids[:, :-1]  # All but last token
            targets = input_ids[:, 1:]  # All but first token (shifted)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(inputs, mask)
            
            # Reshape for loss computation
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            # Compute loss
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0):
    """Generate text using trained transformer model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Create causal mask
            seq_length = input_ids.size(1)
            mask = create_causal_mask(seq_length).to(device)
            
            # Forward pass
            logits = model(input_ids, mask)
            
            # Get logits for last position
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if end token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# Example usage
language_model = TransformerLanguageModel(
    vocab_size=50000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_length=1000
)

print(f"Language Model Parameters: {sum(p.numel() for p in language_model.parameters()):,}")

# Training would require a proper dataloader with tokenized text
# train_transformer(language_model, train_dataloader)`,
          props: {
            language: 'python',
            title: 'Training Transformer for Language Modeling'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Transformer Variants"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Popular Models",
            icon: <Zap className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "BERT: Bidirectional Encoder Representations from Transformers",
                  "GPT: Generative Pre-trained Transformer",
                  "T5: Text-to-Text Transfer Transformer",
                  "RoBERTa: Robustly optimized BERT pretraining"
                ]
              },
            }
          ]
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Recent Innovations",
            icon: <Brain className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "Vision Transformer (ViT): Transformers for images",
                  "Switch Transformer: Sparse mixture of experts",
                  "Reformer: Memory-efficient attention",
                  "Linformer: Linear complexity attention"
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
          content: "Comprehensive resources to deepen your understanding of transformers:"
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
                      title: "Attention Is All You Need",
                      description: "Vaswani et al. (2017) - Original Transformer architecture paper"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "BERT: Pre-training of Deep Bidirectional Transformers",
                      description: "Devlin et al. (2018) - Bidirectional encoder representations"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Language Models are Few-Shot Learners",
                      description: "Brown et al. (2020) - GPT-3 and large-scale language modeling"
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
                      title: "The Illustrated Transformer",
                      description: "Jay Alammar's visual guide to understanding Transformers"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Hugging Face Transformers",
                      description: "Comprehensive library and documentation for Transformer models"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "The Annotated Transformer",
                      description: "Line-by-line implementation of the Transformer model"
                    },
                    {
                      icon: <Video className="w-6 h-6" />,
                      title: "Attention and Augmented RNNs",
                      description: "Interactive visualizations of attention mechanisms"
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

export default function TransformerPage() {
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
        <TopicPageBuilder {...transformerData} />
      </article>
    </div>
  );
}