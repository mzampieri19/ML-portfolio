import { MessageSquare, Brain, Layers, Target, TrendingUp, Zap, Code, BarChart3, Eye, ExternalLink, Book, Cpu } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';
import CodeBlock from '../../components/CodeBlock';

export const metadata = {
  title: 'Large Language Models - ML Portfolio Topics',
  description: 'Learn about Large Language Models (LLMs) - neural networks trained on vast amounts of text data',
};

const llmPageData = {
  title: "Large Language Models (LLMs)",
  header: {
    date: "Natural Language Processing",
    readTime: "10 min read",
    description: "Neural networks trained on vast amounts of text data to understand and generate human language",
    gradientFrom: "from-green-50 to-blue-50",
    gradientTo: "dark:from-green-900/20 dark:to-blue-900/20",
    borderColor: "border-green-200 dark:border-green-800",
    difficulty: "Advanced" as const,
    category: "Natural Language Processing",
    relatedProjects: ["custom-gpt-llm", "real-salary"]
  },
  tags: {
    items: ['NLP', 'GPT', 'Transformers', 'Language Generation', 'Fine-tuning'],
    colorScheme: 'green' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What are Large Language Models?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Large Language Models (LLMs) are neural networks with billions or trillions of parameters that have been trained on massive datasets of text from the internet, books, and other sources. They can understand context, generate human-like text, and perform various language tasks without specific training for each task."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Key Breakthrough",
            icon: <MessageSquare className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "LLMs demonstrate emergent abilities - capabilities that appear when models reach a certain scale, such as few-shot learning, reasoning, and following complex instructions without explicit training."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Transformer Architecture"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "LLMs are built on the Transformer architecture, which uses self-attention mechanisms to process sequences of text. The key innovation is the ability to attend to all positions in a sequence simultaneously, rather than processing sequentially like RNNs."
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Self-Attention Mechanism"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Query, Key, Value",
                      description: "Each token creates queries, keys, and values for attention computation"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Parallel Processing",
                      description: "All positions computed simultaneously, enabling efficient training"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Long-Range Dependencies",
                      description: "Can capture relationships between distant tokens in text"
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
                content: "Model Components"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Layers className="w-6 h-6" />,
                      title: "Multi-Head Attention",
                      description: "Multiple attention heads capture different types of relationships"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "Feed-Forward Networks",
                      description: "Dense layers for non-linear transformations"
                    },
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "Residual Connections",
                      description: "Skip connections and layer normalization for stable training"
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
        filename: "transformer_attention.py",
        code: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
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
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(attention_output)
        
        return output, attention_weights`
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Training Process",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "LLMs are trained using next-token prediction: given a sequence of tokens, the model learns to predict the next token. This simple objective, when applied at scale, leads to sophisticated language understanding and generation capabilities."
        },
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Pre-training",
                date: "Phase 1",
                description: "Train on massive text corpora using next-token prediction objective"
              },
              {
                title: "Supervised Fine-tuning",
                date: "Phase 2",
                description: "Fine-tune on high-quality instruction-following examples"
              },
              {
                title: "Reinforcement Learning from Human Feedback (RLHF)",
                date: "Phase 3",
                description: "Use human preferences to align model behavior with desired outcomes"
              },
              {
                title: "Constitutional AI",
                date: "Phase 4",
                description: "Additional training to make models more helpful, harmless, and honest"
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
        filename: "llm_training.py",
        code: `def train_language_model(model, dataloader, optimizer, device):
    """
    Training loop for language model using next-token prediction
    """
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # batch contains tokenized text sequences
        input_ids = batch['input_ids'].to(device)  # [batch_size, seq_len]
        
        # Prepare inputs and targets for next-token prediction
        inputs = input_ids[:, :-1]  # All tokens except last
        targets = input_ids[:, 1:]  # All tokens except first (shifted)
        
        # Forward pass
        outputs = model(inputs)
        logits = outputs.logits  # [batch_size, seq_len-1, vocab_size]
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # Flatten for loss calculation
            targets.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

# Example usage
for epoch in range(num_epochs):
    avg_loss = train_language_model(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")`
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Model Families and Architectures"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Brain className="w-6 h-6" />,
                title: "GPT Family (Decoder-only)",
                description: "Autoregressive models for text generation (GPT-3, GPT-4, ChatGPT)"
              },
              {
                icon: <Book className="w-6 h-6" />,
                title: "BERT Family (Encoder-only)",
                description: "Bidirectional models for understanding tasks (BERT, RoBERTa, DeBERTa)"
              },
              {
                icon: <Layers className="w-6 h-6" />,
                title: "T5 Family (Encoder-Decoder)",
                description: "Text-to-text models for various NLP tasks (T5, FLAN-T5, UL2)"
              },
              {
                icon: <Cpu className="w-6 h-6" />,
                title: "LLaMA Family",
                description: "Efficient open-source models (LLaMA, Alpaca, Vicuna)"
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
        title: "Key Capabilities",
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
                content: "Language Understanding"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Reading Comprehension",
                      description: "Understand and answer questions about text"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Sentiment Analysis",
                      description: "Determine emotional tone and sentiment"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Named Entity Recognition",
                      description: "Identify people, places, organizations in text"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Text Classification",
                      description: "Categorize text into predefined classes"
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
                content: "Language Generation"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <MessageSquare className="w-6 h-6" />,
                      title: "Text Completion",
                      description: "Continue and complete partial text"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "Code Generation",
                      description: "Write and debug programming code"
                    },
                    {
                      icon: <Book className="w-6 h-6" />,
                      title: "Creative Writing",
                      description: "Generate stories, poems, and articles"
                    },
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "Translation",
                      description: "Translate between different languages"
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
        title: "Fine-tuning and Adaptation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Pre-trained LLMs can be adapted for specific tasks through various fine-tuning techniques, allowing them to perform specialized functions while retaining their general language capabilities."
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Target className="w-6 h-6" />,
                title: "Task-Specific Fine-tuning",
                description: "Adapt model for specific tasks like classification or summarization"
              },
              {
                icon: <Layers className="w-6 h-6" />,
                title: "Parameter-Efficient Fine-tuning (PEFT)",
                description: "LoRA, adapters, and other methods that update few parameters"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "In-Context Learning",
                description: "Learn new tasks from examples provided in the prompt"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "Instruction Tuning",
                description: "Train models to follow instructions and user intent"
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
        title: "References and Further Learning",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Explore these resources to deepen your understanding of Large Language Models:"
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
                      description: "Original Transformer paper by Vaswani et al. (2017)"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Language Models are Few-Shot Learners",
                      description: "GPT-3 paper demonstrating emergent abilities"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Training language models to follow instructions",
                      description: "InstructGPT paper on human feedback training"
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
                      title: "Hugging Face Transformers",
                      description: "Comprehensive library and documentation"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "CS224N: Natural Language Processing",
                      description: "Stanford course on NLP with deep learning"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "The Illustrated Transformer",
                      description: "Visual explanation of Transformer architecture"
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

export default function LargeLanguageModelsPage() {
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
              <a href="/topics" className="text-indigo-600 dark:text-indigo-400 font-medium">
                Topics
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <TopicPageBuilder {...llmPageData} />
      </article>
    </div>
  );
}
