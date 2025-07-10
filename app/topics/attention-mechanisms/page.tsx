import TopicPageBuilder from '../../components/TopicPageBuilder';
import {Brain, Target, AlertTriangle, Focus, ArrowRight, Eye, Network, Zap, Search, MessageSquare, Layers, TrendingUp } from 'lucide-react';

export const metadata = {
  title: 'Attention Mechanisms - ML Portfolio',
  description: 'Neural network components that allow models to focus on relevant parts of input data - the foundation of modern NLP and computer vision',
};

const attentionTopicData = {
  title: "Attention Mechanisms",
  header: {
    category: "Deep Learning",
    difficulty: "Intermediate" as const,
    readTime: "15 min read",
    description: "Neural network components that allow models to focus on relevant parts of input data, revolutionizing how we process sequences and enabling breakthrough models like Transformers.",
    relatedProjects: ["custom-gpt-llm"],
    gradientFrom: "from-purple-50 to-pink-50",
    gradientTo: "dark:from-purple-900/20 dark:to-pink-900/20",
    borderColor: "border-purple-200 dark:border-purple-800"
  },
  tags: {
    items: ['Attention', 'Focus', 'Sequence Modeling', 'NLP', 'Transformers'],
    colorScheme: 'purple' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What are Attention Mechanisms?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Attention mechanisms are neural network components that allow models to dynamically focus on relevant parts of input data when making predictions. Instead of processing all input equally, attention mechanisms compute weights that determine which parts of the input are most relevant for the current task, mimicking how human attention works."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Revolutionary Impact",
            icon: <Brain className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Attention mechanisms have revolutionized machine learning, enabling breakthrough models like Transformers, BERT, and GPT. They solved the fundamental problem of how to process variable-length sequences while maintaining long-range dependencies."
            }
          ]
        },
        {
          type: 'twoColumn' as const,
          props: {
            left: [
              {
                type: 'paragraph' as const,
                content: "Key Benefits:"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Dynamic focus on relevant information",
                    "Parallel processing of sequences",
                    "Long-range dependency modeling",
                    "Interpretable attention weights"
                  ]
                }
              }
            ],
            right: [
              {
                type: 'paragraph' as const,
                content: "Core Components:"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Query (Q): What we're looking for",
                    "Key (K): What we're looking in",
                    "Value (V): What we extract",
                    "Attention weights: How much focus"
                  ]
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
        title: "How Attention Works",
        background: false
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The attention mechanism computes a weighted average of values, where the weights are determined by the similarity between queries and keys. This process can be broken down into three main steps:"
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Mathematical Formula",
            icon: <Target className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "The scaled dot-product attention is computed as: **Attention(Q, K, V) = softmax(QK^T / √d_k)V**"
            },
            {
              type: 'paragraph' as const,
              content: "Where Q, K, V are the query, key, and value matrices, and d_k is the dimension of the key vectors."
            }
          ]
        },
        {
          type: 'codeBlock' as const,
          props: {
            language: 'python',
            title: "Basic Attention Computation"
          },
          content: `import torch
import torch.nn.functional as F
import math

def attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention
    Args:
        query: (batch_size, seq_len, d_model)
        key: (batch_size, seq_len, d_model)
        value: (batch_size, seq_len, d_model)
        mask: Optional mask for padding tokens
    """
    d_k = query.size(-1)
    
    # Step 1: Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Step 2: Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 3: Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 4: Apply attention weights to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights`
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'warning' as const,
            title: "Scaling Factor",
            icon: <AlertTriangle className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "The scaling factor √d_k is crucial for stable gradients. Without it, the softmax function can become too sharp, leading to vanishing gradients during training."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Multi-Head Attention",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Multi-head attention allows the model to attend to different types of information simultaneously. Instead of using a single attention head, the model splits the input into multiple heads, each focusing on different aspects of the relationships."
        },
        {
          type: 'codeBlock' as const,
          props: {
            language: 'python',
            title: "Multi-Head Attention Implementation"
          },
          content: `import torch
import torch.nn as nn

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
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention on each head
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 3. Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights`
        },
        {
          type: 'twoColumn' as const,
          props: {
            left: [
              {
                type: 'paragraph' as const,
                content: "**Why Multiple Heads?**"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Capture different types of relationships",
                    "Attend to different positions simultaneously",
                    "Increase model expressiveness",
                    "Parallel processing efficiency"
                  ]
                }
              }
            ],
            right: [
              {
                type: 'paragraph' as const,
                content: "**Head Specialization:**"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Syntactic relationships",
                    "Semantic similarities",
                    "Positional patterns",
                    "Long-range dependencies"
                  ]
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
        title: "Attention Variants",
        background: false
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Modern attention mechanisms include several sophisticated variants designed to address specific challenges:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            left: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Self-Attention",
                  icon: <Focus className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Query, key, and value come from the same sequence. Used in Transformers for encoding relationships within a sequence."
                  },
                  {
                    type: 'codeBlock' as const,
                    props: {
                      language: 'python',
                      title: "Self-Attention Usage"
                    },
                    content: `# Self-attention: Q, K, V from same input\noutput = self_attention(x, x, x)`
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Cross-Attention",
                  icon: <ArrowRight className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Query from one sequence, key and value from another. Used in encoder-decoder architectures for translation and summarization."
                  },
                  {
                    type: 'codeBlock' as const,
                    props: {
                      language: 'python',
                      title: "Cross-Attention Usage"
                    },
                    content: `# Cross-attention: Q from decoder, K,V from encoder\noutput = cross_attention(decoder_state, encoder_output, encoder_output)`
                  }
                ]
              }
            ],
            right: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'warning' as const,
                  title: "Masked Attention",
                  icon: <Eye className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Prevents attention to future tokens in autoregressive models. Essential for maintaining causal relationships in language modeling."
                  },
                  {
                    type: 'codeBlock' as const,
                    props: {
                      language: 'python',
                      title: "Causal Mask"
                    },
                    content: `# Create causal mask for autoregressive generation\nmask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)\nmasked_scores = scores.masked_fill(mask == 1, -1e9)`
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
        title: "Advanced Attention Mechanisms",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Recent advances have introduced more efficient and powerful attention mechanisms:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            left: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Sparse Attention",
                  icon: <Network className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Reduces computational complexity by attending to only a subset of positions. Used in models like Sparse Transformer and Longformer."
                  },
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Local attention patterns",
                        "Strided attention",
                        "Global attention tokens",
                        "O(n√n) complexity"
                      ]
                    }
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Linear Attention",
                  icon: <Zap className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Achieves linear complexity through kernel methods and feature maps. Examples include Performer and Linear Transformer."
                  },
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Kernel-based approximation",
                        "Feature map decomposition",
                        "O(n) complexity",
                        "Maintained performance"
                      ]
                    }
                  }
                ]
              }
            ],
            right: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'warning' as const,
                  title: "Sliding Window",
                  icon: <Search className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Attends to a fixed-size window around each position. Balances locality and computational efficiency."
                  },
                  {
                    type: 'codeBlock' as const,
                    props: {
                      language: 'python',
                      title: "Sliding Window Implementation"
                    },
                    content: `def sliding_window_attention(query, key, value, window_size):\n    seq_len = query.size(1)\n    attention_mask = torch.zeros(seq_len, seq_len)\n    \n    for i in range(seq_len):\n        start = max(0, i - window_size // 2)\n        end = min(seq_len, i + window_size // 2 + 1)\n        attention_mask[i, start:end] = 1\n    \n    return attention(query, key, value, attention_mask)`
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
        title: "Applications & Use Cases",
        background: false
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Attention mechanisms have found applications across numerous domains in machine learning:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            left: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Natural Language Processing",
                  icon: <MessageSquare className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Machine Translation (Google Translate)",
                        "Language Modeling (GPT, BERT)",
                        "Text Summarization",
                        "Question Answering",
                        "Sentiment Analysis"
                      ]
                    }
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Computer Vision",
                  icon: <Eye className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Vision Transformers (ViT)",
                        "Image Captioning",
                        "Object Detection (DETR)",
                        "Semantic Segmentation",
                        "Visual Question Answering"
                      ]
                    }
                  }
                ]
              }
            ],
            right: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'warning' as const,
                  title: "Multimodal Tasks",
                  icon: <Layers className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Image-Text Matching",
                        "Visual Grounding",
                        "Cross-modal Retrieval",
                        "Multimodal Summarization",
                        "Audio-Visual Learning"
                      ]
                    }
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Other Domains",
                  icon: <Network className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Speech Recognition",
                        "Recommendation Systems",
                        "Graph Neural Networks",
                        "Time Series Analysis",
                        "Reinforcement Learning"
                      ]
                    }
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
        title: "Implementation Best Practices",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "When implementing attention mechanisms, consider these best practices for optimal performance:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            left: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Optimization Techniques",
                  icon: <TrendingUp className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Use gradient clipping for stable training",
                        "Apply dropout to attention weights",
                        "Implement layer normalization",
                        "Use mixed precision training",
                        "Cache key-value pairs for inference"
                      ]
                    }
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'warning' as const,
                  title: "Common Pitfalls",
                  icon: <AlertTriangle className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Forgetting to scale attention scores",
                        "Not masking padding tokens",
                        "Incorrect dimension handling",
                        "Memory inefficient implementations",
                        "Ignoring attention weight sparsity"
                      ]
                    }
                  }
                ]
              }
            ],
            right: [
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  title: "Efficient Attention Implementation"
                },
                content: `import torch
import torch.nn as nn

class EfficientAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Pre-layer norm for better convergence
        normed_x = self.layer_norm(x)
        
        # Self-attention with residual connection
        attn_output, weights = self.attention(normed_x, normed_x, normed_x, mask)
        attn_output = self.dropout(attn_output)
        
        # Residual connection
        output = x + attn_output
        
        return output, weights`
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Future Directions",
        background: false
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Attention mechanisms continue to evolve with new research directions focusing on efficiency, interpretability, and novel applications:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            left: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Emerging Trends",
                  icon: <TrendingUp className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Mixture of Experts attention",
                        "Learnable sparse patterns",
                        "Attention with memory",
                        "Cross-domain attention transfer",
                        "Quantum attention mechanisms"
                      ]
                    }
                  }
                ]
              }
            ],
            right: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Research Challenges",
                  icon: <Brain className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Scaling to very long sequences",
                        "Reducing computational complexity",
                        "Improving interpretability",
                        "Handling multi-scale attention",
                        "Efficient hardware implementations"
                      ]
                    }
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
        title: "Learning Resources",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Comprehensive resources to deepen your understanding of attention mechanisms:"
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
                      description: "Vaswani et al. (2017) - The foundational Transformer paper"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Neural Machine Translation by Jointly Learning to Align and Translate",
                      description: "Bahdanau et al. (2014) - Introduction of attention in NMT"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Show, Attend and Tell",
                      description: "Xu et al. (2015) - Attention for image captioning"
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
                      title: "Hugging Face Transformers",
                      description: "Comprehensive library with pre-trained models"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "The Annotated Transformer",
                      description: "Harvard NLP step-by-step implementation guide"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "CS224N Stanford Course",
                      description: "Natural Language Processing with Deep Learning"
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
    colorScheme: 'yellow' as const
  }
};

export default function AttentionTopicPage() {
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
        <TopicPageBuilder {...attentionTopicData} />
      </article>
    </div>
  );
}
