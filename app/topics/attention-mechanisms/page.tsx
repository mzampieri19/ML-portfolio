import TopicPageBuilder from '../../components/TopicPageBuilder';
import { Brain, Eye, Target, Code, BookOpen } from 'lucide-react';

export const metadata = {
  title: 'Attention Mechanisms - ML Portfolio',
  description: 'Neural network components that focus on relevant parts of input data - the foundation of modern NLP and computer vision models',
};

export default function AttentionMechanismsPage() {
  const data = {
    title: "Attention Mechanisms",
    header: {
      category: "Deep Learning",
      difficulty: "Intermediate" as const,
      readTime: "12 min read",
      description: "Neural network components that focus on relevant parts of input data, revolutionizing how models process sequences and images.",
      relatedProjects: ["custom-gpt-llm"],
      gradientFrom: "from-purple-500",
      gradientTo: "to-blue-500",
      borderColor: "border-purple-200"
    },
    tags: {
      items: ["Attention", "Focus", "Sequence Modeling", "NLP"],
      colorScheme: "purple" as const
    },
    blocks: [
      {
        type: 'section' as const,
        props: { title: 'What are Attention Mechanisms?' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Attention mechanisms are neural network components that allow models to focus on specific parts of input data when making predictions. Instead of processing all input equally, attention mechanisms compute weights that determine which parts of the input are most relevant for the current task. This concept has revolutionized fields like natural language processing and computer vision.'
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Core Concepts' },
        children: [
          {
            type: 'features' as const,
            props: {
              items: [
                {
                  title: 'Attention Weights',
                  description: 'Learned weights that determine how much focus to place on each part of the input'
                },
                {
                  title: 'Query, Key, Value',
                  description: 'Three vectors used in attention computation - query determines what to look for, keys are what can be attended to, values are what is actually retrieved'
                },
                {
                  title: 'Self-Attention',
                  description: 'Attention mechanism where a sequence attends to itself, allowing each position to attend to all positions in the sequence'
                }
              ]
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Attention Computation' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'The basic attention mechanism computes a weighted sum of values, where the weights are determined by the compatibility between queries and keys.'
          },
          {
            type: 'codeBlock' as const,
            props: {
              language: 'python',
              code: `import torch
import torch.nn.functional as F

def attention(query, key, value, mask=None):
    """
    Compute attention weights and apply them to values
    
    Args:
        query: [batch_size, seq_len_q, d_model]
        key: [batch_size, seq_len_k, d_model]
        value: [batch_size, seq_len_v, d_model]
        mask: optional mask to prevent attention to certain positions
    """
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Example: Self-attention layer
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query_projection = torch.nn.Linear(d_model, d_model)
        self.key_projection = torch.nn.Linear(d_model, d_model)
        self.value_projection = torch.nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        
        output, weights = attention(query, key, value)
        return output, weights`
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Applications and Use Cases' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Attention mechanisms have found applications across various domains:'
          },
          {
            type: 'list' as const,
            props: {
              items: [
                'Machine Translation: Aligning words between source and target languages',
                'Text Summarization: Focusing on important sentences or phrases',
                'Image Captioning: Attending to relevant parts of images while generating captions',
                'Speech Recognition: Focusing on relevant audio segments',
                'Question Answering: Attending to relevant parts of context when answering questions'
              ]
            }
          }
        ]
      },
      {
        type: 'section' as const,
        props: { title: 'Advanced Attention Mechanisms' },
        children: [
          {
            type: 'paragraph' as const,
            content: 'Modern attention mechanisms include several sophisticated variants:'
          },
          {
            type: 'section' as const,
            props: { title: 'Multi-Head Attention' },
            children: [
              {
                type: 'paragraph' as const,
                content: 'Uses multiple attention heads to capture different types of relationships simultaneously.'
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  code: `class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = attention(Q, K, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        return output, attention_weights`
                }
              }
            ]
          },
          {
            type: 'paragraph' as const,
            content: 'Other advanced mechanisms include Cross-Attention (where queries come from one sequence and keys/values from another) and Sparse Attention (which only computes attention for a subset of positions).'
          }
        ]
      }
    ]
  };

  return <TopicPageBuilder {...data} />;
}
