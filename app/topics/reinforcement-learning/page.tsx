import { Gamepad2, Brain, Target, TrendingUp, Zap, BarChart3, Settings, Trophy, Search, BookOpen, Code, Video, ExternalLink } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Reinforcement Learning - ML Portfolio',
  description: 'Machine learning paradigm where agents learn through interaction with an environment',
};

const reinforcementLearningTopicData = {
  title: "Reinforcement Learning",
  header: {
    category: "Machine Learning",
    difficulty: "Advanced" as const,
    readTime: "10 min read",
    description: "Machine learning paradigm where agents learn optimal behaviors through trial and error interactions with an environment to maximize cumulative rewards",
    relatedProjects: ["DQN Flappy Bird"],
    gradientFrom: "from-green-50 to-blue-50",
    gradientTo: "dark:from-green-900/20 dark:to-blue-900/20",
    borderColor: "border-green-200 dark:border-green-800"
  },
  tags: {
    items: ['RL', 'Gaming', 'Neural Networks', 'Decision Making', 'Q-Learning'],
    colorScheme: 'green' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What is Reinforcement Learning?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties and aims to learn a strategy (policy) that maximizes the cumulative reward over time."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Key Insight",
            icon: <Brain className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Unlike supervised learning where we learn from labeled examples, RL learns from the consequences of actions. This makes it particularly suitable for sequential decision-making problems where the optimal action depends on the current state and future possibilities."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Core Components"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Reinforcement Learning systems consist of several key components that define the learning process:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "The RL Framework"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Agent: The learner that makes decisions",
                    "Environment: The world the agent interacts with",
                    "State: Current situation of the agent",
                    "Action: Decisions the agent can make",
                    "Reward: Feedback signal from environment",
                    "Policy: Strategy for choosing actions"
                  ]
                }
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "The RL Loop"
              },
              {
                type: 'paragraph' as const,
                content: "The agent-environment interaction follows a continuous cycle:"
              },
              {
                type: 'list' as const,
                props: {
                  ordered: true,
                  items: [
                    "Agent observes current state",
                    "Agent selects action based on policy",
                    "Environment transitions to new state",
                    "Environment provides reward signal",
                    "Agent updates policy based on experience"
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
        title: "Q-Learning Algorithm",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Q-Learning is one of the most fundamental RL algorithms. It learns the quality (Q-value) of state-action pairs without needing a model of the environment:"
        },
        {
          type: 'codeBlock' as const,
          props: {
            language: 'python',
            filename: 'q_learning.py'
          },
          content: `import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-learning update rule
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Example usage
agent = QLearningAgent(state_size=100, action_size=4)

# Training loop
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}: Total Reward = {total_reward}")`
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Deep Q-Networks (DQN)"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "When dealing with large or continuous state spaces, traditional Q-tables become impractical. Deep Q-Networks use neural networks to approximate Q-values:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "DQN Architecture"
              },
              {
                type: 'paragraph' as const,
                content: "DQN replaces the Q-table with a neural network that takes states as input and outputs Q-values for all possible actions."
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Input: State representation (e.g., game screen)",
                    "Hidden layers: Feature extraction",
                    "Output: Q-values for each action",
                    "Loss: MSE between predicted and target Q-values"
                  ]
                }
              }
            ],
            right: [
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'dqn_network.py'
                },
                content: `import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# For image inputs (like Atari games)
class ConvDQN(nn.Module):
    def __init__(self, action_size):
        super(ConvDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)`
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "DQN Training Process",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Training a DQN involves several key techniques to ensure stable learning:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Experience Replay"
              },
              {
                type: 'paragraph' as const,
                content: "Store experiences in a replay buffer and sample random batches for training to break correlation between consecutive experiences."
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'replay_buffer.py'
                },
                content: `from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)`
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Target Network"
              },
              {
                type: 'paragraph' as const,
                content: "Use a separate target network with frozen weights to compute target Q-values, updating it periodically for stable training."
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'target_network.py'
                },
                content: `# Target network update
def update_target_network(main_net, target_net):
    target_net.load_state_dict(main_net.state_dict())

# Training with target network
def compute_loss(batch, main_net, target_net, gamma=0.99):
    states, actions, rewards, next_states, dones = batch
    
    current_q_values = main_net(states).gather(1, actions)
    
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))
    
    return F.mse_loss(current_q_values.squeeze(), target_q_values)`
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "RL Algorithms Comparison"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Different RL algorithms are suited for different types of problems. Here's a comparison of popular approaches:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Value-Based Methods",
                  icon: <BarChart3 className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Q-Learning: Learns action-value function",
                        "DQN: Deep neural network Q-learning",
                        "Double DQN: Reduces overestimation bias",
                        "Dueling DQN: Separates value and advantage"
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
                  variant: 'success' as const,
                  title: "Policy-Based Methods",
                  icon: <Target className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "REINFORCE: Basic policy gradient",
                        "Actor-Critic: Combines value and policy",
                        "PPO: Proximal Policy Optimization",
                        "A3C: Asynchronous Advantage Actor-Critic"
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
        title: "Exploration vs Exploitation",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "One of the fundamental challenges in RL is balancing exploration (trying new actions to discover better strategies) with exploitation (using known good actions):"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Exploration Strategies"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "ε-Greedy: Random action with probability ε",
                    "Upper Confidence Bound (UCB): Optimistic selection",
                    "Thompson Sampling: Bayesian approach",
                    "Curiosity-driven: Intrinsic motivation methods"
                  ]
                }
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'exploration.py'
                },
                content: `# Epsilon-greedy with decay
def epsilon_greedy_action(q_values, epsilon):
    if random.random() < epsilon:
        return random.randint(0, len(q_values) - 1)
    else:
        return np.argmax(q_values)

# UCB exploration
def ucb_action(q_values, counts, t, c=2):
    ucb_values = q_values + c * np.sqrt(np.log(t) / (counts + 1e-5))
    return np.argmax(ucb_values)`
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Exploitation Considerations"
              },
              {
                type: 'paragraph' as const,
                content: "As the agent learns, it should gradually shift from exploration to exploitation of the learned policy."
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'warning' as const,
                  title: "Common Pitfall"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Too much exploration leads to poor performance, while too little exploration can cause the agent to get stuck in suboptimal policies."
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
        title: "Applications and Examples"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Reinforcement Learning has found success in various domains, from games to real-world applications:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Gaming and Simulations",
                  icon: <Gamepad2 className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Atari games: DQN breakthrough",
                        "Go: AlphaGo and AlphaZero",
                        "StarCraft II: Complex strategy games",
                        "Poker: Pluribus multi-player"
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
                  variant: 'success' as const,
                  title: "Real-World Applications",
                  icon: <Trophy className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Robotics: Manipulation and navigation",
                        "Autonomous vehicles: Path planning",
                        "Finance: Algorithmic trading",
                        "Healthcare: Treatment optimization",
                        "Energy: Grid optimization",
                        "Recommendation systems: Content selection"
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
        title: "Challenges and Limitations",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "While powerful, RL comes with several challenges that researchers and practitioners must address:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Technical Challenges"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Sample efficiency: Requires many interactions",
                    "Reward design: Difficult to specify good rewards",
                    "Partial observability: Incomplete state information",
                    "Non-stationarity: Environment changes over time",
                    "Continuous action spaces: Infinite action possibilities"
                  ]
                }
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Practical Considerations"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Computational cost: Training can be expensive",
                    "Safety: Learning through trial and error",
                    "Reproducibility: High variance in results",
                    "Transfer learning: Adapting to new domains",
                    "Interpretability: Understanding learned policies"
                  ]
                }
              }
            ]
          }
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'warning' as const,
            title: "Key Takeaway",
            icon: <Zap className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "RL is powerful but requires careful consideration of the problem formulation, algorithm choice, and evaluation methodology. Start with simpler environments and gradually increase complexity."
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
          content: "Comprehensive resources to deepen your understanding of reinforcement learning:"
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
                      title: "Playing Atari with Deep Reinforcement Learning",
                      description: "Mnih et al. (2013) - Deep Q-Networks for Atari games"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Mastering the Game of Go with Deep Neural Networks",
                      description: "Silver et al. (2016) - AlphaGo and Monte Carlo Tree Search"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Proximal Policy Optimization Algorithms",
                      description: "Schulman et al. (2017) - PPO algorithm for policy gradient methods"
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
                      title: "Reinforcement Learning: An Introduction",
                      description: "Sutton & Barto's comprehensive textbook (free online)"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "OpenAI Gym",
                      description: "Standard toolkit for RL environment development and testing"
                    },
                    {
                      icon: <Video className="w-6 h-6" />,
                      title: "Deep RL Course - UC Berkeley",
                      description: "Comprehensive deep reinforcement learning course materials"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Stable Baselines3",
                      description: "Reliable implementations of RL algorithms in PyTorch"
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

export default function ReinforcementLearningTopicPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-green-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-green-600 dark:hover:text-green-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-green-600 dark:hover:text-green-400 transition-colors">
                Projects
              </a>
              <a href="/topics" className="text-green-600 dark:text-green-400 font-medium">
                Topics
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <TopicPageBuilder {...reinforcementLearningTopicData} />
      </article>
    </div>
  );
}
