import { Brain, Gamepad2, Target, TrendingUp, Zap, Code, Settings, BarChart3, Trophy, Play, ArrowRight, ArrowLeft, Cpu, Database, GitBranch } from 'lucide-react';
import PageBuilder from '../../components/PageBuilder';
import CodeBlock from '../../components/CodeBlock';

export const metadata = {
  title: 'DQN Flappy Bird - ML Portfolio',
  description: 'Training an AI agent to master Flappy Bird using Deep Q-Network (DQN) reinforcement learning',
};

const dqnFlappyBirdPageData = {
  title: "DQN Flappy Bird",
  header: {
    date: "Summer 2025",
    readTime: "8 min read",
    description: "Training an AI agent to master Flappy Bird using Deep Q-Network (DQN) reinforcement learning",
    githubUrl: "https://github.com/mzampieri19/FlaapyBird-DQN",
    gradientFrom: "from-green-50 to-yellow-50",
    gradientTo: "dark:from-green-900/20 dark:to-yellow-900/20",
    borderColor: "border-green-200 dark:border-green-800"
  },
  tags: {
    items: ['Reinforcement Learning', 'From Scratch', 'Neural Network', 'DQN', 'Python', 'Pygame'],
    colorScheme: 'green' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "Project Overview",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "This project implements a Deep Q-Network (DQN) agent capable of learning to play games through reinforcement learning. The agent learns optimal strategies by interacting with environments and receiving rewards for good actions while being penalized for poor ones."
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Gamepad2 className="w-6 h-6" />,
                title: "CartPole-v1",
                description: "Balance a pole on a cart by moving left or right"
              },
              {
                icon: <Play className="w-6 h-6" />,
                title: "FlappyBird-v0",
                description: "Navigate a bird through pipes by controlling vertical movement"
              },
              {
                icon: <ArrowRight className="w-6 h-6" />,
                title: "More Games",
                description: "Additional environments can be easily added"
              }
            ],
            columns: 3
          }
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Advanced DQN Techniques",
            icon: <Brain className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "The implementation includes Double DQN to reduce overestimation bias, Dueling DQN for separating state value from action advantages, Experience Replay for stable training, and Target Networks for training stabilization."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Key Achievements"
      },
      children: [
        {
          type: 'metrics' as const,
          props: {
            metrics: [
              { label: "Final Average Score", value: "120+", change: "Superhuman", trend: "up" },
              { label: "Best Single Game", value: "500+", change: "Points", trend: "up" },
              { label: "Training Episodes", value: "1,800", change: "Total", trend: "neutral" },
              { label: "Success Rate", value: "95%", change: "Score > 50", trend: "up" }
            ],
            columns: 4
          }
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Autonomous Gameplay",
                description: "Agent successfully plays both CartPole and Flappy Bird without human input"
              },
              {
                icon: <Code className="w-6 h-6" />,
                title: "From Scratch Implementation",
                description: "Custom DQN implementation without using high-level RL libraries"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Advanced Techniques",
                description: "Implements Double DQN and Dueling DQN architectures"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Performance Tracking",
                description: "Real-time training progress and comprehensive logging"
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
        title: "Game Environment",
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
                content: "Flappy Bird Mechanics"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Play className="w-6 h-6" />,
                      title: "Bird Control",
                      description: "Controlled by neural network decisions (jump or no jump)"
                    },
                    {
                      icon: <ArrowLeft className="w-6 h-6" />,
                      title: "Pipe Obstacles",
                      description: "Obstacles that move from right to left"
                    },
                    {
                      icon: <Trophy className="w-6 h-6" />,
                      title: "Scoring System",
                      description: "Points earned for successfully passing through pipes"
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
                content: "Environment Setup"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Gymnasium Integration"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "The game is initiated using the gymnasium library which sets up the game environment with proper state representation and reward structure."
                  }
                ]
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
{`# Initialize the game environment using gymnasium
env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

# Reset environment to get initial state
state = env.reset()

# Game loop
while not done:
    # Agent selects action based on current policy
    action = agent.select_action(state)
    
    # Environment executes action and returns new state, reward, done
    next_state, reward, done, info = env.step(action)
    
    # Store experience in replay buffer
    agent.store_experience(state, action, reward, next_state, done)
    
    # Update current state
    state = next_state
    
    # Train the agent periodically
    if len(agent.replay_buffer) > batch_size:
        agent.train()`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Key Concepts"
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
                content: "Reinforcement Learning"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Learning Process"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "The agent learns through trial and error by observing the current state, taking actions based on its policy, receiving rewards, and learning to maximize future rewards over time."
                  }
                ]
              },
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Q-Learning"
              },
              {
                type: 'paragraph' as const,
                content: "Q-Learning estimates the 'quality' (Q-value) of taking each possible action in any given state. The agent learns which actions lead to the highest cumulative rewards."
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Deep Neural Networks"
              },
              {
                type: 'paragraph' as const,
                content: "Instead of storing Q-values in a table (impossible for complex states), a neural network approximates Q-values for any state-action pair."
              },
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Exploration vs Exploitation"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'warning' as const,
                  title: "Balance Strategy"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Exploration involves taking random actions to discover new strategies (high epsilon), while exploitation uses learned knowledge to take the best known actions (low epsilon). The agent starts exploring randomly and gradually shifts to exploiting learned knowledge."
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
        title: "Methodology",
        background: true
      },
      children: [
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Experience Collection",
                date: "Step 1",
                description: "Agent interacts with environment using epsilon-greedy strategy. With probability ε, take random action (exploration); with probability 1-ε, take best known action (exploitation)."
              },
              {
                title: "Experience Storage",
                date: "Step 2",
                description: "All experiences (state, action, reward, next_state, done) are stored in replay memory buffer to learn from past experiences and break correlations."
              },
              {
                title: "Network Training",
                date: "Step 3",
                description: "Agent samples random batches from replay memory and trains neural network to predict Q-values and minimize difference between predicted and target Q-values."
              },
              {
                title: "Target Network Updates",
                date: "Step 4",
                description: "Separate 'target network' provides stable Q-value targets during training. Updated less frequently than main network to prevent unstable learning."
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Model Architecture"
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
                content: "Standard DQN Architecture"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Simple Pipeline"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Input Layer (State) → Hidden Layer (128-512 nodes) → Output Layer (Q-values for each action)"
                  }
                ]
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Dueling DQN Architecture"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Dual Stream Design"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Separates Q-value estimation into Value Stream V(s) for state goodness and Advantage Stream A(s,a) for action quality. Combined as Q(s,a) = V(s) + A(s,a) - mean(A(s,·))"
                  }
                ]
              }
            ]
          }
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Target className="w-6 h-6" />,
                title: "Value Stream V(s)",
                description: "Estimates how good it is to be in a particular state"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "Advantage Stream A(s,a)",
                description: "Estimates how much better each action is compared to the average"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Faster Learning",
                description: "Separation helps agent learn faster, especially when most actions have similar values"
              }
            ],
            columns: 3
          }
        }
      ]
    },
    {
      type: 'custom' as const,
      component: (
        <CodeBlock language="python">
{`class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        # Flag to enable dueling DQN architecture
        self.enable_dueling_dqn = enable_dueling_dqn

        # First fully connected layer: input state to hidden representation
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Dueling DQN: separate streams for value and advantage

            # Value stream: hidden layer and output for state-value V(s)
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # Advantage stream: hidden layer and output for advantages A(s, a)
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            # Standard DQN: single output layer for Q-values
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # Pass input through first hidden layer with ReLU activation
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Dueling DQN forward pass

            # Value stream
            v = F.relu(self.fc_value(x))
            V = self.value(v)  # Shape: (batch_size, 1)

            # Advantage stream
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)  # Shape: (batch_size, action_dim)

            # Combine value and advantage streams to get Q-values
            # Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            # Standard DQN forward pass: directly output Q-values
            Q = self.output(x)

        return Q`}
        </CodeBlock>
      )
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
          content: "The training process involves the agent playing many episodes of Flappy Bird, learning from its mistakes and gradually improving its strategy through experience replay and target network updates."
        }
      ]
    },
    {
      type: 'custom' as const,
      component: (
        <CodeBlock language="python">
{`def optimize(self, mini_batch, policy_dqn, target_dqn):
    states, actions, new_states, rewards, terminations = zip(*mini_batch)

    # Stack tensors for batch processing
    states = torch.stack(states)
    actions = torch.stack(actions)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)
    terminations = torch.tensor(terminations).float().to(device)

    with torch.no_grad():
        # Compute target Q-values using Double DQN or standard DQN
        if self.enable_double_dqn:
            # Double DQN: use policy network to select actions, target network to evaluate
            best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
            target_q = rewards + (1-terminations) * self.discount_factor_g * \\
                            target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
        else:
            # Standard DQN: use target network for both action selection and evaluation
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

    # Compute current Q-values for taken actions
    current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
    
    # Calculate loss (Mean Squared Error between current and target Q-values)
    loss = self.loss_fn(current_q, target_q)

    # Backpropagation and optimizer step
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    return loss.item()`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Results and Performance"
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
                content: "Training Progress"
              },
              {
                type: 'timeline' as const,
                props: {
                  items: [
                    {
                      title: "Episodes 0-500",
                      date: "Phase 1",
                      description: "Random exploration, average score < 5"
                    },
                    {
                      title: "Episodes 500-1000",
                      date: "Phase 2",
                      description: "Learning basic game mechanics, average score 10-20"
                    },
                    {
                      title: "Episodes 1000-1500",
                      date: "Phase 3",
                      description: "Consistent improvement, average score 50-80"
                    },
                    {
                      title: "Episodes 1500+",
                      date: "Phase 4",
                      description: "Mastery achieved, average score > 100"
                    }
                  ]
                }
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Learning Insights"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Epsilon-Greedy Strategy",
                      description: "Crucial for balancing learning and performance"
                    },
                    {
                      icon: <Database className="w-6 h-6" />,
                      title: "Experience Replay",
                      description: "Significantly improved learning stability"
                    },
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "Target Network",
                      description: "Prevented training instability"
                    },
                    {
                      icon: <Trophy className="w-6 h-6" />,
                      title: "Reward Shaping",
                      description: "Encouraged focus on survival and scoring"
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
        title: "Technical Challenges",
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
                content: "State Representation"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'error' as const,
                  title: "Initial Attempt"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Raw pixel input proved too complex for the neural network to learn effectively from."
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Final Solution"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Simplified 4-dimensional state vector with key features: bird position, velocity, distance to next pipe, and height difference."
                  }
                ]
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Code Structure"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "dqn.py",
                      description: "DQN neural network architecture and forward pass"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "agent.py",
                      description: "Training loop, evaluation, and setup logic"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "/runs",
                      description: "Reports on current training run progress"
                    },
                    {
                      icon: <Database className="w-6 h-6" />,
                      title: "models/",
                      description: "Saved model checkpoints for inference"
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
        title: "Conclusion"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "This project successfully demonstrates the power of reinforcement learning in game environments. The DQN agent learned to play Flappy Bird from scratch, achieving superhuman performance through trial and error learning."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Key RL Concepts Demonstrated",
            icon: <Trophy className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "The implementation showcases Q-learning and temporal difference learning, experience replay and target networks, exploration vs exploitation strategies, and neural network function approximation. The agent's ability to master the game without prior knowledge highlights the potential of reinforcement learning for solving complex sequential decision problems."
            }
          ]
        }
      ]
    }
  ],
  navigation: {
    colorScheme: 'green' as const
  }
};

export default function DqnFlappyBirdPage() {
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
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <PageBuilder {...dqnFlappyBirdPageData} />
      </article>
    </div>
  );
}
