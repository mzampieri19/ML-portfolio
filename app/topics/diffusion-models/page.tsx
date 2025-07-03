import { Sparkles, Image, Brain, Layers, Target, TrendingUp, Zap, Code, BarChart3, Eye, ExternalLink } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';
import CodeBlock from '../../components/CodeBlock';

export const metadata = {
  title: 'Diffusion Models - ML Portfolio Topics',
  description: 'Learn about diffusion models - generative models that learn to reverse a gradual noising process',
};

const diffusionModelsPageData = {
  title: "Diffusion Models",
  header: {
    date: "Generative AI",
    readTime: "8 min read",
    description: "Generative models that learn to reverse a gradual noising process to create high-quality images",
    gradientFrom: "from-pink-50 to-purple-50",
    gradientTo: "dark:from-pink-900/20 dark:to-purple-900/20",
    borderColor: "border-pink-200 dark:border-pink-800",
    difficulty: "Advanced" as const,
    category: "Generative AI",
    relatedProjects: ["custom-diffusion-model"]
  },
  tags: {
    items: ['Generative AI', 'Image Generation', 'DDPM', 'Denoising', 'Neural Networks'],
    colorScheme: 'purple' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What are Diffusion Models?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Diffusion models are a class of generative models that learn to create data by reversing a gradual noising process. They work by first learning how to systematically add noise to data, then learning to reverse this process to generate new, high-quality samples from pure noise."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Key Insight",
            icon: <Sparkles className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Instead of generating data directly, diffusion models learn to gradually denoise random noise into coherent data, making the generation process more stable and controllable than other generative approaches."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "How Diffusion Models Work"
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
                content: "Forward Process (Noising)"
              },
              {
                type: 'paragraph' as const,
                content: "The forward process gradually adds Gaussian noise to the original data over T timesteps until it becomes pure noise. This process is mathematically defined and doesn't require learning."
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Gradual Corruption",
                      description: "Data is slowly corrupted with increasing noise levels"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Mathematical Process",
                      description: "Uses predefined noise schedule β₁, β₂, ..., βₜ"
                    },
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Tractable",
                      description: "Can sample any timestep directly without iteration"
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
                content: "Reverse Process (Denoising)"
              },
              {
                type: 'paragraph' as const,
                content: "The reverse process learns to remove noise step by step, transforming random noise back into coherent data. This is where the neural network learns to predict and remove noise."
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Neural Network",
                      description: "U-Net architecture predicts noise to be removed"
                    },
                    {
                      icon: <Layers className="w-6 h-6" />,
                      title: "Iterative Refinement",
                      description: "Step-by-step noise removal over multiple timesteps"
                    },
                    {
                      icon: <Sparkles className="w-6 h-6" />,
                      title: "Generation",
                      description: "Produces high-quality samples from pure noise"
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
        filename: "diffusion_forward_process.py",
        code: `import torch
import torch.nn.functional as F
import numpy as np

class DiffusionProcess:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def forward_diffusion(self, x0, t):
        """
        Add noise to clean image x0 at timestep t
        
        Args:
            x0: Clean image [batch_size, channels, height, width]
            t: Timestep [batch_size]
            
        Returns:
            xt: Noisy image at timestep t
            noise: The noise that was added
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - self.alpha_cumprod[t])
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1)
        
        # Apply noise: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return xt, noise`
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
          content: "Training a diffusion model involves teaching a neural network to predict the noise that was added to an image at any given timestep. The model learns to reverse the noising process by minimizing the difference between predicted and actual noise."
        },
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Sample Training Data",
                date: "Step 1",
                description: "Take a clean image from the training dataset"
              },
              {
                title: "Random Timestep",
                date: "Step 2", 
                description: "Randomly select a timestep t between 1 and T"
              },
              {
                title: "Add Noise",
                date: "Step 3",
                description: "Apply forward diffusion to add noise corresponding to timestep t"
              },
              {
                title: "Predict Noise",
                date: "Step 4",
                description: "Neural network predicts what noise was added"
              },
              {
                title: "Calculate Loss",
                date: "Step 5",
                description: "Compute MSE loss between predicted and actual noise"
              },
              {
                title: "Backpropagate",
                date: "Step 6",
                description: "Update model parameters to minimize prediction error"
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
        filename: "diffusion_training.py",
        code: `def train_step(model, dataloader, optimizer, diffusion, device):
    """
    Single training step for diffusion model
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        
        # Add noise to images
        noisy_images, noise = diffusion.forward_diffusion(images, t)
        
        # Predict noise
        predicted_noise = model(noisy_images, t)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Training loop
for epoch in range(num_epochs):
    avg_loss = train_step(model, train_loader, optimizer, diffusion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")`
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Types of Diffusion Models"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Brain className="w-6 h-6" />,
                title: "DDPM (Denoising Diffusion Probabilistic Models)",
                description: "Original formulation using Markov chain for gradual denoising"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "DDIM (Denoising Diffusion Implicit Models)",
                description: "Faster sampling with deterministic process, fewer steps needed"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Score-Based Models",
                description: "Learn score function (gradient of log probability) for generation"
              },
              {
                icon: <Sparkles className="w-6 h-6" />,
                title: "Latent Diffusion (Stable Diffusion)",
                description: "Operate in compressed latent space for efficiency and control"
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
        title: "Advantages and Applications",
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
                content: "Key Advantages"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "High Quality",
                      description: "Generate extremely high-quality, realistic images"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Stable Training",
                      description: "More stable training compared to GANs"
                    },
                    {
                      icon: <Layers className="w-6 h-6" />,
                      title: "Mode Coverage",
                      description: "Better coverage of data distribution"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "Controllable",
                      description: "Easier to condition and control generation"
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
                content: "Applications"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Image className="w-6 h-6" />,
                      title: "Image Generation",
                      description: "Create photorealistic images from text or noise"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Image Editing",
                      description: "Inpainting, outpainting, and style transfer"
                    },
                    {
                      icon: <Sparkles className="w-6 h-6" />,
                      title: "Super Resolution",
                      description: "Enhance image quality and resolution"
                    },
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Data Augmentation",
                      description: "Generate synthetic training data"
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
          content: "Explore these resources to deepen your understanding of diffusion models:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Research Papers"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Denoising Diffusion Probabilistic Models",
                      description: "Original DDPM paper by Ho et al. (2020)"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Improved Denoising Diffusion Probabilistic Models",
                      description: "Nichol & Dhariwal improvements (2021)"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "High-Resolution Image Synthesis with Latent Diffusion",
                      description: "Stable Diffusion paper by Rombach et al."
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
                      title: "Hugging Face Diffusers Library",
                      description: "Comprehensive implementation and tutorials"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Lil'Log: What are Diffusion Models?",
                      description: "Excellent mathematical explanation"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "The Annotated Diffusion Model",
                      description: "Step-by-step code walkthrough"
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

export default function DiffusionModelsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-pink-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-pink-600 dark:hover:text-pink-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-pink-600 dark:hover:text-pink-400 transition-colors">
                Projects
              </a>
              <a href="/topics" className="text-pink-600 dark:text-pink-400 font-medium">
                Topics
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <TopicPageBuilder {...diffusionModelsPageData} />
      </article>
    </div>
  );
}
