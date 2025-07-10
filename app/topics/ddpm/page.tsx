import { Waves, Brain, Clock, Zap, Target, Camera, Search, Code, ExternalLink } from 'lucide-react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Denoising Diffusion Probabilistic Models (DDPM) - ML Portfolio',
  description: 'Probabilistic generative models that learn to reverse a diffusion process for high-quality image generation',
};

const ddpmData = {
  title: "Denoising Diffusion Probabilistic Models (DDPM)",
  header: {
    category: "Generative AI",
    difficulty: "Advanced" as const,
    readTime: "10 min read",
    description: "State-of-the-art probabilistic generative models that learn to reverse a gradual noising process to generate high-quality images",
    relatedProjects: ["Custom Diffusion Model"],
    gradientFrom: "from-violet-50 to-purple-50",
    gradientTo: "dark:from-violet-900/20 dark:to-purple-900/20",
    borderColor: "border-violet-200 dark:border-violet-800"
  },
  tags: {
    items: ['Generative AI', 'Probabilistic Models', 'Denoising', 'Diffusion'],
    colorScheme: 'purple' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What are DDPMs?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Denoising Diffusion Probabilistic Models (DDPMs) are a class of generative models that learn to generate data by reversing a gradual noising process. They work by learning to denoise images that have been corrupted with Gaussian noise, eventually learning to generate new samples from pure noise."
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
              content: "DDPMs revolutionized generative modeling by framing generation as a denoising process, achieving state-of-the-art results in image quality and sample diversity."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "The Diffusion Process"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Waves className="w-6 h-6" />,
                title: "Forward Process (Diffusion)",
                description: "Gradually adds Gaussian noise to data over T timesteps until it becomes pure noise.",
                color: "red"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Reverse Process (Denoising)",
                description: "Neural network learns to reverse the forward process, removing noise step by step.",
                color: "green"
              },
              {
                icon: <Camera className="w-6 h-6" />,
                title: "Generation",
                description: "Start with random noise and apply learned reverse process to generate new samples.",
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
        title: "Mathematical Foundation"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Forward Process",
            icon: <Clock className="w-6 h-6" />
          },
          children: [
            {
              type: 'math' as const,
              props: {block: true},
              content: "q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)"
            },
            {
              type: 'paragraph' as const,
              content: "Where:\n- β_t is the noise schedule\n- x_t is the noisy image at timestep t\n- x_0 is the original clean image"
            }
          ]
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Reverse Process",
            icon: <Target className="w-6 h-6" />
          },
          children: [
            {
              type: 'math' as const,
              props: {block: true},
              content: "p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))"
            },
            {
              type: 'paragraph' as const,
              content: "The neural network learns to predict μ_θ (mean) and optionally Σ_θ (variance)"
            },
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "DDPM Implementation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's a simplified implementation of DDPM using PyTorch:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """Basic building block with time conditioning"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Add time channel
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SimpleUNet(nn.Module):
    """Simplified U-Net for DDPM"""
    def __init__(self, image_channels=3, time_emb_dim=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, 64, 3, padding=1)
        
        # Downsample
        self.downs = nn.ModuleList([
            Block(64, 128, time_emb_dim),
            Block(128, 256, time_emb_dim),
            Block(256, 512, time_emb_dim),
        ])
        
        # Upsample
        self.ups = nn.ModuleList([
            Block(512, 256, time_emb_dim, up=True),
            Block(256, 128, time_emb_dim, up=True),
            Block(128, 64, time_emb_dim, up=True),
        ])
        
        # Final conv
        self.output = nn.Conv2d(64, image_channels, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv0(x)
        
        # Store residual connections
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        # Upsample
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        
        return self.output(x)

class DDPM:
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, model, beta_start=0.0001, beta_end=0.02, T=1000):
        self.model = model
        self.T = T
        
        # Create noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion (adding noise)"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None, loss_type="l1"):
        """Calculate training loss"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """Single denoising step"""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        # Use model to predict noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, shape):
        """Generate samples by reverse diffusion"""
        device = next(self.model.parameters()).device
        
        # Start from random noise
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.T)):
            img = self.p_sample(img, torch.full((shape[0],), i, device=device, dtype=torch.long), i)
        
        return img

# Example usage
model = SimpleUNet()
ddpm = DDPM(model)

# Training step
def training_step(ddpm, batch):
    batch_size = batch.shape[0]
    device = batch.device
    
    # Sample random timesteps
    t = torch.randint(0, ddpm.T, (batch_size,), device=device).long()
    
    # Calculate loss
    loss = ddpm.p_losses(batch, t, loss_type="l2")
    
    return loss

# Generate samples
with torch.no_grad():
    samples = ddpm.sample(shape=(4, 3, 64, 64))  # Generate 4 samples of 64x64 RGB images
    print(f"Generated samples shape: {samples.shape}")`,
          props: {
            language: 'python',
            title: 'DDPM Implementation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Training DDPM"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's how to train a DDPM model:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

def train_ddpm(ddpm, dataloader, num_epochs=100, lr=1e-4):
    """Train DDPM model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddpm.model.to(device)
    
    # Move noise schedule to device
    ddpm.betas = ddpm.betas.to(device)
    ddpm.alphas = ddpm.alphas.to(device)
    ddpm.alphas_cumprod = ddpm.alphas_cumprod.to(device)
    ddpm.alphas_cumprod_prev = ddpm.alphas_cumprod_prev.to(device)
    ddpm.sqrt_recip_alphas = ddpm.sqrt_recip_alphas.to(device)
    ddpm.sqrt_alphas_cumprod = ddpm.sqrt_alphas_cumprod.to(device)
    ddpm.sqrt_one_minus_alphas_cumprod = ddpm.sqrt_one_minus_alphas_cumprod.to(device)
    ddpm.posterior_variance = ddpm.posterior_variance.to(device)
    
    optimizer = optim.Adam(ddpm.model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            # Normalize to [-1, 1]
            data = (data - 0.5) * 2.0
            
            optimizer.zero_grad()
            loss = training_step(ddpm, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
        
        # Generate samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                samples = ddpm.sample(shape=(4, 3, 32, 32))
                # Denormalize from [-1, 1] to [0, 1]
                samples = (samples + 1.0) / 2.0
                samples = torch.clamp(samples, 0.0, 1.0)
                
                # Save samples
                save_samples(samples, f'samples_epoch_{epoch+1}.png')
    
    return losses

def save_samples(samples, filename):
    """Save generated samples as image grid"""
    import torchvision.utils as vutils
    
    grid = vutils.make_grid(samples, nrow=2, padding=2, normalize=False)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# Prepare dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Initialize and train model
model = SimpleUNet(image_channels=3)
ddpm = DDPM(model, T=1000)

# Train the model
losses = train_ddpm(ddpm, dataloader, num_epochs=50)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('DDPM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()`,
          props: {
            language: 'python',
            title: 'Training DDPM'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Advanced DDPM Techniques"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Improved Sampling",
            icon: <Zap className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "DDIM: Deterministic sampling for faster generation",
                  "Classifier Guidance: Using a classifier to guide generation",
                  "Classifier-free Guidance: Self-conditioning without external classifier",
                  "Progressive Distillation: Reducing sampling steps"
                ]
              }
            }
          ]
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Applications",
            icon: <Camera className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "Image generation and editing",
                  "Text-to-image synthesis (Stable Diffusion)",
                  "Image inpainting and super-resolution",
                  "3D content generation"
                ]
              }
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
          content: "Comprehensive resources to deepen your understanding of DDPM (Denoising Diffusion Probabilistic Models):"
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
                      title: "Denoising Diffusion Probabilistic Models",
                      description: "Ho et al. (2020) - Original DDPM paper introducing the framework"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Improved Denoising Diffusion Probabilistic Models",
                      description: "Nichol & Dhariwal (2021) - Key improvements to DDPM"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Diffusion Models Beat GANs on Image Synthesis",
                      description: "Dhariwal & Nichol (2021) - Comprehensive comparison with GANs"
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
                      icon: <Code className="w-6 h-6" />,
                      title: "Hugging Face Diffusers Library",
                      description: "Comprehensive DDPM implementations and pretrained models"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "OpenAI's Improved DDPM Repository",
                      description: "Official implementation from the researchers"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Annotated DDPM Tutorial",
                      description: "Step-by-step code walkthrough with explanations"
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

export default function DDPMPage() {
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
        <TopicPageBuilder {...ddpmData} />
      </article>
    </div>
  );
}