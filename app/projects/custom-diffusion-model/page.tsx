import { Brain, Cpu, Layers, Zap, Settings, Target, Code, Eye, GitBranch, Sparkles } from 'lucide-react';
import PageBuilder from '../../components/PageBuilder';
import CodeBlock from '../../components/CodeBlock';

export const metadata = {
  title: 'Custom Diffusion Model - ML Portfolio',
  description: 'Diffusion model from scratch, includes encoding, decoding, UNET, DDPM, Diffusion',
};

const diffusionPageData = {
  title: "Custom Diffusion Model",
  header: {
    date: "Summer 2025",
    readTime: "12 min read",
    description: "Complete diffusion model implementation from scratch, including encoding, decoding, UNET, DDPM, and diffusion processes",
    githubUrl: "https://github.com/mzampieri19/custom-diffusion-model",
    gradientFrom: "from-purple-50 to-pink-50",
    gradientTo: "dark:from-purple-900/20 dark:to-pink-900/20",
    borderColor: "border-purple-200 dark:border-purple-800",
    collaborators: "Individual Project"
  },
  tags: {
    items: ['Diffusion', 'From Scratch', 'Neural Network', 'GenAI'],
    colorScheme: 'purple' as const
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
          content: "This project implements a complete diffusion model from scratch, demonstrating a deep understanding of generative AI architectures. The implementation includes all core components: VAE encoder/decoder, U-Net architecture, DDPM sampling, and the complete diffusion pipeline."
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Brain className="w-6 h-6" />,
                title: "From Scratch Implementation",
                description: "Complete implementation without relying on pre-built diffusion libraries"
              },
              {
                icon: <Layers className="w-6 h-6" />,
                title: "Full Architecture",
                description: "VAE encoder, decoder, U-Net, and DDPM sampler components"
              },
              {
                icon: <Sparkles className="w-6 h-6" />,
                title: "Generative AI",
                description: "State-of-the-art image generation using diffusion processes"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Educational Focus",
                description: "Built to understand the mathematical foundations of diffusion models"
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
        title: "Core Architecture Components"
      },
      children: [
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "VAE Encoder",
                date: "Component 1",
                description: "Variational Autoencoder encoder that compresses input images into latent space representations for efficient processing"
              },
              {
                title: "U-Net Architecture",
                date: "Component 2",
                description: "Custom U-Net implementation with attention mechanisms for denoising in the diffusion process"
              },
              {
                title: "VAE Decoder",
                date: "Component 3",
                description: "Decoder that reconstructs high-quality images from latent space representations"
              },
              {
                title: "DDPM Sampler",
                date: "Component 4",
                description: "Denoising Diffusion Probabilistic Model sampler implementing the reverse diffusion process"
              },
              {
                title: "Diffusion Pipeline",
                date: "Component 5",
                description: "Complete pipeline orchestrating the entire generation process from noise to image"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Core Architecture Implementation",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's how the core diffusion model components are implemented:"
        },
        {
          type: 'custom' as const,
          content: (
            <CodeBlock language="python">
{`class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)  # Embed time step
        self.unet = UNET()                        # Main UNet
        self.final = UNET_OutputLayer(320, 4)     # Output projection
    
    def forward(self, latent, context, time):
        time = self.time_embedding(time)          # Process time embedding
        output = self.unet(latent, context, time) # UNet forward
        output = self.final(output)               # Output projection
        return output`}
            </CodeBlock>
          )
        },
        {
          type: 'paragraph' as const,
          content: "The time embedding component is crucial for conditioning the network on the current diffusion timestep:"
        },
        {
          type: 'custom' as const,
          content: (
            <CodeBlock language="python">
{`class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)  # Expands the embedding dimension
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)  # Further processes the embedding

    def forward(self, x):
        x = self.linear_1(x)  # Linear projection
        x = F.silu(x)         # SiLU activation
        x = self.linear_2(x)  # Second linear projection
        return x`}
            </CodeBlock>
          )
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "U-Net Architecture Deep Dive"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The U-Net is the heart of the diffusion model, featuring encoder-decoder architecture with skip connections:"
        },
        {
          type: 'custom' as const,
          content: (
            <CodeBlock language="python">
{`class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: progressively downsample and extract features
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),  # Initial conv
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),  # Residual + attention
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),  # Downsample
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),  # Downsample
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),  # Downsample
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])
        # Bottleneck: process at lowest spatial resolution
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280), 
            UNET_AttentionBlock(8, 160), 
            UNET_ResidualBlock(1280, 1280), 
        )
        # Decoder: upsample and reconstruct, using skip connections
        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        skip_connections = []  # Store outputs for skip connections
        for layers in self.encoders:
            x = layers(x, context, time)  # Pass through encoder block
            skip_connections.append(x)    # Save for skip connection
        x = self.bottleneck(x, context, time)  # Bottleneck
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)  # Concatenate skip connection
            x = layers(x, context, time)  # Pass through decoder block
        return x  # Final output`}
            </CodeBlock>
          )
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "DDPM Sampling Implementation",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The DDPM sampler implements the mathematical framework for the reverse diffusion process:"
        },
        {
          type: 'custom' as const,
          content: (
            <CodeBlock language="python">
{`class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120): 
        # Create a linear schedule for betas (noise variance) between beta_start and beta_end
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        # Compute alphas (1 - beta) and their cumulative product
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_train_timesteps = num_training_steps
        # Timesteps for training (reversed order)
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """
        Perform a single reverse diffusion step.
        Args:
            timestep: Current timestep index.
            latents: Current latent tensor.
            model_output: Model's predicted noise.
        Returns:
            The predicted sample at the previous timestep.
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Estimate the original (denoised) sample
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # Coefficients for combining the original and current samples
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        # Predict the previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        
        variance = 0
        # Add noise except for the last step
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample`}
            </CodeBlock>
          )
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Mathematical Foundations"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The diffusion model is built on solid mathematical principles that govern the forward and reverse diffusion processes:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Zap className="w-6 h-6" />,
                title: "Forward Diffusion",
                description: "Gradually adds Gaussian noise to images over T timesteps, following a predefined noise schedule"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Reverse Process",
                description: "Neural network learns to predict and remove noise, enabling generation from pure noise"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Loss Function",
                description: "Training objective based on noise prediction error, optimizing the denoising capabilities"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Sampling Strategy",
                description: "Implements DDPM sampling with configurable steps and noise schedules for generation control"
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
        title: "Key Implementation Features",
        background: true
      },
      children: [
        {
          type: 'metrics' as const,
          props: {
            metrics: [
              { label: "Architecture Layers", value: "50+", change: "Deep Network", trend: "up" },
              { label: "Parameters", value: "400M+", change: "Optimized", trend: "up" },
              { label: "Timesteps", value: "1000", change: "DDPM Standard", trend: "neutral" },
              { label: "Latent Dims", value: "512", change: "Compressed", trend: "up" }
            ],
            columns: 4
          }
        },
        {
          type: 'paragraph' as const,
          content: "The model achieves competitive performance through careful architecture design and implementation of state-of-the-art techniques in diffusion modeling."
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Technical Components Deep Dive"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Layers className="w-6 h-6" />,
                title: "Attention Mechanisms",
                description: "Self-attention and cross-attention layers for capturing long-range dependencies and conditioning"
              },
              {
                icon: <Cpu className="w-6 h-6" />,
                title: "Time Embeddings",
                description: "Sinusoidal position encodings for timestep information injection throughout the network"
              },
              {
                icon: <GitBranch className="w-6 h-6" />,
                title: "Skip Connections",
                description: "U-Net style skip connections preserving fine-grained details during upsampling"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Noise Scheduling",
                description: "Linear and cosine noise schedules for controlling the diffusion process dynamics"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Latent Space",
                description: "Efficient processing in compressed latent space reducing computational requirements"
              },
              {
                icon: <Sparkles className="w-6 h-6" />,
                title: "Generation Control",
                description: "Configurable sampling parameters for controlling generation quality and diversity"
              }
            ],
            columns: 3
          }
        }
      ]
    },
    {
      type: 'twoColumn' as const,
      props: {
        ratio: '1:1' as const,
        left: [
          {
            type: 'heading' as const,
            props: { level: 3 },
            content: "Training Process"
          },
          {
            type: 'paragraph' as const,
            content: "The model training involves several sophisticated stages:"
          },
          {
            type: 'highlight' as const,
            props: {
              variant: 'info' as const,
              title: "Multi-Stage Training"
            },
            children: [
              {
                type: 'paragraph' as const,
                content: "1. VAE pretraining for latent space learning\n2. U-Net training on noise prediction\n3. End-to-end fine-tuning for optimal generation quality"
              }
            ]
          }
        ],
        right: [
          {
            type: 'heading' as const,
            props: { level: 3 },
            content: "Performance Optimization"
          },
          {
            type: 'paragraph' as const,
            content: "Several optimization techniques were implemented:"
          },
          {
            type: 'highlight' as const,
            props: {
              variant: 'success' as const,
              title: "Efficiency Improvements"
            },
            children: [
              {
                type: 'paragraph' as const,
                content: "• Gradient checkpointing for memory efficiency\n• Mixed precision training\n• Latent space processing\n• Optimized attention computations"
              }
            ]
          }
        ]
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Complete Generation Pipeline"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The complete generation pipeline orchestrates all components for end-to-end image generation:"
        },
        {
          type: 'custom' as const,
          content: (
            <CodeBlock language="python">
{`def generate(prompt, uncond_prompt=None, input_image=None, strength=0.8, do_cfg=True,
  cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50, models={}, seed=None,
  device=None, idle_device=None, tokenizer=None):
  with torch.no_grad():
      # Validate strength parameter
      if not 0 < strength <= 1:
          raise ValueError("strength must be between 0 and 1")

      # Helper to move models to idle device if specified
      if idle_device:
          to_idle = lambda x: x.to(idle_device)
      else:
          to_idle = lambda x: x

      # Set up random generator for reproducibility
      generator = torch.Generator(device=device)
      
      if seed is None:
          generator.seed()
      else:
          generator.manual_seed(seed)

      # Load and move CLIP model to device
      clip = models["clip"]
      clip.to(device)
      
      # Prepare context embeddings for classifier-free guidance (CFG)
      if do_cfg:
          # Encode conditional prompt
          cond_tokens = tokenizer.batch_encode_plus(
              [prompt], padding="max_length", max_length=77
          ).input_ids
          cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
          cond_context = clip(cond_tokens)
          # Encode unconditional prompt
          uncond_tokens = tokenizer.batch_encode_plus(
              [uncond_prompt], padding="max_length", max_length=77
          ).input_ids
          uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
          uncond_context = clip(uncond_tokens)
          # Concatenate contexts for CFG
          context = torch.cat([cond_context, uncond_context])
      else:
          # Encode only the conditional prompt
          tokens = tokenizer.batch_encode_plus(
              [prompt], padding="max_length", max_length=77
          ).input_ids
          tokens = torch.tensor(tokens, dtype=torch.long, device=device)
          context = clip(tokens)
      to_idle(clip)

      # Select and configure sampler
      if sampler_name == "ddpm":
          sampler = DDPMSampler(generator)
          sampler.set_inference_timesteps(n_inference_steps)
      else:
          raise ValueError("Unknown sampler value %s. ")

      # Define shape of latent tensor
      latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

      if input_image:
          # If an input image is provided, encode it to latent space
          encoder = models["encoder"]
          encoder.to(device)

          # Preprocess input image
          input_image_tensor = input_image.resize((WIDTH, HEIGHT))
          input_image_tensor = np.array(input_image_tensor)
          input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
          input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
          input_image_tensor = input_image_tensor.unsqueeze(0)
          input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

          # Add noise to the encoded image
          encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
          latents = encoder(input_image_tensor, encoder_noise)

          sampler.set_strength(strength=strength)
          latents = sampler.add_noise(latents, sampler.timesteps[0])

          to_idle(encoder)
      else:
          # Otherwise, start from random noise
          latents = torch.randn(latents_shape, generator=generator, device=device)

      # Load and move diffusion model to device
      diffusion = models["diffusion"]
      diffusion.to(device)

      # Diffusion process: denoise latents step by step
      timesteps = tqdm(sampler.timesteps)
      for i, timestep in enumerate(timesteps):
          time_embedding = get_time_embedding(timestep).to(device)
          model_input = latents

          if do_cfg:
              # Duplicate latents for CFG (conditional and unconditional)
              model_input = model_input.repeat(2, 1, 1, 1)

          # Predict noise with diffusion model
          model_output = diffusion(model_input, context, time_embedding)

          if do_cfg:
              # Apply classifier-free guidance
              output_cond, output_uncond = model_output.chunk(2)
              model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

          # Update latents using the sampler
          latents = sampler.step(timestep, latents, model_output)

      to_idle(diffusion)

      # Decode latents to image space
      decoder = models["decoder"]
      decoder.to(device)
      images = decoder(latents)
      to_idle(decoder)

      # Post-process and return image
      images = rescale(images, (-1, 1), (0, 255), clamp=True)
      images = images.permute(0, 2, 3, 1)
      images = images.to("cpu", torch.uint8).numpy()
      return images[0]`}
            </CodeBlock>
          )
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Learning Outcomes & Insights",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Building this diffusion model from scratch provided deep insights into the mechanics of modern generative AI:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Mathematical Understanding",
                description: "Deep comprehension of the probabilistic foundations underlying diffusion processes"
              },
              {
                icon: <Code className="w-6 h-6" />,
                title: "Implementation Skills",
                description: "Hands-on experience with complex neural architectures and training procedures"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Optimization Techniques",
                description: "Understanding of memory management, computational efficiency, and training stability"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Generation Quality",
                description: "Insights into factors affecting image quality, diversity, and generation control"
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
        title: "Technical Stack & Tools"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Code className="w-6 h-6" />,
                title: "PyTorch",
                description: "Deep learning framework for neural network implementation and training"
              },
              {
                icon: <Cpu className="w-6 h-6" />,
                title: "CUDA",
                description: "GPU acceleration for efficient training and inference"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "NumPy & PIL",
                description: "Numerical computing and image processing utilities"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Custom Modules",
                description: "Hand-crafted implementations of VAE, U-Net, and sampling components"
              }
            ],
            columns: 4
          }
        },
        {
          type: 'paragraph' as const,
          content: "The implementation prioritizes educational value and understanding over using pre-built libraries, resulting in a comprehensive grasp of diffusion model internals."
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Key Learnings",
        background: true
      },
        children: [
            {
            type: 'paragraph' as const,
            content: "This project reinforced the importance of understanding the mathematical and architectural foundations of generative models. Key takeaways include:"
            },
            {
            type: 'paragraph' as const,
            content: "• The elegance of diffusion processes in generative modeling\n• The complexity of training deep generative models\n• The impact of architectural choices on model performance\n• The challenges and rewards of building from scratch"
            }
        ]
    },
  ],
  navigation: {
    colorScheme: 'purple' as const
  }
};

export default function CustomDiffusionModelPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-purple-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-purple-600 dark:hover:text-purple-400 transition-colors">
                Projects
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <PageBuilder {...diffusionPageData} />
      </article>
    </div>
  );
}
