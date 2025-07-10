import { Archive, Expand, Brain, ArrowRightLeft, Code, Image, Search, ExternalLink } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Encoders and Decoders - ML Portfolio',
  description: 'Neural network components that compress and reconstruct data representations for various tasks',
};

const encoderDecoderData = {
  title: "Encoders and Decoders",
  header: {
    category: "Deep Learning",
    difficulty: "Intermediate" as const,
    readTime: "8 min read",
    description: "Fundamental neural network components that compress data into latent representations (encoders) and reconstruct it (decoders), enabling powerful applications in compression, generation, and translation",
    relatedProjects: ["Custom Diffusion Model", "Custom GPT LLM"],
    gradientFrom: "from-teal-50 to-blue-50",
    gradientTo: "dark:from-teal-900/20 dark:to-blue-900/20",
    borderColor: "border-teal-200 dark:border-teal-800"
  },
  tags: {
    items: ['Representation Learning', 'Autoencoders', 'Sequence-to-Sequence', 'Transformers'],
    colorScheme: 'blue' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What are Encoders and Decoders?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Encoders and decoders are complementary neural network components that form the backbone of many modern AI architectures. Encoders compress input data into a compact latent representation, while decoders reconstruct or generate output from these representations."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Key Concept",
            icon: <Brain className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "This encoder-decoder paradigm enables learning compressed representations that capture essential information while discarding redundancy, making it powerful for tasks like translation, generation, and compression."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Types of Encoder-Decoder Architectures"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Archive className="w-6 h-6" />,
                title: "Autoencoders",
                description: "Unsupervised learning of data representations by reconstructing input data through a bottleneck.",
                color: "blue"
              },
              {
                icon: <ArrowRightLeft className="w-6 h-6" />,
                title: "Sequence-to-Sequence",
                description: "Maps variable-length input sequences to variable-length output sequences for translation tasks.",
                color: "green"
              },
              {
                icon: <Expand className="w-6 h-6" />,
                title: "Variational Autoencoders",
                description: "Probabilistic version of autoencoders that can generate new samples from learned distribution.",
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
        title: "Basic Autoencoder Implementation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's how to implement a basic autoencoder for image reconstruction:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    """Basic autoencoder for image reconstruction"""
    def __init__(self, input_dim=784, encoding_dim=128):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for images"""
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1x28x28
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 16x14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),                      # 64x1x1
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),                    # 32x7x7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, 
                             output_padding=1),               # 16x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, 
                             output_padding=1),               # 1x28x28
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, dataloader, num_epochs=10, lr=1e-3):
    """Train autoencoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            # For fully connected autoencoder, flatten the images
            if isinstance(model, Autoencoder):
                data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
    
    return losses

# Prepare dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Train both types of autoencoders
fc_autoencoder = Autoencoder()
conv_autoencoder = ConvAutoencoder()

print("Training Fully Connected Autoencoder...")
fc_losses = train_autoencoder(fc_autoencoder, train_loader, num_epochs=5)

print("\\nTraining Convolutional Autoencoder...")
conv_losses = train_autoencoder(conv_autoencoder, train_loader, num_epochs=5)

# Visualize results
def visualize_reconstruction(model, dataloader, num_images=8):
    """Visualize original vs reconstructed images"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        data, _ = next(iter(dataloader))
        data = data[:num_images].to(device)
        
        if isinstance(model, Autoencoder):
            data_flat = data.view(data.size(0), -1)
            reconstructed = model(data_flat).view(data.size())
        else:
            reconstructed = model(data)
        
        # Plot original and reconstructed images
        fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
        for i in range(num_images):
            # Original
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Visualize results
visualize_reconstruction(conv_autoencoder, train_loader)`,
          props: {
            language: 'python',
            title: 'Basic Autoencoder Implementation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Variational Autoencoder (VAE)"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "VAEs extend autoencoders to the probabilistic setting, enabling generation of new samples:"
        },
        {
          type: 'codeBlock' as const,
          content: `class VAE(nn.Module):
    """Variational Autoencoder"""
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

def vae_loss(reconstructed, original, mu, logvar, beta=1.0):
    """VAE loss with reconstruction and KL divergence terms"""
    # Reconstruction loss
    recon_loss = nn.functional.binary_cross_entropy(reconstructed, original, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

def train_vae(model, dataloader, num_epochs=10, lr=1e-3, beta=1.0):
    """Train VAE"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.view(data.size(0), -1).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar = model(data)
            loss = vae_loss(reconstructed, data, mu, logvar, beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.2f}')
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.2f}')
    
    return losses

def generate_samples(vae_model, num_samples=8, latent_dim=20):
    """Generate new samples from VAE"""
    vae_model.eval()
    device = next(vae_model.parameters()).device
    
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Decode to generate images
        generated = vae_model.decode(z)
        generated = generated.view(num_samples, 1, 28, 28)
        
        # Visualize generated samples
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
        for i in range(num_samples):
            axes[i].imshow(generated[i].cpu().squeeze(), cmap='gray')
            axes[i].set_title(f'Generated {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Train VAE
vae_model = VAE(latent_dim=20)
vae_losses = train_vae(vae_model, train_loader, num_epochs=10)

# Generate new samples
generate_samples(vae_model, num_samples=8)`,
          props: {
            language: 'python',
            title: 'Variational Autoencoder (VAE)'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Sequence-to-Sequence Models"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Seq2Seq models use encoder-decoder architecture for tasks like machine translation:"
        },
        {
          type: 'codeBlock' as const,
          content: `class Seq2SeqEncoder(nn.Module):
    """RNN-based encoder for sequence-to-sequence models"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super(Seq2SeqEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

class Seq2SeqDecoder(nn.Module):
    """RNN-based decoder for sequence-to-sequence models"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super(Seq2SeqDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.output_projection(output)
        return output, hidden

class Seq2Seq(nn.Module):
    """Complete Sequence-to-Sequence model"""
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, num_layers=2):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Seq2SeqEncoder(src_vocab_size, embed_size, hidden_size, num_layers)
        self.decoder = Seq2SeqDecoder(tgt_vocab_size, embed_size, hidden_size, num_layers)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.output_projection.out_features
        
        # Encode source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Initialize decoder hidden state with encoder's final state
        decoder_hidden = (hidden, cell)
        
        # First input to decoder is SOS token
        decoder_input = tgt[:, 0:1]  # Assuming SOS token is at index 0
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size)
        
        for t in range(1, tgt_len):
            # Forward pass through decoder
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t:t+1] = output
            
            # Teacher forcing: use actual next token as input
            # Otherwise use model's own prediction
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                decoder_input = tgt[:, t:t+1]
            else:
                decoder_input = output.argmax(dim=-1)
        
        return outputs

# Example usage for machine translation
def create_translation_model():
    """Create a simple translation model"""
    
    # Vocabulary sizes (in practice, these would be determined from data)
    src_vocab_size = 10000  # Source language vocabulary
    tgt_vocab_size = 8000   # Target language vocabulary
    
    model = Seq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_size=256,
        hidden_size=512,
        num_layers=2
    )
    
    return model

# Training function for Seq2Seq
def train_seq2seq(model, dataloader, num_epochs=10, lr=1e-3):
    """Train sequence-to-sequence model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt)
            
            # Reshape for loss computation
            output = output.reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)  # Exclude SOS token
            
            loss = criterion(output, tgt)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

# Create model
translation_model = create_translation_model()
print(f"Model parameters: {sum(p.numel() for p in translation_model.parameters()):,}")`,
          props: {
            language: 'python',
            title: 'Sequence-to-Sequence Model'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Applications and Use Cases"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Autoencoder Applications",
            icon: <Image className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "Dimensionality reduction and data compression",
                  "Anomaly detection and outlier identification",
                  "Denoising and image restoration",
                  "Feature learning for downstream tasks"
                ]
              }
            }
          ]
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Seq2Seq Applications",
            icon: <Code className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {  
                items: [
                  "Machine translation between languages",
                  "Text summarization and generation",
                  "Chatbots and conversational AI",
                  "Code generation and programming assistance"
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
          content: "Comprehensive resources to deepen your understanding of encoder-decoder architectures:"
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
                      title: "Sequence to Sequence Learning with Neural Networks",
                      description: "Sutskever et al. (2014) - Foundational seq2seq architecture"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Learning Phrase Representations using RNN Encoder-Decoder",
                      description: "Cho et al. (2014) - GRU-based encoder-decoder model"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "Neural Machine Translation by Jointly Learning to Align and Translate",
                      description: "Bahdanau et al. (2014) - Adding attention to seq2seq"
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
                      title: "TensorFlow Seq2Seq Tutorial",
                      description: "Official implementation guide with detailed examples"
                    },
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "PyTorch Encoder-Decoder Examples",
                      description: "Practical implementations for various tasks"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "OpenNMT Framework",
                      description: "Production-ready encoder-decoder toolkit"
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
    colorScheme: 'blue' as const
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
        <TopicPageBuilder {...encoderDecoderData} />
      </article>
    </div>
  );
}
