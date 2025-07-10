import { Layers, ArrowUpDown, Brain, Image, Target, Zap, Search, BookOpen, Code, ExternalLink } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'U-Net Architecture - ML Portfolio',
  description: 'Convolutional network architecture for biomedical image segmentation with skip connections',
};

const unetData = {
  title: "U-Net Architecture",
  header: {
    category: "Deep Learning",
    difficulty: "Advanced" as const,
    readTime: "8 min read",
    description: "Convolutional neural network architecture designed for biomedical image segmentation, featuring encoder-decoder structure with skip connections",
    relatedProjects: ["Custom Diffusion Model"],
    gradientFrom: "from-cyan-50 to-blue-50",
    gradientTo: "dark:from-cyan-900/20 dark:to-blue-900/20",
    borderColor: "border-cyan-200 dark:border-cyan-800"
  },
  tags: {
    items: ['Image Segmentation', 'Medical Imaging', 'Skip Connections', 'CNN'],
    colorScheme: 'blue' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What is U-Net?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "U-Net is a convolutional neural network architecture developed for biomedical image segmentation. It was first introduced by Ronneberger et al. in 2015 and has since become one of the most popular architectures for image segmentation tasks across various domains."
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
              content: "The U-Net architecture combines the benefits of encoder-decoder networks with skip connections that preserve spatial information, making it highly effective for pixel-level prediction tasks."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "U-Net Architecture Components"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <ArrowUpDown className="w-6 h-6" />,
                title: "Encoder (Contracting Path)",
                description: "Captures context through convolutional and pooling layers, reducing spatial dimensions while increasing feature depth.",
                color: "blue"
              },
              {
                icon: <Layers className="w-6 h-6" />,
                title: "Decoder (Expanding Path)",
                description: "Enables precise localization through upsampling and convolutions, recovering spatial resolution.",
                color: "green"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Skip Connections",
                description: "Connect encoder and decoder at same spatial level, preserving fine-grained spatial information.",
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
        title: "U-Net Implementation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's a complete implementation of U-Net using PyTorch:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block used in U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size differences due to padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Complete U-Net architecture"""
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

# Initialize model
model = UNet(n_channels=3, n_classes=2)  # RGB input, binary segmentation

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass
x = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 image
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")`,
          props: {
            language: 'python',
            title: 'Complete U-Net Implementation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Training U-Net for Image Segmentation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's how to train U-Net for a segmentation task:"
        },
        {
          type: 'codeBlock' as const,
          content: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    """Custom dataset for image segmentation"""
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Grayscale
        
        if self.transform:
            # Apply same transform to both image and mask
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            image = self.transform(image)
            
            torch.manual_seed(seed)
            mask = self.transform(mask)
        
        # Convert mask to long tensor for cross-entropy loss
        mask = (mask > 0.5).long()
        
        return image, mask.squeeze(0)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Flatten tensors
        predictions = torch.sigmoid(predictions).view(-1)
        targets = targets.view(-1).float()
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combination of CrossEntropy and Dice loss"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions[:, 1, :, :], targets)  # Use positive class
        return self.ce_weight * ce + self.dice_weight * dice

def train_unet(model, train_loader, val_loader, num_epochs=50):
    """Training function for U-Net"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
    
    return train_losses, val_losses

# Example usage (with dummy data paths)
# image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
# mask_paths = ['path/to/mask1.png', 'path/to/mask2.png', ...]

# train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, transform)
# val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, transform)

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# model = UNet(n_channels=3, n_classes=2)
# train_losses, val_losses = train_unet(model, train_loader, val_loader)`,
          props: {
            language: 'python',
            title: 'Training U-Net for Segmentation'
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "U-Net Variants and Applications"
      },
      children: [
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Popular Applications",
            icon: <Image className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "Medical image segmentation (organs, tumors)",
                  "Satellite image analysis",
                  "Object detection and localization",
                  "Image inpainting and restoration"
                ]
              }
            }
          ]
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Modern Variants",
            icon: <Zap className="w-6 h-6" />
          },
          children: [
            {
              type: 'list' as const,
              props: {
                items: [
                  "U-Net++: Nested U-Net with dense skip connections",
                  "Attention U-Net: Incorporates attention mechanisms",
                  "3D U-Net: Extended for volumetric data",
                  "ResUNet: Combines U-Net with ResNet blocks"
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
        title: "Evaluation Metrics for Segmentation"
      },
      children: [
        {
          type: 'codeBlock' as const,
          content: `import torch
import numpy as np

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """Calculate Intersection over Union (IoU)"""
    pred_mask = (pred_mask > threshold).float()
    true_mask = true_mask.float()
    
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    
    iou = intersection / (union + 1e-8)
    return iou.item()

def calculate_dice_coefficient(pred_mask, true_mask, threshold=0.5):
    """Calculate Dice coefficient"""
    pred_mask = (pred_mask > threshold).float()
    true_mask = true_mask.float()
    
    intersection = (pred_mask * true_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + true_mask.sum() + 1e-8)
    
    return dice.item()

def calculate_pixel_accuracy(pred_mask, true_mask, threshold=0.5):
    """Calculate pixel-wise accuracy"""
    pred_mask = (pred_mask > threshold).float()
    true_mask = true_mask.float()
    
    correct_pixels = (pred_mask == true_mask).sum()
    total_pixels = true_mask.numel()
    
    accuracy = correct_pixels / total_pixels
    return accuracy.item()

def evaluate_segmentation(model, test_loader, device):
    """Comprehensive evaluation of segmentation model"""
    model.eval()
    
    total_iou = 0
    total_dice = 0
    total_accuracy = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Get predictions
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1)[:, 1, :, :]  # Positive class
            
            # Calculate metrics for each sample in batch
            for i in range(predictions.size(0)):
                pred = predictions[i]
                true = masks[i]
                
                iou = calculate_iou(pred, true)
                dice = calculate_dice_coefficient(pred, true)
                accuracy = calculate_pixel_accuracy(pred, true)
                
                total_iou += iou
                total_dice += dice
                total_accuracy += accuracy
                num_samples += 1
    
    # Calculate averages
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    avg_accuracy = total_accuracy / num_samples
    
    print(f"Evaluation Results:")
    print(f"  Average IoU: {avg_iou:.4f}")
    print(f"  Average Dice: {avg_dice:.4f}")
    print(f"  Average Pixel Accuracy: {avg_accuracy:.4f}")
    
    return avg_iou, avg_dice, avg_accuracy`,
          props: {
            language: 'python',
            title: 'Segmentation Evaluation Metrics'
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
          content: "Comprehensive resources to deepen your understanding of U-Net architecture:"
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
                      title: "U-Net: Convolutional Networks for Biomedical Image Segmentation",
                      description: "Ronneberger et al. (2015) - Original U-Net architecture for segmentation"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "3D U-Net: Learning Dense Volumetric Segmentation",
                      description: "Çiçek et al. (2016) - Extension of U-Net to 3D volumes"
                    },
                    {
                      icon: <Search className="w-6 h-6" />,
                      title: "UNet++: A Nested U-Net Architecture for Medical Image Segmentation",
                      description: "Zhou et al. (2018) - Improved U-Net with nested architecture"
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
                      title: "PyTorch U-Net Implementation",
                      description: "Official PyTorch Hub model for brain segmentation"
                    },
                    {
                      icon: <BookOpen className="w-6 h-6" />,
                      title: "Understanding U-Net",
                      description: "Comprehensive guide to U-Net architecture and applications"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "TensorFlow Image Segmentation",
                      description: "Official TensorFlow tutorial for image segmentation with U-Net"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "U-Net from Scratch",
                      description: "Step-by-step implementation guide"
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


export default function UNETPage() {
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
        <TopicPageBuilder {...unetData} />
      </article>
    </div>
  );
}