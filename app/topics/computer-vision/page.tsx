import { Camera, Eye, Target, TrendingUp, Zap, Code, Settings, BarChart3, Brain, ExternalLink, Layers, Image } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Computer Vision - ML Portfolio Topics',
  description: 'Learn about computer vision - field of AI that enables computers to interpret and understand visual information',
};

const computerVisionPageData = {
  title: "Computer Vision",
  header: {
    date: "AI Applications",
    readTime: "8 min read",
    description: "Field of artificial intelligence that enables computers to interpret and understand visual information from the world",
    gradientFrom: "from-purple-50 to-pink-50",
    gradientTo: "dark:from-purple-900/20 dark:to-pink-900/20",
    borderColor: "border-purple-200 dark:border-purple-800",
    difficulty: "Intermediate" as const,
    category: "AI Applications",
    relatedProjects: ["image-classifier", "tamid-image-classifier", "custom-diffusion-model"]
  },
  tags: {
    items: ['Computer Vision', 'Image Processing', 'Object Detection', 'CNN', 'Deep Learning'],
    colorScheme: 'purple' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What is Computer Vision?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects, and then react to what they 'see'."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Human-like Vision",
            icon: <Eye className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Computer vision attempts to replicate the human visual system, enabling machines to extract meaningful information from visual inputs and make decisions based on that understanding."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Core Computer Vision Tasks"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Target className="w-6 h-6" />,
                title: "Image Classification",
                description: "Assign a single label to an entire image (e.g., cat, dog, car)"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Object Detection",
                description: "Locate and classify multiple objects within an image with bounding boxes"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Semantic Segmentation",
                description: "Classify each pixel in an image to create detailed scene understanding"
              },
              {
                icon: <Camera className="w-6 h-6" />,
                title: "Instance Segmentation",
                description: "Separate individual object instances, even of the same class"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Face Recognition",
                description: "Identify and verify individual faces from images or video"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Optical Character Recognition",
                description: "Extract and recognize text from images and documents"
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
        title: "Image Processing Fundamentals",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Before applying machine learning models, images often need preprocessing to enhance quality, standardize formats, and extract relevant features."
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Basic Operations"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "Resizing & Scaling",
                      description: "Adjust image dimensions for model input requirements"
                    },
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Color Space Conversion",
                      description: "Convert between RGB, HSV, grayscale, and other formats"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Normalization",
                      description: "Scale pixel values to standard ranges (0-1 or -1 to 1)"
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
                content: "Advanced Techniques"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "Filtering",
                      description: "Apply Gaussian blur, edge detection, noise reduction"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Histogram Equalization",
                      description: "Improve contrast and enhance image visibility"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Data Augmentation",
                      description: "Generate variations to increase dataset diversity"
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
        filename: "image_preprocessing.py",
        code: `import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path):
    """
    Comprehensive image preprocessing pipeline
    """
    # Read image using OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Basic preprocessing
    def basic_preprocessing(img):
        # Resize to standard size
        img_resized = cv2.resize(img, (224, 224))
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to grayscale if needed
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        return img_normalized, img_gray
    
    # Advanced preprocessing
    def advanced_preprocessing(img):
        # Gaussian blur for noise reduction
        img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Histogram equalization for contrast enhancement
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_eq = cv2.equalizeHist(img_gray)
        
        # Edge detection using Canny
        edges = cv2.Canny(img_gray, 100, 200)
        
        return img_blurred, img_eq, edges
    
    # Data augmentation using torchvision
    augmentation_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply basic preprocessing
    img_normalized, img_gray = basic_preprocessing(image_rgb)
    
    # Apply advanced preprocessing
    img_blurred, img_eq, edges = advanced_preprocessing(image_rgb)
    
    # Apply augmentation
    pil_image = Image.fromarray(image_rgb)
    augmented_tensor = augmentation_pipeline(pil_image)
    
    return {
        'original': image_rgb,
        'normalized': img_normalized,
        'grayscale': img_gray,
        'blurred': img_blurred,
        'equalized': img_eq,
        'edges': edges,
        'augmented': augmented_tensor
    }

# Example usage
results = preprocess_image('sample_image.jpg')
print(f"Original shape: {results['original'].shape}")
print(f"Normalized range: [{results['normalized'].min():.3f}, {results['normalized'].max():.3f}]")
print(f"Augmented tensor shape: {results['augmented'].shape}")`
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Convolutional Neural Networks for Vision"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "CNNs are the backbone of modern computer vision, designed specifically to process grid-like data such as images. They use convolutional layers to detect spatial features and hierarchical patterns."
        },
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Feature Detection",
                date: "Layer 1-2",
                description: "Low-level features like edges, corners, and basic shapes"
              },
              {
                title: "Pattern Recognition",
                date: "Layer 3-4",
                description: "Mid-level features like textures, parts of objects"
              },
              {
                title: "Object Parts",
                date: "Layer 5-6",
                description: "High-level features like faces, wheels, specific object components"
              },
              {
                title: "Semantic Understanding",
                date: "Final Layers",
                description: "Complete objects and scene understanding for classification"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Popular CNN Architectures",
        background: true
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Layers className="w-6 h-6" />,
                title: "LeNet-5",
                description: "Early CNN for handwritten digit recognition (1998)"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "AlexNet",
                description: "First deep CNN to win ImageNet, popularized deep learning (2012)"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "VGGNet",
                description: "Demonstrated importance of depth with small filters (2014)"
              },
              {
                icon: <Code className="w-6 h-6" />,
                title: "ResNet",
                description: "Residual connections enable training very deep networks (2015)"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "EfficientNet",
                description: "Optimized architecture scaling for better efficiency (2019)"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Vision Transformer",
                description: "Applied transformer architecture to computer vision (2020)"
              }
            ],
            columns: 2
          }
        }
      ]
    },
    {
      type: 'codeBlock' as const,
      props: {
        language: "python",
        filename: "cnn_classification.py",
        code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),  # Assuming 224x224 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for classifier
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

# ResNet Block implementation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + residual)

# Example usage
model = SimpleCNN(num_classes=1000)  # ImageNet classes
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
batch_size = 4
input_tensor = torch.randn(batch_size, 3, 224, 224)
output = model(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")`
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Object Detection Methods"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Object detection goes beyond classification by locating and identifying multiple objects within an image. Modern approaches can be categorized into two main paradigms."
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Two-Stage Detectors"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "R-CNN Family",
                      description: "Region proposals followed by classification (R-CNN, Fast R-CNN, Faster R-CNN)"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "High Accuracy",
                      description: "Generally more accurate but slower inference"
                    },
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "Region Proposal Network",
                      description: "Learn to generate object proposals automatically"
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
                content: "One-Stage Detectors"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "YOLO Family",
                      description: "You Only Look Once - direct prediction of bounding boxes and classes"
                    },
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Real-time Performance",
                      description: "Faster inference suitable for real-time applications"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "SSD",
                      description: "Single Shot MultiBox Detector with multi-scale feature maps"
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
        title: "Applications and Use Cases",
        background: true
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Camera className="w-6 h-6" />,
                title: "Autonomous Vehicles",
                description: "Object detection, lane detection, traffic sign recognition"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Medical Imaging",
                description: "X-ray analysis, MRI interpretation, disease detection"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Security & Surveillance",
                description: "Face recognition, anomaly detection, crowd monitoring"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Manufacturing Quality Control",
                description: "Defect detection, assembly verification, product sorting"
              },
              {
                icon: <Image className="w-6 h-6" />,
                title: "Retail & E-commerce",
                description: "Visual search, inventory management, checkout automation"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Agriculture",
                description: "Crop monitoring, pest detection, yield prediction"
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
        title: "References and Further Learning"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Explore these resources to deepen your understanding of computer vision:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Essential Libraries"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "OpenCV",
                      description: "Comprehensive computer vision and image processing library"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "PyTorch Vision",
                      description: "Deep learning models and utilities for computer vision"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "TensorFlow/Keras",
                      description: "High-level APIs for building and training vision models"
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
                      title: "CS231n: CNNs for Visual Recognition",
                      description: "Stanford's comprehensive computer vision course"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Computer Vision: Algorithms and Applications",
                      description: "Textbook by Richard Szeliski"
                    },
                    {
                      icon: <ExternalLink className="w-6 h-6" />,
                      title: "Papers With Code - Computer Vision",
                      description: "Latest research papers with implementation code"
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

export default function ComputerVisionPage() {
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
              <a href="/topics" className="text-purple-600 dark:text-purple-400 font-medium">
                Topics
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <TopicPageBuilder {...computerVisionPageData} />
      </article>
    </div>
  );
}
