import { Layers, Brain, Camera, Target, TrendingUp, Code, BarChart3, Eye, Zap } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Convolutional Neural Networks (CNNs) - ML Portfolio',
  description: 'Deep learning architecture designed for processing grid-like data such as images',
};

const cnnTopicData = {
  title: "Convolutional Neural Networks (CNNs)",
  header: {
    category: "Deep Learning",
    difficulty: "Intermediate" as const,
    readTime: "8 min read",
    description: "Deep learning architecture designed for processing grid-like data such as images, with applications in computer vision, object detection, and image classification",
    relatedProjects: ["Image Classifier", "TAMID Image Classifier"],
    gradientFrom: "from-blue-50 to-purple-50",
    gradientTo: "dark:from-blue-900/20 dark:to-purple-900/20",
    borderColor: "border-blue-200 dark:border-blue-800"
  },
  tags: {
    items: ['Computer Vision', 'Deep Learning', 'Image Processing', 'Neural Networks'],
    colorScheme: 'blue' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What are CNNs?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Convolutional Neural Networks (CNNs) are a specialized type of neural network architecture particularly effective for processing data with a grid-like topology, such as images. CNNs use mathematical operations called convolutions to detect local features and patterns in data."
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
              content: "Unlike traditional neural networks that treat all input features equally, CNNs leverage the spatial structure of images by applying filters that can detect edges, textures, and more complex patterns as you go deeper into the network."
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
          content: "CNNs consist of several key components that work together to process and understand visual information:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Convolutional Layers"
              },
              {
                type: 'paragraph' as const,
                content: "The heart of CNNs, these layers apply learnable filters (kernels) across the input to detect features. Each filter specializes in detecting specific patterns like edges, corners, or textures."
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Feature detection through convolution operation",
                    "Parameter sharing reduces overfitting",
                    "Translation invariance for pattern recognition",
                    "Multiple filters learn different features"
                  ]
                }
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Pooling Layers"
              },
              {
                type: 'paragraph' as const,
                content: "Pooling layers reduce the spatial dimensions of feature maps, making the network more computationally efficient and providing translation invariance."
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Max pooling selects maximum values",
                    "Average pooling computes mean values",
                    "Reduces computational requirements",
                    "Provides spatial invariance"
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
        title: "CNN Architecture Example",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's a simple CNN implementation using PyTorch, demonstrating the basic architecture used in image classification:"
        },
        {
          type: 'codeBlock' as const,
          props: {
            language: 'python',
            filename: 'simple_cnn.py'
          },
          content: `import torch
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
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Usage example
model = SimpleCNN(num_classes=10)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")`
        }
      ]
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
                content: "Convolution Operation"
              },
              {
                type: 'paragraph' as const,
                content: "The convolution operation is the mathematical foundation of CNNs. It involves sliding a filter (kernel) across the input and computing dot products."
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'convolution_example.py'
                },
                content: `import numpy as np

def convolution_2d(image, kernel):
    """Simple 2D convolution implementation"""
    # Get dimensions
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate output dimensions
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    
    # Initialize output
    output = np.zeros((output_h, output_w))
    
    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            # Extract patch
            patch = image[i:i+kernel_h, j:j+kernel_w]
            # Compute dot product
            output[i, j] = np.sum(patch * kernel)
    
    return output

# Example: Edge detection kernel
edge_kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

# Apply to image (placeholder)
# result = convolution_2d(image, edge_kernel)`
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Feature Learning Hierarchy"
              },
              {
                type: 'paragraph' as const,
                content: "CNNs learn features in a hierarchical manner, from simple edges and textures in early layers to complex objects in deeper layers."
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Layer 1: Edges, lines, simple patterns",
                    "Layer 2-3: Textures, shapes, corners",
                    "Layer 4-5: Object parts, components",
                    "Final layers: Complete objects, faces"
                  ]
                }
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Automatic Feature Learning"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Unlike traditional computer vision methods that require hand-crafted features, CNNs automatically learn the most relevant features for the task during training."
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
        title: "Training CNNs",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Training CNNs involves several important considerations for achieving good performance:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Data Preparation"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Image normalization (0-1 or z-score)",
                    "Data augmentation (rotation, flip, crop)",
                    "Proper train/validation/test splits",
                    "Handling class imbalance"
                  ]
                }
              },
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Training Techniques"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Learning rate scheduling",
                    "Batch normalization for stability",
                    "Dropout for regularization",
                    "Early stopping to prevent overfitting"
                  ]
                }
              }
            ],
            right: [
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'training_loop.py'
                },
                content: `# Training loop example
def train_cnn(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_accuracy = validate_model(model, val_loader)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, '
              f'Val Acc: {val_accuracy:.2f}%')`
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Popular CNN Architectures"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Several groundbreaking CNN architectures have shaped the field of computer vision:"
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
                  title: "Classic Architectures",
                  icon: <Layers className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "LeNet-5 (1998): First successful CNN",
                        "AlexNet (2012): ImageNet breakthrough",
                        "VGG (2014): Deeper networks with small filters",
                        "GoogLeNet (2014): Inception modules"
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
                  title: "Modern Architectures",
                  icon: <Zap className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "ResNet (2015): Skip connections",
                        "DenseNet (2016): Dense connections",
                        "EfficientNet (2019): Compound scaling",
                        "Vision Transformer (2020): Attention-based"
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
        title: "Applications and Use Cases",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "CNNs have revolutionized computer vision and are used in numerous real-world applications:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Computer Vision Tasks"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Image classification (ImageNet, CIFAR)",
                    "Object detection (YOLO, R-CNN)",
                    "Semantic segmentation (U-Net, FCN)",
                    "Face recognition and verification",
                    "Medical image analysis (X-rays, MRI)",
                    "Autonomous driving (traffic signs, pedestrians)"
                  ]
                }
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Industry Applications"
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Healthcare: Cancer detection, radiology",
                    "Retail: Product recognition, inventory",
                    "Agriculture: Crop monitoring, pest detection",
                    "Manufacturing: Quality control, defect detection",
                    "Security: Surveillance, biometric authentication",
                    "Entertainment: Photo editing, content creation"
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
        title: "Best Practices and Tips"
      },
      children: [
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'warning' as const,
                  title: "Common Pitfalls",
                  icon: <Target className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Using too many parameters with limited data",
                        "Forgetting data normalization",
                        "Not using data augmentation",
                        "Learning rate too high or too low",
                        "Inadequate validation strategy"
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
                  title: "Optimization Strategies",
                  icon: <TrendingUp className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Transfer learning from pre-trained models",
                        "Proper data augmentation techniques",
                        "Batch normalization for training stability",
                        "Learning rate scheduling",
                        "Cross-validation for robust evaluation"
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
        title: "References and Further Learning"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Explore these resources to deepen your understanding of Convolutional Neural Networks:"
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
                      icon: <Eye className="w-6 h-6" />,
                      title: "ImageNet Classification with Deep CNNs",
                      description: "AlexNet paper by Krizhevsky et al. (2012)"
                    },
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Very Deep CNNs for Large-Scale Image Recognition",
                      description: "VGG networks by Simonyan & Zisserman (2014)"
                    },
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Deep Residual Learning for Image Recognition",
                      description: "ResNet paper by He et al. (2016)"
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
                      icon: <Eye className="w-6 h-6" />,
                      title: "CS231n: Convolutional Neural Networks",
                      description: "Stanford's comprehensive course on CNNs"
                    },
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Deep Learning Book - Chapter 9",
                      description: "Ian Goodfellow's detailed CNN explanation"
                    },
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "PyTorch CNN Tutorials",
                      description: "Hands-on implementation guides"
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

export default function CNNTopicPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Projects
              </a>
              <a href="/topics" className="text-blue-600 dark:text-blue-400 font-medium">
                Topics
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <TopicPageBuilder {...cnnTopicData} />
      </article>
    </div>
  );
}
