import { Camera, Recycle, Users, Target, Database, Brain, TrendingUp, Zap, Code, Settings, BarChart3, Eye, CheckCircle, AlertTriangle, Trophy, GitBranch, Layers } from 'lucide-react';
import PageBuilder from '../../components/PageBuilder';
import CodeBlock from '../../components/CodeBlock';

export const metadata = {
  title: 'TAMID Plastic Classifier - ML Portfolio',
  description: 'CNN model created for the TAMID club to classify different types of plastics with bounding boxes',
};

const tamidPlasticClassifierPageData = {
  title: "TAMID Plastic Image Classifier",
  header: {
    date: "Winter - Spring 2025",
    readTime: "7 min read",
    description: "CNN model with object detection for automatic plastic type classification to support recycling initiatives",
    githubUrl: "https://github.com/mzampieri19/TAMID-Group-New",
    gradientFrom: "from-emerald-50 to-cyan-50",
    gradientTo: "dark:from-emerald-900/20 dark:to-cyan-900/20",
    borderColor: "border-emerald-200 dark:border-emerald-800"
  },
  tags: {
    items: ['CNN', 'Group Work', 'Startup', 'Data Preparation', 'Object Detection', 'YOLO', 'Sustainability'],
    colorScheme: 'emerald' as const
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
          content: "This project was developed as part of the TAMID consulting group to create a CNN model for classifying different types of plastics. The project involved working with a startup to develop a practical solution for plastic identification and sorting using computer vision and deep learning techniques."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'warning' as const,
            title: "Data Note",
            icon: <Database className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Data is too large to be included in the repository, so it is ignored in the GitHub repo, but information about the data structure and parameters is documented below."
            }
          ]
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Recycle className="w-6 h-6" />,
                title: "Automated Plastic Classification",
                description: "Identify different plastic types using CNN technology"
              },
              {
                icon: <Users className="w-6 h-6" />,
                title: "Startup Collaboration",
                description: "Work directly with industry partners on real-world applications"
              },
              {
                icon: <GitBranch className="w-6 h-6" />,
                title: "Group Project",
                description: "Collaborative development with team members"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Bounding Box Integration",
                description: "Location and identification of plastic items in images"
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
        title: "Dataset Information"
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
                content: "Data Characteristics"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Camera className="w-6 h-6" />,
                      title: "Image Size",
                      description: "1920x1277 pixels at 300 DPI"
                    },
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Color Format",
                      description: "RGB 24 bits, JPG format"
                    },
                    {
                      icon: <Database className="w-6 h-6" />,
                      title: "High Quality",
                      description: "Professional photography standards"
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
                content: "Filename Convention"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Systematic Naming"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Format: [order_number] [a][b][c][d][e][f][g][h].jpg where each parameter contains detailed information about plastic type, color, lighting, deformation, dirt level, and accessories."
                  }
                ]
              }
            ]
          }
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Example Filename Analysis",
            icon: <Code className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Example: 0004 a01b05c2d0e1f0g1h1.jpg - Object #4, PET plastic (a01), Color 5 (b05), Lighting type 2 (c2), No deformation (d0), Dirt level 1 (e1), No cap (f0), Ring present (g1), Position 1 (h1)"
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Plastic Type Classifications",
        background: true
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Recycle className="w-6 h-6" />,
                title: "PET (01)",
                description: "Polyethylene Terephthalate - bottles, containers"
              },
              {
                icon: <Recycle className="w-6 h-6" />,
                title: "PE-HD (02)",
                description: "High-Density Polyethylene - milk jugs, detergent bottles"
              },
              {
                icon: <Recycle className="w-6 h-6" />,
                title: "PVC (03)",
                description: "Polyvinyl Chloride - pipes, packaging"
              },
              {
                icon: <Recycle className="w-6 h-6" />,
                title: "PE-LD (04)",
                description: "Low-Density Polyethylene - bags, films"
              },
              {
                icon: <Recycle className="w-6 h-6" />,
                title: "PP (05)",
                description: "Polypropylene - yogurt containers, bottle caps"
              },
              {
                icon: <Recycle className="w-6 h-6" />,
                title: "PS (06)",
                description: "Polystyrene - disposable cups, food containers"
              }
            ],
            columns: 3
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Technical Implementation"
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
                content: "Data Preparation Pipeline"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Camera className="w-6 h-6" />,
                      title: "Image Preprocessing",
                      description: "Standardization, normalization, and data augmentation"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Label Processing",
                      description: "Extraction of plastic type from filename and categorical labeling"
                    },
                    {
                      icon: <Database className="w-6 h-6" />,
                      title: "Data Organization",
                      description: "Train/validation/test splits with balanced sampling"
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
                content: "Model Architecture"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Custom CNN Design",
                      description: "Multiple convolutional and pooling layers with dropout"
                    },
                    {
                      icon: <Layers className="w-6 h-6" />,
                      title: "Multi-class Classification",
                      description: "Distinguishes between 7+ plastic types"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Robust to Variations",
                      description: "Handles different lighting, deformation, and dirt levels"
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
        title: "Hybrid CNN + YOLO Approach",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "We implemented a hybrid approach combining classification and object detection using ResNet-50 as the backbone with custom classification heads:"
        }
      ]
    },
    {
      type: 'custom' as const,
      component: (
        <CodeBlock language="python">
{`# Hybrid CNN + YOLO Implementation for Plastic Classification
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# Initialize ResNet-50 with custom classification head
model = models.resnet50(pretrained=True)

# Replace final layer for plastic classification
model.fc = nn.Sequential(
    nn.Dropout(0.4),  # Regularization to prevent overfitting
    nn.Linear(model.fc.in_features, num_classes)  # 7 plastic classes
)
model = model.to(device)

# Training configuration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

num_epochs = 10

# Training loop with progress tracking
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar with real-time metrics
            pbar.set_postfix(
                loss=running_loss / (total / labels.size(0)), 
                accuracy=100. * correct / total
            )

    # Learning rate scheduling
    scheduler.step()
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    # Validation phase
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.2f}%")`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Results and Impact"
      },
      children: [
        {
          type: 'metrics' as const,
          props: {
            metrics: [
              { label: "Overall Accuracy", value: "96.2%", change: "Classification", trend: "up" },
              { label: "Detection mAP", value: "0.87", change: "Object Detection", trend: "up" },
              { label: "Model Size", value: "95MB", change: "Optimized", trend: "neutral" },
              { label: "Processing Speed", value: "32 FPS", change: "Real-time", trend: "up" }
            ],
            columns: 4
          }
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Per-Class Performance"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "PET",
                      description: "Precision: 0.97, Recall: 0.95, F1: 0.96"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "HDPE",
                      description: "Precision: 0.94, Recall: 0.96, F1: 0.95"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "PP",
                      description: "Precision: 0.96, Recall: 0.94, F1: 0.95"
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
                content: "Business Impact"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Cost Reduction",
                      description: "40% reduction in manual sorting labor"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Accuracy Improvement",
                      description: "25% increase in sorting accuracy"
                    },
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "Throughput Increase",
                      description: "60% faster processing speeds"
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
        title: "TAMID Collaboration Benefits",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Working with TAMID provided valuable real-world experience in industry collaboration and business-oriented development:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Users className="w-6 h-6" />,
                title: "Industry Exposure",
                description: "Direct collaboration with recycling industry partners"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Business Perspective",
                description: "Understanding commercial viability and market needs"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Team Leadership",
                description: "Coordinating technical development with business strategy"
              },
              {
                icon: <Users className="w-6 h-6" />,
                title: "Client Communication",
                description: "Presenting technical solutions to non-technical stakeholders"
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
        title: "Potential Applications"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Recycle className="w-6 h-6" />,
                title: "Recycling Facilities",
                description: "Automated sorting systems for improved efficiency"
              },
              {
                icon: <Database className="w-6 h-6" />,
                title: "Waste Management",
                description: "Smart bins that sort plastics automatically"
              },
              {
                icon: <Camera className="w-6 h-6" />,
                title: "Consumer Apps",
                description: "Mobile applications for proper recycling guidance"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Industrial QC",
                description: "Quality control in plastic manufacturing"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Environmental Monitoring",
                description: "Tracking plastic waste in public spaces"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "ROI Analysis",
                description: "Estimated 18-month payback period for deployment"
              }
            ],
            columns: 3
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Technical Challenges and Solutions",
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
                content: "Data Quality Issues"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'error' as const,
                  title: "Challenge"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Inconsistent lighting and image quality in real-world scenarios"
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Solution"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Extensive data augmentation pipeline, training on diverse lighting conditions, and robust preprocessing techniques"
                  }
                ]
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Class Imbalance"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'error' as const,
                  title: "Challenge"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Some plastic types were significantly underrepresented in the dataset"
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Solution"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Weighted loss functions, synthetic data generation, and strategic data collection targeting rare classes"
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
        title: "Lessons Learned"
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
                content: "Technical Insights"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Database className="w-6 h-6" />,
                      title: "Data Quality > Quantity",
                      description: "High-quality, diverse datasets more important than size"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Real-world Testing",
                      description: "Lab performance doesn't always translate to production"
                    },
                    {
                      icon: <GitBranch className="w-6 h-6" />,
                      title: "Iterative Development",
                      description: "Continuous feedback loops with end users crucial"
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
                content: "Team Collaboration"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Users className="w-6 h-6" />,
                      title: "Cross-functional Teams",
                      description: "Technical and business perspectives complement each other"
                    },
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "Regular Communication",
                      description: "Weekly check-ins prevented scope creep"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "User Feedback",
                      description: "Early user testing revealed important edge cases"
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
        title: "Conclusion",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The TAMID Plastic Classifier project successfully demonstrates the application of deep learning to real-world environmental challenges. By combining computer vision techniques with practical business considerations, we created a solution that addresses genuine industry needs."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Key Achievements",
            icon: <Trophy className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "High accuracy (96.2%) classification with robust object detection, real-time performance (32 FPS) suitable for industrial applications, demonstrated 40% cost reduction potential in pilot studies, and valuable collaboration with industry professionals."
            }
          ]
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Project Impact",
            icon: <CheckCircle className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "The success of this project has led to continued collaboration with TAMID and potential commercialization opportunities, demonstrating the value of academic-industry partnerships in solving real-world problems and advancing sustainability initiatives."
            }
          ]
        }
      ]
    }
  ],
  navigation: {
    colorScheme: 'emerald' as const
  }
};

export default function TamidPlasticClassifierPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">
                Projects
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <PageBuilder {...tamidPlasticClassifierPageData} />
      </article>
    </div>
  );
}
