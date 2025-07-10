import { Layers, BarChart3, Target, TrendingUp, Zap, Settings, Eye, Users } from 'lucide-react';
import TopicPageBuilder from '../../components/TopicPageBuilder';

export const metadata = {
  title: 'Clustering Algorithms - ML Portfolio',
  description: 'Unsupervised learning techniques for grouping similar data points',
};

const clusteringTopicData = {
  title: "Clustering Algorithms",
  header: {
    category: "Machine Learning",
    difficulty: "Beginner" as const,
    readTime: "6 min read",
    description: "Unsupervised learning techniques for discovering hidden patterns and grouping similar data points without labeled examples",
    relatedProjects: ["Clustering Exploration"],
    gradientFrom: "from-yellow-50 to-orange-50",
    gradientTo: "dark:from-yellow-900/20 dark:to-orange-900/20",
    borderColor: "border-yellow-200 dark:border-yellow-800"
  },
  tags: {
    items: ['Unsupervised Learning', 'K-means', 'TSNE', 'Data Analysis'],
    colorScheme: 'yellow' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "What is Clustering?",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Clustering is an unsupervised machine learning technique that groups similar data points together based on their characteristics. Unlike supervised learning, clustering doesn't require labeled data – instead, it discovers hidden patterns and structures within the data."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Key Insight",
            icon: <Target className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "The goal is to maximize intra-cluster similarity (points within the same cluster are similar) while minimizing inter-cluster similarity (points in different clusters are different)."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Popular Clustering Algorithms"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Different clustering algorithms use various approaches to group data points. Here are the most commonly used methods:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "K-Means Clustering"
              },
              {
                type: 'paragraph' as const,
                content: "The most popular clustering algorithm that partitions data into k clusters by minimizing within-cluster sum of squares."
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "Requires specifying number of clusters (k)",
                    "Iteratively updates cluster centroids",
                    "Works well with spherical clusters",
                    "Computationally efficient for large datasets"
                  ]
                }
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Hierarchical Clustering"
              },
              {
                type: 'paragraph' as const,
                content: "Creates a tree-like hierarchy of clusters, either by merging (agglomerative) or splitting (divisive) clusters."
              },
              {
                type: 'list' as const,
                props: {
                  items: [
                    "No need to specify number of clusters",
                    "Produces a dendrogram showing relationships",
                    "Can capture nested cluster structures",
                    "More computationally expensive than k-means"
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
        title: "K-Means Implementation",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Here's a step-by-step implementation of the K-Means algorithm from scratch:"
        },
        {
          type: 'codeBlock' as const,
          props: {
            language: 'python',
            filename: 'kmeans_from_scratch.py'
          },
          content: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
    
    def fit(self, X):
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Initialize centroids randomly
        n_samples, n_features = X.shape
        self.centroids = np.random.uniform(
            low=X.min(axis=0), 
            high=X.max(axis=0), 
            size=(self.k, n_features)
        )
        
        for i in range(self.max_iters):
            # Assign points to closest centroid
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([
                X[self.labels == j].mean(axis=0) for j in range(self.k)
            ])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged after {i+1} iterations")
                break
                
            self.centroids = new_centroids
        
        return self
    
    def _compute_distances(self, X):
        """Compute distances from each point to each centroid"""
        distances = np.zeros((X.shape[0], self.k))
        for idx, centroid in enumerate(self.centroids):
            distances[:, idx] = np.linalg.norm(X - centroid, axis=1)
        return distances
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

# Example usage
# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit K-Means
kmeans = KMeans(k=4)
kmeans.fit(X)

# Visualize results
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'purple']
for i in range(kmeans.k):
    cluster_points = X[kmeans.labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               c=colors[i], alpha=0.6, label=f'Cluster {i+1}')
    plt.scatter(kmeans.centroids[i, 0], kmeans.centroids[i, 1], 
               c='black', marker='x', s=200)

plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()`
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Choosing the Right Number of Clusters"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "One of the main challenges in K-Means is determining the optimal number of clusters. Several methods can help with this decision:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Elbow Method"
              },
              {
                type: 'paragraph' as const,
                content: "Plot the within-cluster sum of squares (WCSS) for different values of k and look for an 'elbow' in the curve."
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'elbow_method.py'
                },
                content: `def elbow_method(X, max_k=10):
    wcss = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(k=k)
        kmeans.fit(X)
        
        # Calculate WCSS
        wcss_k = 0
        for i in range(k):
            cluster_points = X[kmeans.labels == i]
            if len(cluster_points) > 0:
                wcss_k += np.sum((cluster_points - kmeans.centroids[i]) ** 2)
        wcss.append(wcss_k)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.show()
    
    return wcss

# Usage
wcss_values = elbow_method(X, max_k=10)`
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Silhouette Analysis"
              },
              {
                type: 'paragraph' as const,
                content: "Measures how similar points are to their own cluster compared to other clusters. Higher silhouette scores indicate better clustering."
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'silhouette_analysis.py'
                },
                content: `from sklearn.metrics import silhouette_score, silhouette_samples

def silhouette_analysis(X, max_k=10):
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(k=k)
        cluster_labels = kmeans.fit(X).labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"k={k}: Silhouette Score = {silhouette_avg:.3f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k + 1), silhouette_scores, 'ro-')
    plt.title('Silhouette Analysis')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.show()
    
    return silhouette_scores

# Usage
sil_scores = silhouette_analysis(X, max_k=10)`
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Other Clustering Algorithms",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "While K-Means is popular, other algorithms may be better suited for specific data characteristics:"
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
                  title: "DBSCAN",
                  icon: <Layers className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Density-based algorithm that can find arbitrarily shaped clusters and identify outliers."
                  },
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "No need to specify number of clusters",
                        "Can handle noise and outliers",
                        "Finds clusters of varying densities",
                        "Sensitive to hyperparameters"
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
                  title: "Gaussian Mixture Models",
                  icon: <BarChart3 className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Probabilistic model that assumes data comes from a mixture of Gaussian distributions."
                  },
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Provides soft clustering (probabilities)",
                        "Can handle elliptical clusters",
                        "Model selection via information criteria",
                        "More complex than K-Means"
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
        title: "Dimensionality Reduction for Visualization"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "When working with high-dimensional data, dimensionality reduction techniques help visualize clustering results:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "t-SNE (t-Distributed Stochastic Neighbor Embedding)"
              },
              {
                type: 'paragraph' as const,
                content: "Excellent for visualizing clusters in 2D or 3D, preserving local neighborhood structures."
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'tsne_visualization.py'
                },
                content: `from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load high-dimensional data
digits = load_digits()
X, y = digits.data, digits.target

# Apply K-Means clustering
kmeans = KMeans(k=10, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Visualize clustering results
plt.figure(figsize=(15, 5))

# Original labels
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.title('t-SNE: True Labels')
plt.colorbar(scatter)

# Clustering results
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='tab10')
plt.title('t-SNE: K-Means Clusters')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()`
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "PCA (Principal Component Analysis)"
              },
              {
                type: 'paragraph' as const,
                content: "Linear dimensionality reduction that preserves maximum variance, useful for preprocessing before clustering."
              },
              {
                type: 'codeBlock' as const,
                props: {
                  language: 'python',
                  filename: 'pca_clustering.py'
                },
                content: `from sklearn.decomposition import PCA

# Apply PCA before clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Cluster in reduced space
kmeans_pca = KMeans(k=10, random_state=42)
cluster_labels_pca = kmeans_pca.fit_predict(X_pca)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=cluster_labels_pca, cmap='tab10')
plt.title('PCA + K-Means Clustering')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.colorbar(scatter)
plt.show()

print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")`
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
          content: "Clustering has numerous practical applications across various domains:"
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
                  title: "Business Applications",
                  icon: <Users className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Customer segmentation for marketing",
                        "Market research and demographics",
                        "Product recommendation systems",
                        "Fraud detection in finance",
                        "Supply chain optimization"
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
                  title: "Scientific Applications",
                  icon: <Eye className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Gene expression analysis in bioinformatics",
                        "Image segmentation in computer vision",
                        "Document classification in NLP",
                        "Social network analysis",
                        "Climate pattern recognition"
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
                  icon: <Settings className="w-6 h-6" />
                },
                children: [
                  {
                    type: 'list' as const,
                    props: {
                      items: [
                        "Not scaling features before clustering",
                        "Choosing k arbitrarily without validation",
                        "Ignoring the curse of dimensionality",
                        "Not considering algorithm assumptions",
                        "Over-interpreting cluster meanings"
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
                        "Always scale/normalize your features",
                        "Use multiple evaluation metrics",
                        "Try different algorithms and compare",
                        "Validate results with domain knowledge",
                        "Consider ensemble clustering methods"
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
          content: "Explore these resources to deepen your understanding of Clustering Algorithms:"
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
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "A Density-Based Algorithm",
                      description: "DBSCAN paper by Ester et al. (1996)"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Scikit-learn Clustering",
                      description: "Comprehensive clustering comparison study"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "A Tutorial on Spectral Clustering",
                      description: "Von Luxburg's detailed survey (2007)"
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
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Pattern Recognition and ML",
                      description: "Bishop's textbook - Chapter 9"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Scikit-learn Documentation",
                      description: "Practical clustering implementations"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Elements of Statistical Learning",
                      description: "Hastie et al. - Chapter 14"
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
    colorScheme: 'yellow' as const
  }
};

export default function ClusteringTopicPage() {
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
        <TopicPageBuilder {...clusteringTopicData} />
      </article>
    </div>
  );
}
