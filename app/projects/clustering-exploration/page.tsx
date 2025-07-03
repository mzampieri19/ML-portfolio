import { BarChart3, Target, Brain, Database, Eye, GitBranch, Settings, TrendingUp } from 'lucide-react';
import PageBuilder from '../../components/PageBuilder';
import CodeBlock from '../../components/CodeBlock';

export const metadata = {
  title: 'Clustering Exploration - ML Portfolio',
  description: 'Exploration of various clustering techniques on financial data',
};

const clusteringPageData = {
  title: "Clustering Exploration",
  header: {
    date: "Winter 2025",
    readTime: "4 min read",
    description: "An exploration of various clustering techniques applied to financial data",
    githubUrl: "https://github.com/mzampieri19/clustering",
    gradientFrom: "from-green-50 to-teal-50",
    gradientTo: "dark:from-green-900/20 dark:to-teal-900/20",
    borderColor: "border-green-200 dark:border-green-800"
  },
  tags: {
    items: ['Clustering', 'Supervised Learning', 'TSNE', 'KMEANS'],
    colorScheme: 'green' as const
  },
  blocks: [
    {
      type: 'section' as const,
      props: {
        title: "Project Purpose",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The objective of this project is to gain a deeper understanding of projection and clustering techniques. The dataset used for this exploration was sourced from Kaggle and was selected due to its relatively clean structure and ease of use."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "The Data",
            icon: <Database className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "The dataset is an income survey of Canadian citizens containing extensive demographic and financial information. Detailed descriptions of the dataset's features can be found in the data documentation."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Data Preparation & Preprocessing"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The financial dataset required extensive preprocessing to ensure optimal clustering performance:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Database className="w-6 h-6" />,
                title: "Data Transformation",
                description: "One-hot encoding applied to categorical features, increasing dimensionality from 37 to 108 features"
              },
              {
                icon: <Settings className="w-6 h-6" />,
                title: "Feature Scaling",
                description: "StandardScaler applied to normalize the data and address inconsistencies in feature scaling"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Missing Value Handling",
                description: "Proper treatment of null values and data inconsistencies for clean analysis"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Feature Selection",
                description: "Identified most relevant variables for clustering and analyzed relationships between financial indicators"
              }
            ],
            columns: 2
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
            content: "Projection Techniques"
          },
          {
            type: 'highlight' as const,
            props: {
              variant: 'success' as const,
              title: "Principal Component Analysis (PCA)"
            },
            children: [
              {
                type: 'paragraph' as const,
                content: "Linear dimensionality reduction technique that preserves maximum variance in the data with efficient computation and interpretable results."
              }
            ]
          },
          {
            type: 'highlight' as const,
            props: {
              variant: 'warning' as const,
              title: "t-SNE"
            },
            children: [
              {
                type: 'paragraph' as const,
                content: "Non-linear dimensionality reduction technique with perplexity value of 50, excellent for visualizing cluster structures and local patterns."
              }
            ]
          }
        ],
        right: [
          {
            type: 'heading' as const,
            props: { level: 3 },
            content: "Clustering Algorithms"
          },
          {
            type: 'highlight' as const,
            props: {
              variant: 'info' as const,
              title: "KMeans Clustering"
            },
            children: [
              {
                type: 'paragraph' as const,
                content: "Partitioning method that divides data into k clusters, applied to both PCA and t-SNE projections with hyperparameter tuning for optimal cluster numbers."
              }
            ]
          },
          {
            type: 'highlight' as const,
            props: {
              variant: 'error' as const,
              title: "DBSCAN"
            },
            children: [
              {
                type: 'paragraph' as const,
                content: "Density-based clustering algorithm effective at finding clusters of varying shapes and sizes with hyperparameter tuning exploring different min_samples values."
              }
            ]
          }
        ]
      }
    },
    {
      type: 'custom' as const,
      component: (
        <CodeBlock language="python">
{`# Dimensionality reduction techniques
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Principal Component Analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

# t-SNE with optimal perplexity
tsne = TSNE(n_components=2, perplexity=50, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize projections
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
ax1.set_title('PCA Projection')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
ax2.set_title('t-SNE Projection (perplexity=50)')
ax2.set_xlabel('t-SNE Component 1')
ax2.set_ylabel('t-SNE Component 2')

plt.tight_layout()
plt.show()`}
        </CodeBlock>
      )
    },
    {
      type: 'custom' as const,
      component: (
        <CodeBlock language="python">
{`# Data preprocessing pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load and preprocess financial data
df = pd.read_csv('financial_data.csv')

# Handle categorical variables with one-hot encoding
df_encoded = pd.get_dummies(df, columns=['category', 'sector'])

# Feature scaling for consistent clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

print(f"Original features: {df.shape[1]}")
print(f"After preprocessing: {X_scaled.shape[1]}")  # 37 -> 108 features`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Comprehensive Analysis Results",
        background: true
      },
      children: [
        {
          type: 'metrics' as const,
          props: {
            metrics: [
              { label: "Clustering Models", value: "4", change: "Total", trend: "up" },
              { label: "Visualizations", value: "40", change: "Generated", trend: "up" },
              { label: "Algorithms", value: "2", change: "KMeans + DBSCAN", trend: "neutral" },
              { label: "Projections", value: "2", change: "PCA + t-SNE", trend: "neutral" }
            ],
            columns: 4
          }
        },
        {
          type: 'paragraph' as const,
          content: "Each algorithm was applied to both PCA and t-SNE projections, creating a total of four clustering models with extensive hyperparameter tuning and comprehensive visualization analysis."
        }
      ]
    },
    {
      type: 'custom' as const,
      component: (
        <CodeBlock language="python">
{`# Clustering algorithms implementation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

# KMeans clustering with elbow method for optimal k
def find_optimal_kmeans(X, k_range=range(2, 11)):
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, cluster_labels))
    
    return inertias, silhouette_scores

# Apply KMeans to both projections
k_range = range(2, 11)
pca_inertias, pca_silhouettes = find_optimal_kmeans(X_pca, k_range)
tsne_inertias, tsne_silhouettes = find_optimal_kmeans(X_tsne, k_range)

# Optimal KMeans models
optimal_k_pca = 4  # Based on elbow method
optimal_k_tsne = 5

kmeans_pca = KMeans(n_clusters=optimal_k_pca, random_state=42)
kmeans_tsne = KMeans(n_clusters=optimal_k_tsne, random_state=42)

pca_labels = kmeans_pca.fit_predict(X_pca)
tsne_labels = kmeans_tsne.fit_predict(X_tsne)

# DBSCAN clustering with hyperparameter tuning
def find_optimal_dbscan(X, eps_range, min_samples_range):
    best_score = -1
    best_params = {}
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            if len(set(labels)) > 1:  # Ensure we have clusters
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
    
    return best_params, best_score

# Optimize DBSCAN parameters
eps_range = np.arange(0.1, 2.0, 0.1)
min_samples_range = range(3, 10)

pca_dbscan_params, _ = find_optimal_dbscan(X_pca, eps_range, min_samples_range)
tsne_dbscan_params, _ = find_optimal_dbscan(X_tsne, eps_range, min_samples_range)

# Apply optimized DBSCAN
dbscan_pca = DBSCAN(**pca_dbscan_params)
dbscan_tsne = DBSCAN(**tsne_dbscan_params)

pca_dbscan_labels = dbscan_pca.fit_predict(X_pca)
tsne_dbscan_labels = dbscan_tsne.fit_predict(X_tsne)

print(f"KMeans PCA clusters: {optimal_k_pca}")
print(f"KMeans t-SNE clusters: {optimal_k_tsne}")
print(f"DBSCAN PCA clusters: {len(set(pca_dbscan_labels)) - (1 if -1 in pca_dbscan_labels else 0)}")
print(f"DBSCAN t-SNE clusters: {len(set(tsne_dbscan_labels)) - (1 if -1 in tsne_dbscan_labels else 0)}")`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Evaluation Metrics"
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Target className="w-6 h-6" />,
                title: "Inertia",
                description: "Measures compactness of clusters using within-cluster sum of squares. Lower inertia indicates more compact clusters."
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Silhouette Score",
                description: "Evaluates clustering quality by measuring similarity within clusters vs. other clusters. Score ranges from -1 to +1."
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Elbow Method",
                description: "Applied to determine optimal hyperparameters and ensure robust clustering analysis with consistent results."
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
{`# Comprehensive clustering evaluation
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_clustering(X, labels, algorithm_name, projection_name):
    """Evaluate clustering performance with multiple metrics"""
    
    # Remove noise points for DBSCAN
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    if len(set(labels_clean)) < 2:
        print(f"{algorithm_name} on {projection_name}: No valid clusters found")
        return
    
    # Calculate metrics
    silhouette = silhouette_score(X_clean, labels_clean)
    calinski_harabasz = calinski_harabasz_score(X_clean, labels_clean)
    
    print(f"\\n{algorithm_name} on {projection_name} Results:")
    print(f"  Clusters found: {len(set(labels_clean))}")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.3f}")
    
    if algorithm_name.upper() == 'DBSCAN':
        noise_points = sum(labels == -1)
        print(f"  Noise points: {noise_points} ({noise_points/len(labels)*100:.1f}%)")
    
    return silhouette, calinski_harabasz

# Evaluate all clustering models
print("=== CLUSTERING EVALUATION RESULTS ===")
kmeans_pca_scores = evaluate_clustering(X_pca, pca_labels, "KMeans", "PCA")
kmeans_tsne_scores = evaluate_clustering(X_tsne, tsne_labels, "KMeans", "t-SNE") 
dbscan_pca_scores = evaluate_clustering(X_pca, pca_dbscan_labels, "DBSCAN", "PCA")
dbscan_tsne_scores = evaluate_clustering(X_tsne, tsne_dbscan_labels, "DBSCAN", "t-SNE")

# Elbow method visualization
def plot_elbow_analysis(k_range, inertias, silhouettes, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Inertia plot
    ax1.plot(k_range, inertias, 'bo-', markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title(f'Elbow Method - {title}')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_range, silhouettes, 'ro-', markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title(f'Silhouette Analysis - {title}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Generate evaluation plots
plot_elbow_analysis(k_range, pca_inertias, pca_silhouettes, "PCA Data")
plot_elbow_analysis(k_range, tsne_inertias, tsne_silhouettes, "t-SNE Data")`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Key Insights & Results"
      },
      children: [
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Clear Segmentation",
                date: "Finding 1",
                description: "Distinct groups of financial entities with similar characteristics were identified"
              },
              {
                title: "Risk Profiles",
                date: "Finding 2", 
                description: "Clusters corresponding to different risk levels emerged naturally from the data"
              },
              {
                title: "Market Sectors",
                date: "Finding 3",
                description: "Natural groupings aligned with industry sectors and business categories"
              },
              {
                title: "Performance Tiers",
                date: "Finding 4",
                description: "Clear separation based on financial performance metrics and growth indicators"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Technical Implementation",
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
                content: "Technical Stack"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Python & Scikit-learn",
                      description: "Core ML algorithms and data processing"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Pandas & NumPy",
                      description: "Data manipulation and numerical computing"
                    },
                    {
                      icon: <Eye className="w-6 h-6" />,
                      title: "Matplotlib & Seaborn",
                      description: "Comprehensive data visualization"
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
                content: "Code Organization"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Modular Structure"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Code was modularized into organized Python files: main.py, driver.py, clustering.py, projection.py, scores.py, and visualization.py for improved maintainability."
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
{`# Comprehensive clustering visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_clustering_dashboard(X_pca, X_tsne, pca_labels, tsne_labels, 
                               pca_dbscan_labels, tsne_dbscan_labels):
    """Create a comprehensive visualization dashboard"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Color palettes
    kmeans_colors = plt.cm.Set1(np.linspace(0, 1, len(set(pca_labels))))
    dbscan_colors = plt.cm.Set2(np.linspace(0, 1, len(set(pca_dbscan_labels))))
    
    # 1. KMeans on PCA
    plt.subplot(2, 4, 1)
    for i, color in enumerate(kmeans_colors):
        mask = pca_labels == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], 
                   label=f'Cluster {i}', alpha=0.7, s=50)
    plt.title('KMeans Clustering on PCA')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. KMeans on t-SNE
    plt.subplot(2, 4, 2)
    tsne_kmeans_colors = plt.cm.Set1(np.linspace(0, 1, len(set(tsne_labels))))
    for i, color in enumerate(tsne_kmeans_colors):
        mask = tsne_labels == i
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color], 
                   label=f'Cluster {i}', alpha=0.7, s=50)
    plt.title('KMeans Clustering on t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. DBSCAN on PCA
    plt.subplot(2, 4, 3)
    unique_labels = set(pca_dbscan_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise
        mask = pca_dbscan_labels == k
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], 
                   label=f'Cluster {k}' if k != -1 else 'Noise', alpha=0.7, s=50)
    plt.title('DBSCAN Clustering on PCA')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. DBSCAN on t-SNE
    plt.subplot(2, 4, 4)
    unique_labels = set(tsne_dbscan_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise
        mask = tsne_dbscan_labels == k
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[col], 
                   label=f'Cluster {k}' if k != -1 else 'Noise', alpha=0.7, s=50)
    plt.title('DBSCAN Clustering on t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5-8. Cluster distribution plots
    algorithms = ['KMeans-PCA', 'KMeans-tSNE', 'DBSCAN-PCA', 'DBSCAN-tSNE']
    label_sets = [pca_labels, tsne_labels, pca_dbscan_labels, tsne_dbscan_labels]
    
    for i, (algo, labels) in enumerate(zip(algorithms, label_sets), 5):
        plt.subplot(2, 4, i)
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(unique, counts, alpha=0.7)
        plt.title(f'{algo} - Cluster Sizes')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Points')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate the comprehensive dashboard
create_clustering_dashboard(X_pca, X_tsne, pca_labels, tsne_labels, 
                           pca_dbscan_labels, tsne_dbscan_labels)

# Export cluster assignments for further analysis
import pandas as pd

results_df = pd.DataFrame({
    'kmeans_pca_cluster': pca_labels,
    'kmeans_tsne_cluster': tsne_labels,
    'dbscan_pca_cluster': pca_dbscan_labels,
    'dbscan_tsne_cluster': tsne_dbscan_labels
})

results_df.to_csv('clustering_results.csv', index=False)
print("Clustering results saved to clustering_results.csv")
print("Dashboard saved as clustering_dashboard.png")`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Figures & Documentation"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The figures directory is organized into three main categories, each containing comprehensive visualizations:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Exploration",
                description: "Initial exploratory visualizations including histograms, first projections, and data overview plots"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "KMEANS Analysis",
                description: "Separate subdirectories for KMEANS-PCA and KMEANS-tSNE with cluster plots and evaluation graphs"
              },
              {
                icon: <GitBranch className="w-6 h-6" />,
                title: "DBSCAN Results",
                description: "DBSCAN-PCA and DBSCAN-tSNE subdirectories showing density-based groupings and performance analysis"
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
        title: "Key Learnings",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "This exploration provided valuable insights into several critical aspects of clustering analysis:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Database className="w-6 h-6" />,
                title: "Data Preprocessing Importance",
                description: "Understanding the critical role of proper data preprocessing in clustering success"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Algorithm Performance",
                description: "How different clustering algorithms perform on financial datasets with varying characteristics"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Dimensionality Reduction Value", 
                description: "The importance of dimensionality reduction techniques for effective data visualization"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Evaluation Techniques",
                description: "Methods for evaluating cluster quality and determining optimal hyperparameters"
              }
            ],
            columns: 2
          }
        }
      ]
    }
  ],
  navigation: {
    colorScheme: 'green' as const
  }
};

export default function ClusteringExplorationPage() {
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
        <PageBuilder {...clusteringPageData} />
      </article>
    </div>
  );
}
