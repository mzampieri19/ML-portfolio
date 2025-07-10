import { Target, Lightbulb, Award, Users } from 'lucide-react';
import PageBuilder from '../../components/PageBuilder';

export const metadata = {
  title: 'Example Project - ML Portfolio',
  description: 'Example project demonstrating the PageBuilder component system',
};

const examplePageData = {
  title: "Example ML Project",
  header: {
    date: "Spring 2025",
    readTime: "8 min read",
    description: "A comprehensive machine learning project demonstrating modern AI techniques",
    githubUrl: "https://github.com/example/project",
    demoUrl: "https://example-demo.com",
    gradientFrom: "from-blue-50 to-purple-50",
    gradientTo: "dark:from-blue-900/20 dark:to-purple-900/20",
    borderColor: "border-blue-200 dark:border-blue-800",
    collaborators: "Team Project"
  },
  tags: {
    items: ['Machine Learning', 'PyTorch', 'Computer Vision', 'Deep Learning'],
    colorScheme: 'blue' as const
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
          content: "This project demonstrates the power of modern machine learning techniques applied to real-world problems."
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Target className="w-6 h-6" />,
                title: "High Accuracy",
                description: "Achieved 95.2% accuracy on validation dataset"
              },
              {
                icon: <Lightbulb className="w-6 h-6" />,
                title: "Novel Approach", 
                description: "Implemented innovative neural architecture"
              },
              {
                icon: <Award className="w-6 h-6" />,
                title: "Award Winning",
                description: "Recognized at ML conference 2024"
              },
              {
                icon: <Users className="w-6 h-6" />,
                title: "Team Collaboration",
                description: "Successful multi-disciplinary team effort"
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
        title: "Key Results"
      },
      children: [
        {
          type: 'metrics' as const,
          props: {
            metrics: [
              { label: "Accuracy", value: "95.2%", change: "+3.1%", trend: "up" },
              { label: "F1-Score", value: "0.94", change: "+0.05", trend: "up" },
              { label: "Inference Time", value: "12ms", change: "-8ms", trend: "up" },
              { label: "Model Size", value: "24MB", change: "-12MB", trend: "up" }
            ],
            columns: 4
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
            content: "Technical Approach"
          },
          {
            type: 'list' as const,
            children: [
              { type: 'listItem' as const, content: "Data preprocessing and augmentation" },
              { type: 'listItem' as const, content: "Custom neural network architecture" },
              { type: 'listItem' as const, content: "Transfer learning techniques" },
              { type: 'listItem' as const, content: "Hyperparameter optimization" }
            ]
          }
        ],
        right: [
          {
            type: 'heading' as const,
            props: { level: 3 },
            content: "Key Innovations"
          },
          {
            type: 'highlight' as const,
            props: {
              variant: 'info' as const,
              title: "Novel Architecture"
            },
            children: [
              {
                type: 'paragraph' as const,
                content: "We developed a new attention mechanism that improves performance by 15% over baseline models."
              }
            ]
          }
        ]
      }
    },
    {
      type: 'section' as const,
      props: {
        title: "Project Timeline"
      },
      children: [
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Research Phase",
                date: "Week 1-2",
                description: "Literature review and problem analysis"
              },
              {
                title: "Data Collection",
                date: "Week 3-4",
                description: "Dataset preparation and preprocessing"
              },
              {
                title: "Model Development",
                date: "Week 5-8",
                description: "Architecture design and implementation"
              },
              {
                title: "Evaluation",
                date: "Week 9-10",
                description: "Testing and performance optimization"
              }
            ]
          }
        }
      ]
    },
    {
      type: 'quote' as const,
      props: {
        author: "Project Lead",
        source: "Team Retrospective"
      },
      content: "This project pushed the boundaries of what we thought was possible with our current technology stack."
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
          content: "This project successfully demonstrated the effectiveness of combining modern ML techniques with innovative architectural designs."
        },
        {
          type: 'paragraph' as const,
          content: "The results have implications for future research in this domain and provide a solid foundation for commercial applications."
        }
      ]
    }
  ],
  navigation: {
    colorScheme: 'blue' as const
  }
};

export default function ExampleProjectPage() {
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
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <PageBuilder {...examplePageData} />
      </article>
    </div>
  );
}
