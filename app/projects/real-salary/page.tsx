import { Users, Database, Brain, TrendingUp, Code, BarChart3, Target, Zap, GitBranch, Settings, Trophy, AlertCircle, CheckCircle, Eye } from 'lucide-react';
import PageBuilder from '../../components/PageBuilder';
import CodeBlock from '../../components/CodeBlock';

export const metadata = {
  title: 'Real Salary - ML Portfolio',
  description: 'Group work for the BTTAI industry project with the company Real Salary',
};

const realSalaryPageData = {
  title: "Real Salary",
  header: {
    date: "Fall 2024",
    readTime: "6 min read",
    description: "Data exploration and analysis with OpenAI LLM integration for Real Salary company",
    githubUrl: "https://github.com/mzampieri19/real-salary",
    gradientFrom: "from-orange-50 to-red-50",
    gradientTo: "dark:from-orange-900/20 dark:to-red-900/20",
    borderColor: "border-orange-200 dark:border-orange-800",
    collaborators: "Group Project - 4 Team Members"
  },
  tags: {
    items: ['Data Science', 'Group Work', 'Data Analysis', 'LLMs', 'Data Preparation'],
    colorScheme: 'orange' as const
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
          content: "This repository contains all work for Group 12 - Team Real Salary, an industry project completed for the Break Through Tech AI (BTTAI) program in Fall 2024. Our team worked directly with Real Salary, a company focused on salary transparency and data analysis."
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'info' as const,
            title: "Team Composition",
            icon: <Users className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "Authors: Anh Le, Cheyanne Atole, Hanh Pham, Michelangelo Zampieri. Teaching Assistant: Caroline Cunningham. Challenge Advisors: Amy Shratter, Chad Woodrick."
            }
          ]
        }
      ]
    },
    {
      type: 'section' as const,
      props: {
        title: "Project Summary"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "Our team conducted comprehensive data exploration with integration of OpenAI LLM capabilities to analyze job market data and predict industry classifications and job functions from company and role descriptions."
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Target className="w-6 h-6" />,
                title: "Industry Classification",
                description: "Use LLM APIs to predict company industries from descriptions"
              },
              {
                icon: <Brain className="w-6 h-6" />,
                title: "Job Function Mapping",
                description: "Classify job roles into standardized job functions"
              },
              {
                icon: <Database className="w-6 h-6" />,
                title: "Data Pipeline Development",
                description: "Create scalable prediction systems for Real Salary"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Accuracy Optimization",
                description: "Achieve high prediction accuracy through prompt engineering"
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
        title: "Technical Implementation",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The project utilized several datasets and evolved through multiple phases of LLM integration:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "Original Datasets"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Database className="w-6 h-6" />,
                      title: "job_descriptions",
                      description: "Original dataset from Kaggle with comprehensive job data"
                    },
                    {
                      icon: <Settings className="w-6 h-6" />,
                      title: "filtered_country_work_type",
                      description: "Data filtered by country and work type, nulls removed"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "mapped_sectors",
                      description: "Mapped industry classifications to real-salary categories"
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
                content: "LLM Integration Journey"
              },
              {
                type: 'timeline' as const,
                props: {
                  items: [
                    {
                      title: "Word Frequency Analysis",
                      date: "Phase 1",
                      description: "Initial analysis found relationships but weren't immediately useful"
                    },
                    {
                      title: "LLAMA API Testing",
                      date: "Phase 2",
                      description: "Successful testing but required payment for continued use"
                    },
                    {
                      title: "OLLAMA Implementation",
                      date: "Phase 3",
                      description: "Local execution, free to use, better performance"
                    }
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
        title: "Current Results"
      },
      children: [
        {
          type: 'metrics' as const,
          props: {
            metrics: [
              { label: "Industry Prediction Accuracy", value: "65%", change: "Average", trend: "up" },
              { label: "Response Time", value: "1:37", change: "50 entries", trend: "neutral" },
              { label: "Improvement", value: "60% → 65%", change: "Methodology refinement", trend: "up" },
              { label: "Cost Efficiency", value: "100%", change: "Local execution", trend: "up" }
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
                content: "Key Outputs"
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Company Dictionary"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "prediction_dictionary.json maps company names to predicted industries as final deliverable for challenge advisors."
                  }
                ]
              },
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Job Function Mapping"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "job_function.json maps job roles to predicted job functions using descriptions, responsibilities, and required skills."
                  }
                ]
              }
            ],
            right: [
              {
                type: 'heading' as const,
                props: { level: 3 },
                content: "OLLAMA Advantages"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "Local Execution",
                      description: "No internet connectivity issues or dependencies"
                    },
                    {
                      icon: <Trophy className="w-6 h-6" />,
                      title: "Free to Use",
                      description: "No API costs or usage limitations"
                    },
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Better Performance",
                      description: "Optimized for our specific use case and data"
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
        title: "Project Structure",
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
                content: "Core Analysis Files"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "analysis.ipynb",
                      description: "Main analysis for industry prediction with shorter industry list"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "job_function.ipynb",
                      description: "Notebook for job function prediction implementation"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "accuracy-test-ollama.ipynb",
                      description: "First round of accuracy testing using OLLAMA model"
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
                content: "Testing Directory"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Code className="w-6 h-6" />,
                      title: "api.py",
                      description: "Metro area prediction API test (scrapped due to data issues)"
                    },
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "llama_api.ipynb",
                      description: "LLAMA3 API testing (discontinued for cost reasons)"
                    },
                    {
                      icon: <Zap className="w-6 h-6" />,
                      title: "ollama.ipynb",
                      description: "OLLAMA API testing and implementation"
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
        title: "Methodology Evolution"
      },
      children: [
        {
          type: 'timeline' as const,
          props: {
            items: [
              {
                title: "Data Understanding",
                date: "Phase 1",
                description: "Initial dataset preview, exploratory analysis, and word frequency analysis across sectors"
              },
              {
                title: "API Development",
                date: "Phase 2",
                description: "Metro area prediction attempt, LLAMA API exploration, and cost-benefit analysis leading to OLLAMA adoption"
              },
              {
                title: "OLLAMA Implementation",
                date: "Phase 3",
                description: "Local model setup, prompt engineering for optimal results, and accuracy testing"
              },
              {
                title: "Production Pipeline",
                date: "Phase 4",
                description: "Dictionary construction, job function classification system, and scalable prediction pipeline development"
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
{`def getSector(prompt):
    """
    Predict company sector using OLLAMA LLM API
    
    Args:
        prompt (str): Company name for sector prediction
        
    Returns:
        str: Predicted sector or 'Unknown' if parsing fails
    """
    sectors_str = ", ".join(sectors_list)
    data = {
        "model": "llama3",
        "messages": [
            {
                "role": "user",
                # What we are asking the API, will be given the name of company in the prompt
                "content": f"Identify the sector for the company named {prompt}. " 
                            "Respond strictly in the format: 'Sector: [best matching sector]'. "
                            f"Select the most appropriate sector from this list: {sectors_str}." 
                            "Do not include any additional information or commentary."
            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    # Make API request to local OLLAMA instance
    response = requests.post(url, headers=headers, json=data)
    response = response.json()["message"]["content"]
    
    # Extract sector from response with error handling
    try:
        sector = str(response).split("Sector: ")[1].split(",")[0]
    except IndexError:
        sector = "Unknown"
        print(f"Warning: Could not parse sector for {prompt}")

    return sector

# Batch processing function for multiple companies
def process_companies_batch(company_list, batch_size=50):
    """Process multiple companies in batches for efficiency"""
    results = {}
    
    for i in range(0, len(company_list), batch_size):
        batch = company_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        for company in batch:
            results[company] = getSector(company)
            
    return results`}
        </CodeBlock>
      )
    },
    {
      type: 'section' as const,
      props: {
        title: "Key Achievements",
        background: true
      },
      children: [
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "65% Prediction Accuracy",
                description: "Industry classification with significant improvement over baseline"
              },
              {
                icon: <Trophy className="w-6 h-6" />,
                title: "Cost-Effective Solution",
                description: "Local OLLAMA implementation eliminates API costs"
              },
              {
                icon: <Database className="w-6 h-6" />,
                title: "Scalable Pipeline",
                description: "Future-ready data processing architecture"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Complete Documentation",
                description: "Reproducible methodology and deliverable artifacts"
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
        title: "Key Findings"
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
                content: "Salary Distribution Analysis"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Industry Variations",
                      description: "Significant salary differences across industries"
                    },
                    {
                      icon: <TrendingUp className="w-6 h-6" />,
                      title: "Experience Impact",
                      description: "Strong correlation between years of experience and compensation"
                    },
                    {
                      icon: <Target className="w-6 h-6" />,
                      title: "Geographic Trends",
                      description: "Location-based salary variations and market differences"
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
                content: "LLM-Enhanced Recommendations"
              },
              {
                type: 'features' as const,
                props: {
                  features: [
                    {
                      icon: <Brain className="w-6 h-6" />,
                      title: "Salary Benchmarking Tool",
                      description: "Develop automated salary comparison features"
                    },
                    {
                      icon: <BarChart3 className="w-6 h-6" />,
                      title: "Market Analysis",
                      description: "Regular market trend analysis using AI insights"
                    },
                    {
                      icon: <Users className="w-6 h-6" />,
                      title: "Personalized Recommendations",
                      description: "Custom salary guidance based on individual profiles"
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
        title: "Team Collaboration",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "This project showcased effective collaboration in a data science team environment with clear role divisions and collaborative development practices."
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Database className="w-6 h-6" />,
                title: "Data Engineering",
                description: "Collaborative data pipeline development and preprocessing"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Analysis Division",
                description: "Team members specialized in different analytical aspects"
              },
              {
                icon: <GitBranch className="w-6 h-6" />,
                title: "Code Review",
                description: "Peer review process for quality assurance and knowledge sharing"
              },
              {
                icon: <Eye className="w-6 h-6" />,
                title: "Documentation",
                description: "Comprehensive project documentation and methodology recording"
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
        title: "Impact and Results"
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The project delivered significant value to Real Salary through comprehensive analysis and scalable automation solutions:"
        },
        {
          type: 'twoColumn' as const,
          props: {
            ratio: '1:1' as const,
            left: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'success' as const,
                  title: "Technical Stack"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Python, Pandas/NumPy for data manipulation, Matplotlib/Seaborn for visualization, OpenAI API for LLM integration, Jupyter Notebooks for development, and Git for collaboration."
                  }
                ]
              }
            ],
            right: [
              {
                type: 'highlight' as const,
                props: {
                  variant: 'info' as const,
                  title: "Key Learnings"
                },
                children: [
                  {
                    type: 'paragraph' as const,
                    content: "Real-world data constraints, stakeholder communication, AI integration in workflows, and professional team dynamics in data science projects."
                  }
                ]
              }
            ]
          }
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Data-Driven Insights",
                description: "Comprehensive analysis of salary trends and market patterns"
              },
              {
                icon: <Zap className="w-6 h-6" />,
                title: "Automation",
                description: "Streamlined analysis processes using LLM integration"
              },
              {
                icon: <Database className="w-6 h-6" />,
                title: "Scalable Solutions",
                description: "Reusable code and methodologies for future analysis"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Strategic Recommendations",
                description: "Actionable insights for business development and growth"
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
        title: "Future Applications",
        background: true
      },
      children: [
        {
          type: 'paragraph' as const,
          content: "The methodologies developed in this project have broader applications across various domains:"
        },
        {
          type: 'features' as const,
          props: {
            features: [
              {
                icon: <Users className="w-6 h-6" />,
                title: "HR Analytics",
                description: "Employee compensation analysis and workforce planning"
              },
              {
                icon: <BarChart3 className="w-6 h-6" />,
                title: "Market Research",
                description: "Industry salary trend analysis and competitive intelligence"
              },
              {
                icon: <Target className="w-6 h-6" />,
                title: "Career Guidance",
                description: "Personal career development insights and planning tools"
              },
              {
                icon: <TrendingUp className="w-6 h-6" />,
                title: "Policy Research",
                description: "Economic and labor market analysis for policy making"
              }
            ],
            columns: 2
          }
        },
        {
          type: 'highlight' as const,
          props: {
            variant: 'success' as const,
            title: "Project Recognition",
            icon: <Trophy className="w-6 h-6" />
          },
          children: [
            {
              type: 'paragraph' as const,
              content: "This project was completed as part of the Break Through Tech AI industry partnership program, demonstrating successful collaboration between academic learning and real-world business applications. The work showcases practical AI implementation in industry settings and effective team-based data science project execution."
            }
          ]
        }
      ]
    }
  ],
  navigation: {
    colorScheme: 'orange' as const
  }
};

export default function RealSalaryPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-orange-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <a href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </a>
            <div className="hidden md:flex space-x-8">
              <a href="/" className="text-slate-700 dark:text-slate-300 hover:text-orange-600 dark:hover:text-orange-400 transition-colors">
                Home
              </a>
              <a href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-orange-600 dark:hover:text-orange-400 transition-colors">
                Projects
              </a>
            </div>
          </div>
        </div>
      </nav>
      
      <article className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 prose prose-slate dark:prose-invert prose-lg max-w-none">
        <PageBuilder {...realSalaryPageData} />
      </article>
    </div>
  );
}
