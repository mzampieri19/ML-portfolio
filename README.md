# ML Portfolio - Michelangelo Zampieri

A comprehensive Machine Learning portfolio website showcasing projects, interactive visualizations, educational resources, and in-depth explorations of AI/ML concepts. Built with Next.js, TensorFlow.js, and modern web technologies.

![ML Portfolio](https://img.shields.io/badge/Next.js-15.3.4-black?style=flat-square&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square&logo=typescript)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22.0-orange?style=flat-square&logo=tensorflow)
![React](https://img.shields.io/badge/React-18.3.1-blue?style=flat-square&logo=react)

## 🎯 Project Overview

This portfolio demonstrates expertise in machine learning through a multi-faceted approach:
- **Hands-on Projects**: From-scratch implementations of ML models including GPT, Diffusion Models, and Reinforcement Learning
- **Interactive Learning**: Real-time model visualization and experimentation tools
- **Educational Content**: Comprehensive guides on ML topics and algorithms
- **Professional Experience**: Showcasing real-world applications and achievements

## 🏗️ Architecture & Tech Stack

### Frontend Technologies
- **Framework**: Next.js 15.3.4 with App Router
- **Language**: TypeScript for type safety
- **Styling**: Tailwind CSS for responsive design
- **Animations**: Framer Motion for smooth interactions
- **Icons**: Lucide React for consistent iconography

### ML & Visualization
- **TensorFlow.js**: For client-side machine learning
- **D3.js**: For custom data visualizations
- **Plotly.js**: For interactive charts and graphs
- **TensorFlow.js Vis**: For model visualization

### Content & Documentation
- **MDX**: For rich, interactive documentation
- **React Syntax Highlighter**: For code blocks
- **KaTeX**: For mathematical equations
- **Prism.js**: For syntax highlighting

## 📁 Project Structure

```
ml-portfolio/
├── app/                          # Next.js App Router pages
│   ├── components/               # Reusable React components
│   │   ├── CodeBlock.tsx        # Syntax-highlighted code display
│   │   ├── InteractiveDemo.tsx  # Interactive ML demonstrations
│   │   ├── PageBuilder.tsx      # Dynamic page content builder
│   │   └── ProjectComponents.tsx # Project-specific components
│   ├── projects/                # Individual project pages
│   │   ├── dqn-flappy-bird/    # Reinforcement Learning project
│   │   ├── custom-diffusion-model/ # Generative AI project
│   │   ├── custom-gpt-llm/     # Large Language Model project
│   │   └── ...                 # Additional projects
│   ├── topics/                  # ML concept explanations
│   │   ├── neural-networks/    # Deep learning fundamentals
│   │   ├── transformers/       # Attention mechanisms
│   │   ├── diffusion-models/   # Generative modeling
│   │   └── ...                 # Additional topics
│   ├── visualize/              # Interactive ML tools
│   │   ├── neural-network/     # Neural network playground
│   │   ├── linear-regression/  # Regression visualizer
│   │   └── components/         # Visualization components
│   ├── resources/              # Learning resources and tools
│   └── about/                  # Professional information
├── hooks/                       # Custom React hooks
│   ├── useDataset.ts           # Dataset management
│   ├── useModelTraining.ts     # Training utilities
│   ├── useParameterControl.ts  # Parameter tuning
│   └── useVisualization.ts     # Visualization helpers
├── lib/                        # Utility libraries
│   └── visualize/              # ML visualization utilities
│       ├── algorithms.ts       # ML algorithm implementations
│       ├── datasets.ts         # Sample datasets
│       ├── types.ts            # TypeScript definitions
│       └── utils.ts            # Helper functions
├── process/                    # Development documentation
└── tasks/                      # Project planning and PRDs
```

## 🚀 Features by Section

### 1. Projects Section (`/projects`)
**Purpose**: Showcase comprehensive ML projects with detailed explanations, code, and results.

**Current Projects**:
- **DQN Flappy Bird**: Deep Reinforcement Learning implementation
- **Custom Diffusion Model**: From-scratch generative model with UNET, DDPM
- **Custom GPT LLM**: Transformer architecture implementation
- **TAMID Image Classifier**: CNN for plastic type classification
- **Image Classifier**: Satellite image classification using CNNs
- **Clustering Exploration**: Financial data analysis with various clustering techniques

**Features**:
- MDX-based content with interactive code blocks
- Mathematical equation rendering with KaTeX
- Downloadable code and documentation
- Project categorization and filtering
- Performance metrics and visualizations

### 2. Topics Section (`/topics`)
**Purpose**: Educational content explaining ML concepts with mathematical foundations and practical examples.

**Current Topics**:
- Neural Network Architectures
- Convolutional Neural Networks (CNNs)
- Transformers and Attention Mechanisms
- Diffusion Models and DDPM
- Reinforcement Learning
- Clustering Algorithms
- Loss Functions and Optimization
- Regularization Techniques

**Features**:
- Interactive search and filtering
- Difficulty-based categorization
- Project cross-references
- Mathematical notation with KaTeX
- Visual diagrams and illustrations

### 3. Visualize Section (`/visualize`)
**Purpose**: Interactive ML model experimentation and real-time visualization.

**Current Tools**:
- **Neural Network Playground**: Interactive architecture design
- **Linear Regression Explorer**: Polynomial fitting and regularization

**In Development**:
- Decision Tree Visualizer
- Clustering Sandbox
- CNN Architecture Builder
- RNN Sequence Modeling
- Optimization Algorithm Visualizer

**Features**:
- Real-time parameter adjustment
- Live model training visualization
- Decision boundary plotting
- Performance metric tracking
- Educational tooltips and explanations

### 4. Resources Section (`/resources`)
**Purpose**: Curated collection of ML tools, libraries, and learning materials.

**Categories**:
- IDEs & Editors (VS Code, Jupyter, PyCharm)
- ML Frameworks (TensorFlow, PyTorch, Scikit-learn)
- Data Tools (Pandas, NumPy, Matplotlib)
- Cloud Platforms (AWS, Google Cloud, Azure)
- Learning Resources (Courses, Books, Tutorials)

**Features**:
- Categorized resource organization
- Direct links to official documentation
- Tool comparisons and recommendations
- Beginner-friendly annotations

### 5. About Section (`/about`)
**Purpose**: Professional background, experience, and achievements.

**Content**:
- Work experience (Imago Rehab, TAMID Group, Real Salary)
- Education (Brandeis University - CS & Applied Math)
- Technical skills and expertise
- Downloadable resume
- Contact information

## 🛠️ Getting Started

### Prerequisites
- Node.js 18.0 or later
- npm, yarn, pnpm, or bun package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-portfolio
```

2. Install dependencies:
```bash
npm install
# or
yarn install
# or
pnpm install
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## 🔧 Development

### Adding New Projects

1. Create a new folder in `app/projects/your-project-name/`
2. Add a `page.mdx` file using the project template from `/process/CONTENT_GUIDE.md`
3. Update the projects array in `/app/projects/page.tsx`
4. Add relevant tags and categorization

### Adding New Topics

1. Create a new folder in `app/topics/your-topic-name/`
2. Follow the topic template structure
3. Update the topics array in `/app/topics/page.tsx`
4. Include mathematical notation using KaTeX syntax

### Creating Interactive Visualizations

1. Add new visualization components in `app/visualize/components/`
2. Implement corresponding algorithms in `lib/visualize/algorithms.ts`
3. Create custom hooks for state management in `hooks/`
4. Update the models array in `/app/visualize/page.tsx`

## 🚀 Deployment 

This project is deployed with Vercel and can be found at this [link][https://ml-portfolio-713krgb1m-mzampieri19s-projects.vercel.app/]

## 🚧 Current Development Status

### ✅ Completed Features
- **Core Infrastructure**: Next.js setup with TypeScript and Tailwind
- **Project Showcase**: Complete project documentation with MDX
- **Educational Content**: Comprehensive topic explanations
- **Basic Visualizations**: Neural network and linear regression tools
- **Resource Collection**: Curated ML learning materials
- **Professional Portfolio**: Experience and background information

### 🔨 In Progress
- **Enhanced Visualizations**: 
  - Decision tree interactive builder
  - Advanced clustering visualization
  - CNN architecture designer
- **Performance Optimizations**: 
  - Code splitting for visualization components
  - Lazy loading for heavy computational tools
- **Content Expansion**: 
  - Additional project case studies
  - More detailed mathematical explanations


## 🎯 Project Goals

### Educational Impact
- Make complex ML concepts accessible through visualization
- Provide hands-on learning experiences
- Bridge the gap between theory and practice
- Support various learning styles (visual, kinesthetic, analytical)

### Technical Excellence
- Demonstrate modern web development practices
- Showcase ML implementation skills
- Maintain high code quality and documentation standards
- Optimize for performance and accessibility

### Professional Development
- Establish expertise in ML and web technologies
- Create a platform for sharing knowledge and insights
- Build a foundation for future educational initiatives
- Connect with the broader ML community

## 📞 Contact

**Michelangelo Zampieri**
- Email: [michelangeloz03@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/michelangelo-zampieri-87675b288/]
- GitHub: [https://github.com/mzampieri19]

---
