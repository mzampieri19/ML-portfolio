import Link from "next/link";
import { Brain, Mail, Github, Linkedin, Download, BookOpen, Users } from "lucide-react";

export default function AboutPage() {
  const experience = [
    {
      title: "Software Engineer Intern",
      company: "Imago Rehab",
      period: "May 2025 - Aug 2025",
      description: "Developed software and AI solutions for the leading virtual physical therapy provider company. Aided in maintaining code, creating features and designing the product and system",
      achievements: [
        "Made numerous improvements on the codebase, including bug fixes and feature enhancements",
        "Worked alongside a team of other full stack engineers directly leading to an increased investor budget of over 100%",
      ]
    },
    {
      title: "Machine Learning Engineer Intern",
      company: "TAMID Group",
      period: "Jan 2025 - May 2025",
      description: "Created various CNN models to predict plastics for a startup company.",
      achievements: [
        "Deployed KNN, Random Forest, LSTM and CNN models to production all with > 90% accuracy",
        "Revolutionized and automated the data preparation process, reducing time by a great margin",
        "Presented findings to C-level executives"
      ]
    },
    {
      title: "AI Fellow",
      company: "Real Salary",
      period: "Aug 2024 - Dec 2024",
      description: "Created a novel AI model to predict industry and sector for a nationwide job searching platform.",
      achievements: [
        "Prepared and cleaned large datasets consisting of over 70,000 samples to train the AI model",
        "Achieved a model accuracy of over 85% on unseen data",
        "Coordinated with a team of 5 peers to develop the model and present findings to C-level executives",
      ]
    }
  ];

  const education = [
    {
      degree: "B.S. in Computer Science",
      school: "Brandeis University",
      period: "2022 - 2025",
    },
    {
      degree: "B.S. in Applied Mathematics",
      school: "Brandeis University",
      period: "2022 - 2025",
    },
    {
      degree: "Certificate of Foundations in Machine Learning",
      school: "Cornell University",
      period: "Summer 2024",
    }
  ];

  const skills = [
    { category: "Programming", items: ["Python", "Java", "JavaScript", "Dart", "React"] },
    { category: "ML/AI", items: ["PyTorch", "TensorFlow", "Scikit-learn", "Keras"] },
    { category: "Data", items: ["Pandas", "NumPy", "Matplotlib", "Seaborn"] },
    { category: "Cloud", items: ["AWS", "Google Cloud", "Azure", "Docker"] },
    { category: "Tools", items: ["Git", "Jupyter"] },
    { category: "Soft Skills", items: ["Teamwork", "Communication", "Problem Solving", "Adaptability", "Leadership"] },
    { category: "Languages", items: ["English", "Italian"] }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-2">
              <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              <span className="text-xl font-bold text-slate-900 dark:text-white">ML Portfolio</span>
            </Link>
            <div className="hidden md:flex space-x-8">
              <Link href="/" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Home
              </Link>
              <Link href="/projects" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Projects
              </Link>
              <Link href="/topics" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Topics
              </Link>
              <Link href="/visualize" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Visualize
              </Link>
              <Link href="/resources" className="text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                Resources
              </Link>
              <Link href="/about" className="text-blue-600 dark:text-blue-400 font-medium">
                About
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <div className="w-32 h-32 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mx-auto mb-8 flex items-center justify-center">
            <span className="text-4xl font-bold text-white">ML</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-6">
            About Me
          </h1>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto leading-relaxed">
            Senior undergraduate student at Brandeis University majoring in Computer Science and Applied Mathematics. Passionate about machine learning, AI, and software engineering. I have experience in developing production ML systems, creating intelligent applications, and working with large datasets.
          </p>
          
          {/* Contact Links */}
          <div className="flex justify-center space-x-4 mt-8">
            <a 
              href="mailto:michelangeloz03@gmail.com"
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Mail className="w-4 h-4" />
              <span>Email</span>
            </a>
            <a 
              href="https://github.com/mzampieri19"
              className="flex items-center space-x-2 px-4 py-2 bg-slate-800 text-white rounded-lg hover:bg-slate-700 transition-colors"
            >
              <Github className="w-4 h-4" />
              <span>GitHub</span>
            </a>
            <a 
              href="https://www.linkedin.com/in/michelangelo-zampieri-87675b288/"
              className="flex items-center space-x-2 px-4 py-2 bg-blue-700 text-white rounded-lg hover:bg-blue-800 transition-colors"
            >
              <Linkedin className="w-4 h-4" />
              <span>LinkedIn</span>
            </a>
            <a 
              href="/Users/michelangelozampieri/Desktop/ml-portfolio/app/about/Michelangelo_Zampieri_ML_Resume.pdf"
              className="flex items-center space-x-2 px-4 py-2 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
            >
              <Download className="w-4 h-4" />
              <span>Resume</span>
            </a>
          </div>
        </div>

        {/* Experience Section */}
        <section className="mb-16">
          <div className="flex items-center space-x-3 mb-8">
            <Users className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white">Experience</h2>
          </div>
          <div className="space-y-8">
            {experience.map((job, index) => (
              <div key={index} className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-slate-700">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-slate-900 dark:text-white">{job.title}</h3>
                    <p className="text-blue-600 dark:text-blue-400 font-medium">{job.company}</p>
                  </div>
                  <span className="text-slate-500 dark:text-slate-400 font-medium">{job.period}</span>
                </div>
                <p className="text-slate-600 dark:text-slate-300 mb-4">{job.description}</p>
                <ul className="space-y-2">
                  {job.achievements.map((achievement, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full mt-2 flex-shrink-0"></span>
                      <span className="text-slate-600 dark:text-slate-300">{achievement}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* Education Section */}
        <section className="mb-16">
          <div className="flex items-center space-x-3 mb-8">
            <BookOpen className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white">Education</h2>
          </div>
          <div className="grid md:grid-cols-2 gap-6">
            {education.map((edu, index) => (
              <div key={index} className="bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-lg border border-slate-200 dark:border-slate-700">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">{edu.degree}</h3>
                <p className="text-blue-600 dark:text-blue-400 font-medium mb-2">{edu.school}</p>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500 dark:text-slate-400">{edu.period}</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Skills Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-8 text-center">Skills & Technologies</h2>
          <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-6">
            {skills.map((skillGroup, index) => (
              <div key={index} className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-slate-700">
                <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">{skillGroup.category}</h3>
                <ul className="space-y-2">
                  {skillGroup.items.map((skill, idx) => (
                    <li key={idx} className="text-slate-600 dark:text-slate-300 text-sm">{skill}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
