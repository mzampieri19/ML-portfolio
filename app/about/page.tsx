import Link from "next/link";
import { Brain, Mail, Github, Linkedin, Download, Award, BookOpen, Users } from "lucide-react";

export default function AboutPage() {
  const experience = [
    {
      title: "Machine Learning Engineer",
      company: "TechCorp AI",
      period: "2023 - Present",
      description: "Developing production ML systems for computer vision and NLP applications",
      achievements: [
        "Improved model accuracy by 15% through advanced feature engineering",
        "Deployed 8 ML models to production serving 1M+ users",
        "Led team of 3 junior ML engineers"
      ]
    },
    {
      title: "Data Scientist",
      company: "DataTech Solutions",
      period: "2022 - 2023",
      description: "Built predictive models for business intelligence and customer analytics",
      achievements: [
        "Created recommendation system increasing revenue by 12%",
        "Automated data pipeline reducing processing time by 60%",
        "Presented findings to C-level executives"
      ]
    }
  ];

  const education = [
    {
      degree: "M.S. in Computer Science",
      school: "Stanford University",
      period: "2020 - 2022",
      focus: "Machine Learning & AI",
      gpa: "3.9/4.0"
    },
    {
      degree: "B.S. in Mathematics",
      school: "UC Berkeley",
      period: "2016 - 2020",
      focus: "Statistics & Applied Mathematics",
      gpa: "3.8/4.0"
    }
  ];

  const skills = [
    { category: "Programming", items: ["Python", "R", "JavaScript", "SQL", "C++"] },
    { category: "ML/AI", items: ["PyTorch", "TensorFlow", "Scikit-learn", "Keras", "XGBoost"] },
    { category: "Data", items: ["Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly"] },
    { category: "Cloud", items: ["AWS", "Google Cloud", "Azure", "Docker", "Kubernetes"] },
    { category: "Tools", items: ["Git", "Jupyter", "MLflow", "Apache Spark", "Tableau"] }
  ];

  const certifications = [
    "AWS Certified Machine Learning - Specialty",
    "Google Cloud Professional Data Engineer",
    "Deep Learning Specialization - Coursera",
    "Advanced Machine Learning - Stanford Online"
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
            Passionate Machine Learning Engineer with 3+ years of experience building intelligent systems 
            that solve real-world problems. I specialize in deep learning, computer vision, and NLP.
          </p>
          
          {/* Contact Links */}
          <div className="flex justify-center space-x-4 mt-8">
            <a 
              href="mailto:your.email@example.com"
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Mail className="w-4 h-4" />
              <span>Email</span>
            </a>
            <a 
              href="https://github.com"
              className="flex items-center space-x-2 px-4 py-2 bg-slate-800 text-white rounded-lg hover:bg-slate-700 transition-colors"
            >
              <Github className="w-4 h-4" />
              <span>GitHub</span>
            </a>
            <a 
              href="https://linkedin.com"
              className="flex items-center space-x-2 px-4 py-2 bg-blue-700 text-white rounded-lg hover:bg-blue-800 transition-colors"
            >
              <Linkedin className="w-4 h-4" />
              <span>LinkedIn</span>
            </a>
            <a 
              href="/resume.pdf"
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
                <p className="text-slate-600 dark:text-slate-300 mb-2">{edu.focus}</p>
                <div className="flex justify-between items-center">
                  <span className="text-slate-500 dark:text-slate-400">{edu.period}</span>
                  <span className="text-green-600 dark:text-green-400 font-medium">GPA: {edu.gpa}</span>
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

        {/* Certifications Section */}
        <section>
          <div className="flex items-center space-x-3 mb-8">
            <Award className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white">Certifications</h2>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {certifications.map((cert, index) => (
              <div key={index} className="flex items-center space-x-3 bg-white dark:bg-slate-800 rounded-lg p-4 shadow-md border border-slate-200 dark:border-slate-700">
                <Award className="w-5 h-5 text-green-600 dark:text-green-400" />
                <span className="text-slate-700 dark:text-slate-300">{cert}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
