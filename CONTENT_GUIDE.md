# Content Management Guide

## Adding New Projects

### 1. Create a new MDX file
Create a new folder in `app/projects/` with your project slug:
```bash
mkdir app/projects/your-project-name
```

Then create a `page.mdx` file in that folder.

### 2. Project Template
Use this template for new projects:

```mdx
import Link from 'next/link';
import { ArrowLeft, ExternalLink, Github, Calendar, Clock, Tag } from 'lucide-react';
import CodeBlock from '../../components/CodeBlock';

export const metadata = {
  title: 'Your Project Title - ML Portfolio',
  description: 'Brief description of your project',
};

# Your Project Title

<div className="flex items-center justify-between mb-8 p-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl border border-blue-200 dark:border-blue-800">
  <div>
    <div className="flex items-center space-x-4 text-sm text-slate-600 dark:text-slate-400 mb-2">
      <div className="flex items-center space-x-1">
        <Calendar className="w-4 h-4" />
        <span>Month Year</span>
      </div>
      <div className="flex items-center space-x-1">
        <Clock className="w-4 h-4" />
        <span>X min read</span>
      </div>
    </div>
    <p className="text-lg text-slate-700 dark:text-slate-300">
      Project description
    </p>
  </div>
  <div className="flex space-x-3">
    <a href="your-github-link" className="...">
      <Github className="w-4 h-4" />
      <span>Code</span>
    </a>
  </div>
</div>

## Your content here...

<CodeBlock language="python" filename="example.py">
{`# Your code here
print("Hello, ML!")`}
</CodeBlock>

---

<div className="mt-12 pt-8 border-t border-slate-200 dark:border-slate-700">
  <Link href="/projects" className="...">
    <ArrowLeft className="w-4 h-4 mr-2" />
    Back to Projects
  </Link>
</div>
```

### 3. Update Projects List
Add your project to the projects array in:
- `app/page.tsx` (for featured projects)
- `app/projects/page.tsx` (for all projects)

## Available Components

### CodeBlock
```tsx
<CodeBlock language="python" filename="optional-filename.py">
{`your code here`}
</CodeBlock>
```

### Interactive Demo
```tsx
<InteractiveDemo />
```

### Custom Styling
All standard markdown elements are pre-styled. You can also use Tailwind CSS classes directly in your MDX.

## Writing Tips

1. **Use clear headings** - They create automatic navigation
2. **Include code examples** - Show, don't just tell
3. **Add interactive elements** - Make it engaging
4. **Use visuals** - Images, diagrams, charts
5. **Link between projects** - Create a connected experience

## File Structure
```
app/
├── projects/
│   ├── page.tsx              # Projects listing
│   ├── project-name/
│   │   └── page.mdx          # Project content
│   └── another-project/
│       └── page.mdx
├── components/
│   ├── CodeBlock.tsx         # Syntax highlighted code
│   └── InteractiveDemo.tsx   # Interactive components
└── page.tsx                  # Homepage
```

## Deployment

The portfolio can be deployed to:
- Vercel (recommended for Next.js)
- Netlify
- GitHub Pages
- Any hosting service that supports Node.js

Simply push your changes to GitHub and connect your repository to your hosting platform.
