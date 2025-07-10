# Using Individual Components in MDX

This document demonstrates how to use the individual content components in your MDX files.

## Basic Components

### Headings
You can use the Heading component with different levels:

<Heading level={2}>This is a Level 2 Heading</Heading>
<Heading level={3}>This is a Level 3 Heading</Heading>

### Paragraphs
Use the Paragraph component for better control over text styling:

<Paragraph size="lg" emphasis>
This is a large, emphasized paragraph that stands out from regular text.
</Paragraph>

<Paragraph>
This is a regular paragraph with standard styling.
</Paragraph>

### Lists
Create styled lists with custom icons:

<List variant="bullet">
  <ListItem>First item in the list</ListItem>
  <ListItem>Second item with more content</ListItem>
  <ListItem>Third item to complete the example</ListItem>
</List>

## Advanced Components

### Highlight Boxes
Use highlight boxes to draw attention to important information:

<HighlightBox variant="info" title="Important Note" icon="💡">
  <Paragraph>
    This is an information box that helps highlight key points in your content.
  </Paragraph>
</HighlightBox>

<HighlightBox variant="success" title="Great Results">
  <Paragraph>
    Use success variant to highlight positive outcomes and achievements.
  </Paragraph>
</HighlightBox>

### Quotes
Add impactful quotes to your content:

<Quote author="Albert Einstein" source="Theoretical Physics">
Imagination is more important than knowledge. For knowledge is limited, whereas imagination embraces the entire world.
</Quote>

### Feature Lists
Showcase key features or benefits:

<FeatureList 
  columns={2}
  features={[
    {
      icon: "🚀",
      title: "Fast Performance",
      description: "Optimized algorithms for rapid processing"
    },
    {
      icon: "🎯",
      title: "High Accuracy",
      description: "Precision-tuned for maximum reliability"
    },
    {
      icon: "🔧",
      title: "Easy Integration",
      description: "Simple API for seamless implementation"
    },
    {
      icon: "📊",
      title: "Rich Analytics",
      description: "Comprehensive metrics and reporting"
    }
  ]}
/>

### Metrics Grid
Display key performance indicators:

<MetricsGrid 
  columns={3}
  metrics={[
    { label: "Accuracy", value: "94.5%", change: "+2.3%", trend: "up" },
    { label: "Processing Time", value: "12ms", change: "-5ms", trend: "up" },
    { label: "Model Size", value: "24MB", change: "Same", trend: "neutral" }
  ]}
/>

### Timeline
Show project progression or milestones:

<Timeline 
  items={[
    {
      title: "Project Kickoff",
      date: "Jan 2025",
      description: "Initial planning and team formation"
    },
    {
      title: "Data Collection",
      date: "Feb 2025", 
      description: "Gathered and preprocessed training data"
    },
    {
      title: "Model Development",
      date: "Mar 2025",
      description: "Built and trained the machine learning model"
    },
    {
      title: "Deployment",
      date: "Apr 2025",
      description: "Released to production environment"
    }
  ]}
/>

## Layout Components

### Sections
Organize content into logical sections:

<Section title="Technical Details" background>
  <Paragraph>
    Content within a section can include any combination of other components.
  </Paragraph>
  
  <List>
    <ListItem>Structured organization</ListItem>
    <ListItem>Optional background styling</ListItem>
    <ListItem>Clear visual separation</ListItem>
  </List>
</Section>

### Two Column Layout
Create side-by-side content:

<TwoColumn 
  ratio="1:1"
  left={
    <>
      <Heading level={3}>Left Column</Heading>
      <Paragraph>
        This content appears in the left column of the two-column layout.
      </Paragraph>
    </>
  }
  right={
    <>
      <Heading level={3}>Right Column</Heading>
      <Paragraph>
        This content appears in the right column with equal width.
      </Paragraph>
    </>
  }
/>

### Dividers
Add visual separation between sections:

<Divider variant="solid" />

Content after a solid divider.

<Divider variant="gradient" />

Content after a gradient divider.

## Usage Tips

1. **Consistency**: Use the same color schemes across similar project types
2. **Hierarchy**: Employ proper heading levels for good document structure  
3. **Emphasis**: Use highlight boxes sparingly for maximum impact
4. **Visual Balance**: Mix text components with visual elements like metrics and timelines

These components provide a flexible foundation for creating professional, consistent project documentation while maintaining the power and simplicity of MDX.
