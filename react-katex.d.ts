declare module 'react-katex' {
  import * as React from 'react';
  export interface KatexProps {
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => React.ReactNode;
    className?: string;
    style?: React.CSSProperties;
  }
  export const InlineMath: React.FC<KatexProps>;
  export const BlockMath: React.FC<KatexProps>;
} 