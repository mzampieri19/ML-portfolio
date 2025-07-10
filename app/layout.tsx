import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ML Portfolio - Michelangelo Zampieri",
  description: "Machine Learning and AI project portfolio showcasing computer vision, NLP, and deep learning projects",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
