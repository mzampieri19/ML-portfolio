'use client';

import { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, Loader2, CheckCircle, XCircle } from 'lucide-react';

interface PredictionResult {
  class: string;
  confidence: number;
}

export default function InteractiveDemo() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setPredictions([]);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setPredictions([]);
    }
  };

  const simulateClassification = async () => {
    setIsLoading(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock predictions for demo purposes
    const mockPredictions = [
      { class: 'Cat', confidence: 0.89 },
      { class: 'Dog', confidence: 0.08 },
      { class: 'Bird', confidence: 0.02 },
      { class: 'Horse', confidence: 0.01 },
      { class: 'Deer', confidence: 0.00 }
    ];
    
    setPredictions(mockPredictions);
    setIsLoading(false);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.7) return 'text-green-600 dark:text-green-400';
    if (confidence > 0.3) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence > 0.7) return <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />;
    if (confidence > 0.3) return <CheckCircle className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />;
    return <XCircle className="w-4 h-4 text-red-600 dark:text-red-400" />;
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-slate-700 mb-8">
      <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
        Try the Model
      </h3>
      
      <div className="grid md:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div>
          <div
            className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-8 text-center hover:border-blue-400 dark:hover:border-blue-500 transition-colors cursor-pointer"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept="image/*"
              className="hidden"
            />
            
            {preview ? (
              <div className="space-y-4">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-w-full h-48 object-contain mx-auto rounded-lg"
                />
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  {selectedFile?.name}
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex justify-center">
                  <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                    <ImageIcon className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                  </div>
                </div>
                <div>
                  <p className="text-lg font-medium text-slate-900 dark:text-white mb-2">
                    Upload an image
                  </p>
                  <p className="text-slate-600 dark:text-slate-400">
                    Drag & drop or click to select
                  </p>
                  <p className="text-sm text-slate-500 dark:text-slate-500 mt-2">
                    Supports: JPG, PNG, GIF (max 10MB)
                  </p>
                </div>
              </div>
            )}
          </div>
          
          {selectedFile && (
            <button
              onClick={simulateClassification}
              disabled={isLoading}
              className="w-full mt-4 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Classifying...</span>
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  <span>Classify Image</span>
                </>
              )}
            </button>
          )}
        </div>
        
        {/* Results Section */}
        <div>
          <h4 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Predictions
          </h4>
          
          {predictions.length > 0 ? (
            <div className="space-y-3">
              {predictions.map((prediction, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    {getConfidenceIcon(prediction.confidence)}
                    <span className="font-medium text-slate-900 dark:text-white">
                      {prediction.class}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className={`font-bold ${getConfidenceColor(prediction.confidence)}`}>
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                    <div className="w-24 h-2 bg-slate-200 dark:bg-slate-600 rounded-full mt-1">
                      <div
                        className="h-full bg-blue-500 rounded-full transition-all duration-500"
                        style={{ width: `${prediction.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500 dark:text-slate-400">
              <ImageIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Upload an image to see predictions</p>
            </div>
          )}
        </div>
      </div>
      
      {predictions.length > 0 && (
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <p className="text-sm text-blue-800 dark:text-blue-300">
            <strong>Note:</strong> This is a demo with simulated results. The actual model would process your image through the CNN architecture described above.
          </p>
        </div>
      )}
    </div>
  );
}
