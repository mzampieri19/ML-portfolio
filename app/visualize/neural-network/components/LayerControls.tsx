'use client';

import React, { useState } from 'react';
import { LayerConfig } from '@/lib/visualize/types';

interface LayerControlsProps {
  layers: LayerConfig[];
  selectedLayer: number | null;
  onLayerSelect: (index: number | null) => void;
  onLayerAdd: (layerConfig: LayerConfig) => void;
  onLayerRemove: (index: number) => void;
  onLayerUpdate: (index: number, layerConfig: LayerConfig) => void;
  isTraining?: boolean;
  className?: string;
}

interface LayerCardProps {
  layer: LayerConfig;
  index: number;
  isSelected: boolean;
  onSelect: () => void;
  onUpdate: (layerConfig: LayerConfig) => void;
  onRemove: () => void;
  isTraining?: boolean;
}

const LayerCard: React.FC<LayerCardProps> = ({
  layer,
  index,
  isSelected,
  onSelect,
  onUpdate,
  onRemove,
  isTraining = false
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editValues, setEditValues] = useState(layer);

  const handleSave = () => {
    onUpdate(editValues);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditValues(layer);
    setIsEditing(false);
  };

  const getLayerIcon = () => {
    switch (layer.type) {
      case 'dense':
        return (
          <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center">
            <span className="text-white text-xs font-bold">FC</span>
          </div>
        );
      case 'conv2d':
        return (
          <div className="w-8 h-8 bg-green-600 rounded flex items-center justify-center">
            <span className="text-white text-xs font-bold">CV</span>
          </div>
        );
      case 'maxPooling2d':
        return (
          <div className="w-8 h-8 bg-purple-600 rounded flex items-center justify-center">
            <span className="text-white text-xs font-bold">MP</span>
          </div>
        );
      case 'dropout':
        return (
          <div className="w-8 h-8 bg-red-600 rounded flex items-center justify-center">
            <span className="text-white text-xs font-bold">DR</span>
          </div>
        );
      default:
        return (
          <div className="w-8 h-8 bg-gray-600 rounded flex items-center justify-center">
            <span className="text-white text-xs font-bold">??</span>
          </div>
        );
    }
  };

  const getActivationColor = (activation: string) => {
    switch (activation) {
      case 'relu': return 'text-blue-600';
      case 'sigmoid': return 'text-green-600';
      case 'tanh': return 'text-purple-600';
      case 'linear': return 'text-gray-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div 
      className={`border rounded-lg p-4 cursor-pointer transition-all ${
        isSelected 
          ? 'border-blue-500 bg-blue-50 shadow-md' 
          : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          {getLayerIcon()}
          <div>
            <div className="font-medium text-sm text-gray-800">
              Layer {index + 1}
            </div>
            <div className="text-xs text-gray-500 capitalize">
              {layer.type.replace(/([A-Z])/g, ' $1').trim()}
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-1">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsEditing(true);
            }}
            disabled={isTraining}
            className="text-blue-600 hover:text-blue-700 disabled:opacity-50 p-1"
            title="Edit layer"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                    d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </button>
          
          {index > 0 && ( // Don't allow removing the first layer
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemove();
              }}
              disabled={isTraining}
              className="text-red-600 hover:text-red-700 disabled:opacity-50 p-1"
              title="Remove layer"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {!isEditing ? (
        <div className="space-y-1 text-sm">
          {layer.type === 'dense' && (
            <div className="flex justify-between">
              <span className="text-gray-600">Units:</span>
              <span className="font-medium">{layer.units}</span>
            </div>
          )}
          
          {layer.activation && (
            <div className="flex justify-between">
              <span className="text-gray-600">Activation:</span>
              <span className={`font-medium capitalize ${getActivationColor(layer.activation)}`}>
                {layer.activation}
              </span>
            </div>
          )}
          
          {layer.type === 'dropout' && layer.rate && (
            <div className="flex justify-between">
              <span className="text-gray-600">Rate:</span>
              <span className="font-medium">{layer.rate}</span>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-3 text-sm" onClick={(e) => e.stopPropagation()}>
          {layer.type === 'dense' && (
            <div>
              <label className="block text-gray-600 mb-1">Units:</label>
              <input
                type="number"
                value={editValues.units || 1}
                onChange={(e) => setEditValues({...editValues, units: parseInt(e.target.value)})}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                min="1"
                max="1000"
              />
            </div>
          )}
          
          <div>
            <label className="block text-gray-600 mb-1">Activation:</label>
            <select
              value={editValues.activation || 'relu'}
              onChange={(e) => setEditValues({...editValues, activation: e.target.value})}
              className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
            >
              <option value="relu">ReLU</option>
              <option value="tanh">Tanh</option>
              <option value="sigmoid">Sigmoid</option>
              <option value="linear">Linear</option>
            </select>
          </div>
          
          {layer.type === 'dropout' && (
            <div>
              <label className="block text-gray-600 mb-1">Dropout Rate:</label>
              <input
                type="number"
                value={editValues.rate || 0.5}
                onChange={(e) => setEditValues({...editValues, rate: parseFloat(e.target.value)})}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                min="0"
                max="1"
                step="0.1"
              />
            </div>
          )}
          
          <div className="flex space-x-2 pt-2">
            <button
              onClick={handleSave}
              className="flex-1 bg-blue-600 text-white py-1 px-2 rounded text-xs hover:bg-blue-700"
            >
              Save
            </button>
            <button
              onClick={handleCancel}
              className="flex-1 bg-gray-600 text-white py-1 px-2 rounded text-xs hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export const LayerControls: React.FC<LayerControlsProps> = ({
  layers,
  selectedLayer,
  onLayerSelect,
  onLayerAdd,
  onLayerRemove,
  onLayerUpdate,
  isTraining = false,
  className = ''
}) => {
  const [showAddMenu, setShowAddMenu] = useState(false);

  const handleAddLayer = (layerType: string) => {
    const newLayer: LayerConfig = {
      id: `layer-${Date.now()}`,
      type: layerType as any,
      position: { x: 0, y: 0 },
      units: layerType === 'dense' ? 10 : undefined,
      activation: layerType === 'dense' ? 'relu' : undefined,
      rate: layerType === 'dropout' ? 0.5 : undefined
    };

    onLayerAdd(newLayer);
    setShowAddMenu(false);
  };

  return (
    <div className={`bg-white border border-gray-200 rounded-lg ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Network Architecture</h3>
          <div className="relative">
            <button
              onClick={() => setShowAddMenu(!showAddMenu)}
              disabled={isTraining}
              className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Add Layer
            </button>
            
            {showAddMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-white border border-gray-200 rounded-lg shadow-lg z-10">
                <div className="py-2">
                  <button
                    onClick={() => handleAddLayer('dense')}
                    className="w-full text-left px-4 py-2 hover:bg-gray-50 text-sm"
                  >
                    <div className="font-medium">Dense Layer</div>
                    <div className="text-gray-500 text-xs">Fully connected layer</div>
                  </button>
                  <button
                    onClick={() => handleAddLayer('dropout')}
                    className="w-full text-left px-4 py-2 hover:bg-gray-50 text-sm"
                  >
                    <div className="font-medium">Dropout Layer</div>
                    <div className="text-gray-500 text-xs">Regularization layer</div>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
        <p className="text-sm text-gray-600 mt-1">
          Configure your neural network architecture by adding, removing, and editing layers.
        </p>
      </div>

      {/* Layers */}
      <div className="p-4">
        <div className="space-y-3">
          {layers.map((layer, index) => (
            <LayerCard
              key={layer.id || index}
              layer={layer}
              index={index}
              isSelected={selectedLayer === index}
              onSelect={() => onLayerSelect(selectedLayer === index ? null : index)}
              onUpdate={(layerConfig) => onLayerUpdate(index, layerConfig)}
              onRemove={() => onLayerRemove(index)}
              isTraining={isTraining}
            />
          ))}
        </div>

        {/* Architecture Summary */}
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Architecture Summary</h4>
          <div className="text-xs text-gray-600 space-y-1">
            <div>Total Layers: {layers.length}</div>
            <div>
              Parameters: ~{layers.reduce((total, layer) => {
                if (layer.type === 'dense' && layer.units) {
                  const prevUnits = layers[layers.indexOf(layer) - 1]?.units || 2; // Assume 2 input features
                  return total + (prevUnits * layer.units) + layer.units; // weights + biases
                }
                return total;
              }, 0).toLocaleString()}
            </div>
            <div className="flex flex-wrap gap-1 mt-2">
              {layers.map((layer, index) => (
                <span key={index} className="inline-block px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">
                  {layer.type === 'dense' ? `FC(${layer.units})` : layer.type}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Click outside to close add menu */}
      {showAddMenu && (
        <div
          className="fixed inset-0 z-5"
          onClick={() => setShowAddMenu(false)}
        />
      )}
    </div>
  );
};

export default LayerControls;
