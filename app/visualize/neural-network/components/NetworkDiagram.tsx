'use client';

import React, { useEffect, useRef } from 'react';
import { LayerConfig, DataPoint } from '@/lib/visualize/types';

interface NetworkDiagramProps {
  layers: LayerConfig[];
  data: DataPoint[];
  predictions: DataPoint[];
  selectedLayer: number | null;
  onLayerSelect: (index: number | null) => void;
  isTraining?: boolean;
  className?: string;
}

interface NodePosition {
  x: number;
  y: number;
  layerIndex: number;
  nodeIndex: number;
}

export const NetworkDiagram: React.FC<NetworkDiagramProps> = ({
  layers,
  data,
  predictions,
  selectedLayer,
  onLayerSelect,
  isTraining = false,
  className = ''
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate node positions for the network diagram
  const calculateNodePositions = (width: number, height: number): NodePosition[] => {
    const positions: NodePosition[] = [];
    const margin = 60;
    const layerSpacing = (width - 2 * margin) / (layers.length - 1);
    
    layers.forEach((layer, layerIndex) => {
      const units = layer.units || 1;
      const nodeSpacing = (height - 2 * margin) / Math.max(units - 1, 1);
      const startY = height / 2 - ((units - 1) * nodeSpacing) / 2;
      
      for (let nodeIndex = 0; nodeIndex < units; nodeIndex++) {
        positions.push({
          x: margin + layerIndex * layerSpacing,
          y: startY + nodeIndex * nodeSpacing,
          layerIndex,
          nodeIndex
        });
      }
    });
    
    return positions;
  };

  // Generate connection lines between layers
  const generateConnections = (positions: NodePosition[]) => {
    const connections: { from: NodePosition; to: NodePosition; weight?: number }[] = [];
    
    for (let i = 0; i < layers.length - 1; i++) {
      const currentLayerNodes = positions.filter(p => p.layerIndex === i);
      const nextLayerNodes = positions.filter(p => p.layerIndex === i + 1);
      
      currentLayerNodes.forEach(fromNode => {
        nextLayerNodes.forEach(toNode => {
          // Generate random weights for visualization (in real implementation, these would come from the model)
          const weight = Math.random() * 2 - 1; // Random weight between -1 and 1
          connections.push({ from: fromNode, to: toNode, weight });
        });
      });
    }
    
    return connections;
  };

  const getLayerColor = (layerIndex: number, isSelected: boolean) => {
    if (isSelected) return '#3B82F6'; // Blue
    
    switch (layers[layerIndex]?.type) {
      case 'dense':
        return '#10B981'; // Green
      case 'dropout':
        return '#EF4444'; // Red
      default:
        return '#6B7280'; // Gray
    }
  };

  const getActivationColor = (activation: string) => {
    switch (activation) {
      case 'relu': return '#3B82F6';
      case 'sigmoid': return '#10B981';
      case 'tanh': return '#8B5CF6';
      case 'linear': return '#6B7280';
      default: return '#6B7280';
    }
  };

  const drawNetwork = () => {
    if (!svgRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const svg = svgRef.current;
    const rect = container.getBoundingClientRect();
    const width = rect.width;
    const height = 400;

    // Set SVG dimensions
    svg.setAttribute('width', width.toString());
    svg.setAttribute('height', height.toString());

    // Clear previous content
    svg.innerHTML = '';

    const positions = calculateNodePositions(width, height);
    const connections = generateConnections(positions);

    // Draw connections
    connections.forEach(connection => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', connection.from.x.toString());
      line.setAttribute('y1', connection.from.y.toString());
      line.setAttribute('x2', connection.to.x.toString());
      line.setAttribute('y2', connection.to.y.toString());
      
      // Color lines based on weight strength
      const opacity = Math.abs(connection.weight || 0);
      const color = (connection.weight || 0) > 0 ? '#10B981' : '#EF4444';
      line.setAttribute('stroke', color);
      line.setAttribute('stroke-width', '1');
      line.setAttribute('opacity', (opacity * 0.5 + 0.1).toString());
      line.setAttribute('class', 'transition-opacity duration-300');
      
      svg.appendChild(line);
    });

    // Draw nodes
    positions.forEach(position => {
      const isSelected = selectedLayer === position.layerIndex;
      const layer = layers[position.layerIndex];
      
      // Node circle
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', position.x.toString());
      circle.setAttribute('cy', position.y.toString());
      circle.setAttribute('r', isSelected ? '8' : '6');
      circle.setAttribute('fill', getLayerColor(position.layerIndex, isSelected));
      circle.setAttribute('stroke', '#FFFFFF');
      circle.setAttribute('stroke-width', '2');
      circle.setAttribute('class', 'cursor-pointer transition-all duration-200 hover:r-7');
      
      // Add click handler
      circle.addEventListener('click', () => {
        onLayerSelect(selectedLayer === position.layerIndex ? null : position.layerIndex);
      });
      
      // Add animation for training
      if (isTraining) {
        const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
        animate.setAttribute('attributeName', 'r');
        animate.setAttribute('values', '6;8;6');
        animate.setAttribute('dur', '1s');
        animate.setAttribute('repeatCount', 'indefinite');
        circle.appendChild(animate);
      }
      
      svg.appendChild(circle);
    });

    // Draw layer labels
    layers.forEach((layer, index) => {
      const layerPositions = positions.filter(p => p.layerIndex === index);
      if (layerPositions.length === 0) return;
      
      const centerY = layerPositions.reduce((sum, p) => sum + p.y, 0) / layerPositions.length;
      const x = layerPositions[0].x;
      
      // Layer title
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      title.setAttribute('x', x.toString());
      title.setAttribute('y', (centerY - 30).toString());
      title.setAttribute('text-anchor', 'middle');
      title.setAttribute('class', 'text-xs font-medium fill-gray-700');
      title.textContent = `Layer ${index + 1}`;
      svg.appendChild(title);
      
      // Layer type
      const type = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      type.setAttribute('x', x.toString());
      type.setAttribute('y', (centerY - 18).toString());
      type.setAttribute('text-anchor', 'middle');
      type.setAttribute('class', 'text-xs fill-gray-500');
      type.textContent = layer.type;
      svg.appendChild(type);
      
      // Units count
      if (layer.units) {
        const units = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        units.setAttribute('x', x.toString());
        units.setAttribute('y', (centerY + layerPositions.length * 15 + 20).toString());
        units.setAttribute('text-anchor', 'middle');
        units.setAttribute('class', 'text-xs fill-gray-600');
        units.textContent = `${layer.units} units`;
        svg.appendChild(units);
      }
      
      // Activation function
      if (layer.activation) {
        const activation = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        activation.setAttribute('x', x.toString());
        activation.setAttribute('y', (centerY + layerPositions.length * 15 + 32).toString());
        activation.setAttribute('text-anchor', 'middle');
        activation.setAttribute('class', 'text-xs font-medium');
        activation.setAttribute('fill', getActivationColor(layer.activation));
        activation.textContent = layer.activation;
        svg.appendChild(activation);
      }
    });
  };

  // Redraw network when layers or selection changes
  useEffect(() => {
    drawNetwork();
  }, [layers, selectedLayer, isTraining]);

  // Redraw on window resize
  useEffect(() => {
    const handleResize = () => {
      setTimeout(drawNetwork, 100); // Small delay to ensure container has new size
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className={`bg-white border border-gray-200 rounded-lg ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Network Architecture</h3>
          <div className="flex items-center space-x-4 text-xs text-gray-600">
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>Dense Layer</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span>Dropout Layer</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>Selected</span>
            </div>
          </div>
        </div>
        <p className="text-sm text-gray-600 mt-1">
          Click on nodes to select layers. {isTraining && 'Training in progress...'}
        </p>
      </div>

      {/* Network Diagram */}
      <div className="p-4">
        <div ref={containerRef} className="w-full">
          <svg
            ref={svgRef}
            className="w-full border border-gray-100 rounded bg-gray-50"
            style={{ minHeight: '400px' }}
          />
        </div>

        {/* Architecture Info */}
        {selectedLayer !== null && layers[selectedLayer] && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="text-sm font-medium text-blue-800 mb-2">
              Layer {selectedLayer + 1} Details
            </h4>
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <span className="text-blue-600 font-medium">Type:</span>
                <span className="ml-2 capitalize">{layers[selectedLayer].type}</span>
              </div>
              {layers[selectedLayer].units && (
                <div>
                  <span className="text-blue-600 font-medium">Units:</span>
                  <span className="ml-2">{layers[selectedLayer].units}</span>
                </div>
              )}
              {layers[selectedLayer].activation && (
                <div>
                  <span className="text-blue-600 font-medium">Activation:</span>
                  <span className="ml-2 capitalize">{layers[selectedLayer].activation}</span>
                </div>
              )}
              {layers[selectedLayer].rate && (
                <div>
                  <span className="text-blue-600 font-medium">Dropout Rate:</span>
                  <span className="ml-2">{layers[selectedLayer].rate}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Data Flow Info */}
        <div className="mt-4 grid grid-cols-3 gap-4 text-center text-xs">
          <div className="p-2 bg-gray-50 rounded">
            <div className="text-gray-600">Input Shape</div>
            <div className="font-medium">2D Features</div>
          </div>
          <div className="p-2 bg-gray-50 rounded">
            <div className="text-gray-600">Data Points</div>
            <div className="font-medium">{data.length}</div>
          </div>
          <div className="p-2 bg-gray-50 rounded">
            <div className="text-gray-600">Output</div>
            <div className="font-medium">
              {layers[layers.length - 1]?.units === 1 ? 'Binary' : 'Multi-class'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetworkDiagram;
