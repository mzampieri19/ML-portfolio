'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, RotateCcw, Info } from 'lucide-react';
import { ControlConfig, ControlPanelConfig } from '../../../lib/visualize/types';

interface ControlPanelProps {
  config: ControlPanelConfig;
  values: Record<string, any>;
  onChange: (controlId: string, value: any) => void;
  onReset?: () => void;
  className?: string;
  disabled?: boolean;
}

export default function ControlPanel({
  config,
  values,
  onChange,
  onReset,
  className = '',
  disabled = false
}: ControlPanelProps) {
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set());
  const [showTooltip, setShowTooltip] = useState<string | null>(null);

  const toggleCategory = (category: string) => {
    const newCollapsed = new Set(collapsedCategories);
    if (newCollapsed.has(category)) {
      newCollapsed.delete(category);
    } else {
      newCollapsed.add(category);
    }
    setCollapsedCategories(newCollapsed);
  };

  const renderControl = (control: ControlConfig) => {
    const value = values[control.id] ?? control.value;

    switch (control.type) {
      case 'slider':
        return (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label 
                htmlFor={control.id}
                className="text-sm font-medium text-slate-700 dark:text-slate-300"
              >
                {control.label}
              </label>
              <span className="text-sm text-slate-500 dark:text-slate-400 font-mono">
                {typeof value === 'number' ? value.toFixed(3) : value}
              </span>
            </div>
            <input
              id={control.id}
              type="range"
              min={control.min}
              max={control.max}
              step={control.step || 0.01}
              value={value}
              onChange={(e) => onChange(control.id, parseFloat(e.target.value))}
              disabled={disabled}
              className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer
                slider:h-2 slider:rounded-lg slider:border-0
                slider:bg-gradient-to-r slider:from-purple-500 slider:to-blue-500
                slider:shadow-lg slider:shadow-purple-500/25
                slider-thumb:appearance-none slider-thumb:h-4 slider-thumb:w-4 
                slider-thumb:rounded-full slider-thumb:bg-white slider-thumb:shadow-lg
                slider-thumb:border-2 slider-thumb:border-purple-500
                hover:slider-thumb:scale-110 transition-transform
                disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <div className="flex justify-between text-xs text-slate-400 dark:text-slate-500">
              <span>{control.min}</span>
              <span>{control.max}</span>
            </div>
          </div>
        );

      case 'select':
        return (
          <div className="space-y-2">
            <label 
              htmlFor={control.id}
              className="text-sm font-medium text-slate-700 dark:text-slate-300"
            >
              {control.label}
            </label>
            <select
              id={control.id}
              value={value}
              onChange={(e) => onChange(control.id, e.target.value)}
              disabled={disabled}
              className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 
                rounded-lg text-slate-900 dark:text-white text-sm
                focus:ring-2 focus:ring-purple-500 focus:border-transparent
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {control.options?.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        );

      case 'checkbox':
        return (
          <div className="flex items-center space-x-3">
            <input
              id={control.id}
              type="checkbox"
              checked={value}
              onChange={(e) => onChange(control.id, e.target.checked)}
              disabled={disabled}
              className="w-4 h-4 text-purple-600 bg-slate-100 dark:bg-slate-700 border-slate-300 dark:border-slate-600 
                rounded focus:ring-purple-500 dark:focus:ring-purple-600 focus:ring-2
                disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <label 
              htmlFor={control.id}
              className="text-sm font-medium text-slate-700 dark:text-slate-300"
            >
              {control.label}
            </label>
          </div>
        );

      case 'input':
        return (
          <div className="space-y-2">
            <label 
              htmlFor={control.id}
              className="text-sm font-medium text-slate-700 dark:text-slate-300"
            >
              {control.label}
            </label>
            <input
              id={control.id}
              type="number"
              value={value}
              onChange={(e) => onChange(control.id, parseFloat(e.target.value) || 0)}
              disabled={disabled}
              min={control.min}
              max={control.max}
              step={control.step || 0.01}
              className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 
                rounded-lg text-slate-900 dark:text-white text-sm
                focus:ring-2 focus:ring-purple-500 focus:border-transparent
                disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>
        );

      default:
        return null;
    }
  };

  const groupedControls = config.categories 
    ? config.categories.reduce((acc, category) => {
        acc[category] = config.controls.filter(control => control.category === category);
        return acc;
      }, {} as Record<string, ControlConfig[]>)
    : { 'Parameters': config.controls };

  return (
    <div className={`bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
            {config.title}
          </h3>
          {onReset && (
            <button
              onClick={onReset}
              disabled={disabled}
              className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-600 dark:text-slate-400 
                hover:text-slate-900 dark:hover:text-white transition-colors
                disabled:opacity-50 disabled:cursor-not-allowed"
              title="Reset to defaults"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="p-4 space-y-6">
        {Object.entries(groupedControls).map(([categoryName, controls]) => (
          <div key={categoryName}>
            {/* Category Header */}
            {config.categories && config.categories.length > 1 && (
              <div className="mb-4">
                <button
                  onClick={() => toggleCategory(categoryName)}
                  className="flex items-center justify-between w-full text-left"
                >
                  <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 uppercase tracking-wide">
                    {categoryName}
                  </h4>
                  {collapsedCategories.has(categoryName) ? (
                    <ChevronDown className="w-4 h-4 text-slate-500" />
                  ) : (
                    <ChevronUp className="w-4 h-4 text-slate-500" />
                  )}
                </button>
              </div>
            )}

            {/* Category Controls */}
            {!collapsedCategories.has(categoryName) && (
              <div className="space-y-4">
                {controls.map((control) => (
                  <div key={control.id} className="relative">
                    <div className="flex items-start gap-2">
                      <div className="flex-1">
                        {renderControl(control)}
                      </div>
                      {control.description && (
                        <button
                          onMouseEnter={() => setShowTooltip(control.id)}
                          onMouseLeave={() => setShowTooltip(null)}
                          className="mt-1 p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                        >
                          <Info className="w-4 h-4" />
                        </button>
                      )}
                    </div>

                    {/* Tooltip */}
                    {showTooltip === control.id && control.description && (
                      <div className="absolute z-10 right-0 top-0 mt-8 w-64 p-3 bg-slate-900 dark:bg-slate-700 
                        text-white text-sm rounded-lg shadow-xl border border-slate-600">
                        <div className="absolute -top-1 right-6 w-2 h-2 bg-slate-900 dark:bg-slate-700 
                          border-l border-t border-slate-600 transform rotate-45"></div>
                        {control.description}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
