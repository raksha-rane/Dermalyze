import React, { useState } from 'react';
import {
  ClassInfo,
  getRiskSeverity,
  getRiskBadgeStyles,
  getRiskLabel,
  getConfidenceColor,
} from '../../lib/classInfo';

interface ResultCardProps {
  classId: string;
  className: string;
  confidence: number;
  info: ClassInfo;
  imageUrl?: string | null;
  gradcamImage?: string | null;
}

/**
 * ResultCard — Primary focal point of the analysis screen.
 *
 * UX Laws applied:
 * - Visual Hierarchy: largest type size for the predicted class, establishing
 *   a clear F-pattern reading flow (diagnosis → confidence → risk).
 * - Aesthetic-Usability Effect: clean card with generous whitespace feels
 *   trustworthy and professional — critical in a clinical context.
 * - Fitts's Law: risk badge is large enough to be instantly perceived.
 * - Law of Proximity: image placed alongside diagnosis for immediate
 *   visual correlation between the lesion and the classification.
 */
const ResultCard: React.FC<ResultCardProps> = ({
  classId,
  className,
  confidence,
  info,
  imageUrl,
  gradcamImage,
}) => {
  const severity = getRiskSeverity(info.riskLevel);
  const badge = getRiskBadgeStyles(severity);
  const confidenceColor = getConfidenceColor(confidence);
  const [showHeatmap, setShowHeatmap] = useState(false);

  // Determine which image to display
  const displayImage = showHeatmap && gradcamImage ? gradcamImage : imageUrl;

  return (
    <section className="bg-white rounded-xl border border-slate-400 shadow-sm overflow-hidden">
      {/* Top accent strip keyed to risk severity */}
      <div
        className={`h-1 w-full ${
          severity === 'critical' || severity === 'high'
            ? 'bg-red-500'
            : severity === 'moderate'
              ? 'bg-amber-400'
              : 'bg-emerald-500'
        }`}
      />

      <div className="px-4 sm:px-8 pt-5 sm:pt-7 pb-6 sm:pb-8">
        <div className="flex flex-col md:flex-row md:items-start gap-6">
          {/* Left — Diagnosis info */}
          <div className="flex-1 min-w-0">
            {/* Overline label — small, muted to set context */}
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-4">
              Predicted Classification
            </p>

            {/* Primary diagnosis — largest element on screen */}
            <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight mb-1">
              {className}
            </h2>
            <p className="text-sm text-slate-400 font-medium mb-6">
              Class: <span className="font-semibold text-slate-500 uppercase">{classId}</span>
            </p>

            {/* Confidence + Risk — grouped via Law of Proximity */}
            <div className="flex flex-wrap items-center gap-4">
              {/* Confidence score */}
              <div className="flex items-baseline gap-2">
                <span className={`text-4xl font-extrabold tabular-nums ${confidenceColor}`}>
                  {confidence.toFixed(1)}
                  <span className="text-2xl">%</span>
                </span>
                <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">
                  Confidence
                </span>
              </div>

              {/* Vertical divider */}
              <div className="hidden sm:block w-px h-8 bg-slate-200" />

              {/* Risk badge */}
              <span
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider border ${badge.bg} ${badge.text} ${badge.border}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${badge.dot}`} />
                {getRiskLabel(severity)}
              </span>
            </div>
          </div>

          {/* Right — Analyzed image with Grad-CAM toggle */}
          <div className="w-full md:w-48 lg:w-56 flex-shrink-0">
            <div className="flex items-center justify-between md:justify-end gap-2 mb-3">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">
                {showHeatmap && gradcamImage ? 'Explainability' : 'Analyzed Image'}
              </p>
              {gradcamImage && (
                <button
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wide transition-colors ${
                    showHeatmap
                      ? 'bg-teal-200 text-teal-700 hover:bg-teal-200'
                      : 'bg-slate-200 text-slate-600 hover:bg-slate-300 '
                  }`}
                  title={showHeatmap ? 'Show original image' : 'Show Grad-CAM heatmap'}
                >
                  <svg
                    className="w-3.5 h-3.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    {showHeatmap ? (
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                      />
                    ) : (
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                      />
                    )}
                  </svg>
                  {showHeatmap ? 'Original' : 'Heatmap'}
                </button>
              )}
            </div>
            <div className="aspect-square w-full bg-slate-50 rounded-lg overflow-hidden border border-slate-100 flex items-center justify-center relative">
              {displayImage ? (
                <>
                  <img
                    src={displayImage}
                    alt={
                      showHeatmap && gradcamImage
                        ? 'Grad-CAM heatmap showing important regions'
                        : 'Analyzed lesion'
                    }
                    className="w-full h-full object-cover"
                    draggable={false}
                  />
                  {showHeatmap && gradcamImage && (
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-2">
                      <p className="text-[9px] text-white/90 text-center font-medium">
                        Highlighted regions influenced the prediction
                      </p>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-slate-300 flex flex-col items-center gap-2">
                  <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <span className="text-xs">No image</span>
                </div>
              )}
            </div>
            {/* Heatmap legend when showing Grad-CAM */}
            {showHeatmap && gradcamImage && (
              <div className="mt-2 flex items-center justify-center gap-1">
                <span className="text-[9px] text-slate-400">Low</span>
                <div className="flex h-2 rounded-sm overflow-hidden">
                  <div className="w-4 bg-blue-500" />
                  <div className="w-4 bg-cyan-400" />
                  <div className="w-4 bg-green-400" />
                  <div className="w-4 bg-yellow-400" />
                  <div className="w-4 bg-red-500" />
                </div>
                <span className="text-[9px] text-slate-400">High</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ResultCard;
