import React, { useState } from 'react';

export interface ClassProbability {
  id: string;
  name: string;
  score: number;
}

interface ProbabilityChartProps {
  classes: ClassProbability[];
  predictedClassId: string;
}

/**
 * ProbabilityChart — Sorted horizontal bar chart for prediction distribution.
 *
 * UX Laws applied:
 * - Hick's Law: sorted highest-to-lowest so the clinician instantly sees
 *   the dominant prediction without scanning all 7 classes.
 * - Miller's Law: 7 classes is within the 7±2 cognitive chunk limit.
 *   Sub-1% scores are visually de-emphasized to reduce noise.
 * - Doherty Threshold: hover micro-interaction provides instant feedback
 *   (< 100ms visual response), keeping the interface feeling alive.
 * - Visual Hierarchy: predicted class bar is accented; others are muted.
 */
const ProbabilityChart: React.FC<ProbabilityChartProps> = ({ classes, predictedClassId }) => {
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Sort descending by score (Hick's Law — most likely first)
  const sorted = [...classes].sort((a, b) => b.score - a.score);

  return (
    <section className="bg-white rounded-xl border border-slate-300 shadow-sm p-6 md:p-8">
      <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-6">
        Prediction Confidence Distribution
      </h3>

      <div className="space-y-4">
        {sorted.map((cls) => {
          const isPredicted = cls.id === predictedClassId;
          const isHovered = hoveredId === cls.id;

          return (
            <div
              key={cls.id}
              className={`group rounded-lg px-3 py-2.5 -mx-3 transition-colors duration-150 cursor-default ${
                isHovered ? 'bg-slate-50' : ''
              }`}
              onMouseEnter={() => setHoveredId(cls.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              {/* Label row */}
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className={`text-xs font-bold uppercase w-11 shrink-0 ${
                      isPredicted ? 'text-teal-700' : 'text-slate-500'
                    }`}
                  >
                    {cls.id}
                  </span>
                  <span
                    className={`text-xs truncate ${
                      isPredicted ? 'text-slate-700 font-medium' : 'text-slate-400'
                    }`}
                  >
                    {cls.name}
                  </span>
                </div>
                <span
                  className={`text-sm tabular-nums font-semibold ml-3 shrink-0 ${
                    isPredicted ? 'text-teal-700' : 'text-slate-500'
                  }`}
                >
                  {cls.score.toFixed(1)}%
                </span>
              </div>

              {/* Bar */}
              <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out ${
                    isPredicted ? 'bg-teal-600' : isHovered ? 'bg-slate-400' : 'bg-slate-300'
                  }`}
                  style={{ width: `${Math.max(cls.score, 0.5)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};

export default ProbabilityChart;
