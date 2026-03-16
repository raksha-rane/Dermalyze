import React, { useState, useEffect } from 'react';
import ResultCard from './ui/ResultCard';
import ProbabilityChart from './ui/ProbabilityChart';
import MedicalInfoCard from './ui/MedicalInfoCard';
import { classInfoMap } from '../lib/classInfo';
import type { AnalysisHistoryItem } from '../lib/types';

interface HistoryDetailScreenProps {
  item: AnalysisHistoryItem | null;
  onBack: () => void;
}

/**
 * HistoryDetailScreen — Historical analysis record viewer.
 *
 * Shares identical visual structure with ResultsScreen (Jakob's Law —
 * consistency within the product). Reuses the same sub-components:
 * ResultCard, ProbabilityChart, ImageCard, MedicalInfoCard.
 *
 * Additional UX considerations:
 * - Back navigation is prominent in the header (Fitts's Law).
 * - Case ID and timestamp are surfaced in the header for clinical
 *   record-keeping (Jakob's Law — matches EHR conventions).
 * - Skeleton loading state for perceived speed (Doherty Threshold).
 */
const HistoryDetailScreen: React.FC<HistoryDetailScreenProps> = ({ item, onBack }) => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 300);
    return () => clearTimeout(timer);
  }, []);

  if (!item) return null;

  // Only use real scores from the DB — never fabricate a distribution.
  const classes = item.allScores ?? null;

  const info = classInfoMap[item.classId];
  const caseId = `DRM-${item.id.slice(0, 8).toUpperCase()}`;

  /* Skeleton loading state (Doherty Threshold) */
  if (loading) {
    return (
      <div className="flex-1 flex flex-col bg-slate-50 text-slate-900">
        <main className="max-w-6xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
          <div className="h-40 bg-white border border-slate-200 rounded-xl animate-pulse" />
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            <div className="lg:col-span-3 h-72 bg-white border border-slate-200 rounded-xl animate-pulse" />
            <div className="lg:col-span-2 space-y-6">
              <div className="h-52 bg-white border border-slate-200 rounded-xl animate-pulse" />
              <div className="h-48 bg-white border border-slate-200 rounded-xl animate-pulse" />
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900">
      {/* MAIN CONTENT */}
      <main className="max-w-6xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-6">
        {/* Case metadata bar */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs">
          <h1 className="text-sm font-semibold text-slate-500 uppercase tracking-widest">Analysis Record</h1>
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-400 uppercase tracking-widest">Case ID:</span>
            <span className="font-semibold text-slate-700 tabular-nums">{caseId}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-400 uppercase tracking-widest">Date:</span>
            <span className="font-semibold text-slate-700">{item.date} &middot; {item.time}</span>
          </div>
        </div>

        {/* PRIMARY RESULT CARD — includes analyzed image */}
        <ResultCard
          classId={item.classId}
          className={item.className}
          confidence={item.confidence}
          info={info}
          imageUrl={item.imageUrl}
        />

        {/* PROBABILITY CHART — only render when real scores are available */}
        {classes ? (
          <ProbabilityChart classes={classes} predictedClassId={item.classId} />
        ) : (
          <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Probability Breakdown</p>
            <p className="text-sm text-slate-500">Full probability breakdown is not available for this record.</p>
          </section>
        )}

        {/* MEDICAL INFO — Full-width row below the bento pair */}
        {info ? (
          <MedicalInfoCard info={info} />
        ) : (
          <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
            <p className="text-sm text-slate-500">
              No condition information available for this classification.
            </p>
          </section>
        )}

        {/* Back to History — full width */}
        <button
          onClick={onBack}
          className="w-full py-2.5 px-4 rounded-full font-medium text-sm border border-slate-200 text-slate-600 transition-all duration-200 hover:bg-teal-50 hover:border-teal-300 hover:text-teal-700 active:bg-teal-100 active:border-teal-400 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-400"
        >
          <span className="flex items-center justify-center gap-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to History
          </span>
        </button>
      </main>

      {/* FOOTER — Clinical Disclaimer */}
      <footer className="mt-auto py-6 border-t border-slate-100 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <p className="text-[11px] text-slate-400 leading-relaxed text-center">
            Historical records are provided for longitudinal review. Classification scores are
            probabilistic outputs and serve as diagnostic aids only. Always correlate with clinical
            findings and professional judgment.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default HistoryDetailScreen;
