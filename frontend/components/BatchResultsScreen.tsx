import React, { useEffect, useRef, useState } from 'react';
import type { BatchItemResult, ClassResult } from '../lib/types';
import { supabase } from '../lib/supabase';
import { optimizeImage } from '../lib/imageOptimization';
import { encryptImage, getEncryptedExtension } from '../lib/imageEncryption';
import { useDataCache } from '../lib/dataCache';

interface BatchResultsScreenProps {
  results: BatchItemResult[];
  onViewDetail: (item: BatchItemResult) => void;
  onAnalyzeAnother: () => void;
  onNavigateToHistory: () => void;
}

const RISK: Record<string, { label: string; cls: string }> = {
  mel: { label: 'Critical', cls: 'text-red-600 bg-red-50 border-red-200' },
  bcc: { label: 'High', cls: 'text-orange-600 bg-orange-50 border-orange-200' },
  akiec: { label: 'Moderate', cls: 'text-amber-600 bg-amber-50 border-amber-200' },
  bkl: { label: 'Low', cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  df: { label: 'Low', cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  nv: { label: 'Low', cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  vasc: { label: 'Low', cls: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
};

const TRUST_BADGE: Record<
  string,
  { label: string; cls: string }
> = {
  classify: { label: 'Classified', cls: 'text-teal-700 bg-teal-50 border-teal-200' },
  review_required: { label: 'Review', cls: 'text-amber-700 bg-amber-50 border-amber-200' },
  reject: { label: 'Rejected', cls: 'text-red-700 bg-red-50 border-red-200' },
};

const topClass = (classes?: ClassResult[]): ClassResult | null =>
  classes?.length ? classes.reduce((a, b) => (a.score > b.score ? a : b)) : null;

const BatchResultsScreen: React.FC<BatchResultsScreenProps> = ({
  results,
  onViewDetail,
  onAnalyzeAnother,
  onNavigateToHistory,
}) => {
  const { invalidateAll } = useDataCache();
  const savedRef = useRef(false);
  const [saveSummary, setSaveSummary] = useState<'saving' | 'done' | 'partial' | 'failed'>('saving');

  const classified = results.filter((r) => r.status === 'success');
  const errors = results.filter((r) => r.status === 'error');
  const rejected = results.filter((r) => r.status === 'rejected');
  const needsReview = results.filter(
    (r) => r.status === 'success' && r.trustResult?.recommendation === 'review_required'
  );

  // Save all results to Supabase (non-blocking, best-effort)
  useEffect(() => {
    if (savedRef.current) return;
    savedRef.current = true;

    const saveAll = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (!user) { setSaveSummary('failed'); return; }

      let failCount = 0;

      for (const item of results) {
        if (item.status !== 'success' && item.status !== 'rejected') continue;
        const pred = topClass(item.classes);
        if (!pred) continue;

        try {
          let image_url: string | null = null;
          let gradcam_image_url: string | null = null;

          // Upload image
          const optimizedBlob = await optimizeImage(item.image, {
            maxDimension: 1024,
            quality: 0.75,
            format: 'image/webp',
          });
          const encryptedBlob = await encryptImage(optimizedBlob, user.id);
          const ext = getEncryptedExtension();
          const path = `${user.id}/${item.caseId}.${ext}`;
          const { error: uploadErr } = await supabase.storage
            .from('analysis-images')
            .upload(path, encryptedBlob, { contentType: 'application/octet-stream' });
          if (!uploadErr) image_url = path;

          if (item.gradcamImage) {
            try {
              const optimizedGradcamBlob = await optimizeImage(item.gradcamImage, {
                maxDimension: 1024,
                quality: 0.75,
                format: 'image/webp',
              });
              const encryptedGradcamBlob = await encryptImage(optimizedGradcamBlob, user.id);
              const gradcamPath = `${user.id}/${item.caseId}_gradcam.${ext}`;
              const { error: uploadGradcamErr } = await supabase.storage
                .from('analysis-images')
                .upload(gradcamPath, encryptedGradcamBlob, { contentType: 'application/octet-stream' });
              if (!uploadGradcamErr) gradcam_image_url = gradcamPath;
            } catch (e) {
              console.error('Failed to upload Grad-CAM image:', e);
            }
          }

          const metadataPayload = {
            age_approx: item.metadata.ageApprox ?? null,
            sex: item.metadata.sex ?? null,
            anatom_site: item.metadata.anatomSite ?? null,
            trust: item.trustResult,
          };

          const { error: insertErr } = await supabase.from('analyses').insert({
            id: item.caseId,
            user_id: user.id,
            image_url,
            gradcam_image_url,
            predicted_class_id: pred.id,
            predicted_class_name: pred.name,
            confidence: pred.score,
            all_scores: item.classes,
            metadata: metadataPayload,
            trust_recommendation: item.trustResult?.recommendation ?? null,
            trust_uncertainty_score: item.trustResult?.uncertainty.score ?? null,
            trust_quality_flags: item.trustResult?.quality_flags ?? [],
          });
          if (insertErr) failCount++;
        } catch {
          failCount++;
        }
      }

      invalidateAll();

      const saveable = results.filter((r) => r.status === 'success' || r.status === 'rejected').length;
      if (failCount === 0) setSaveSummary('done');
      else if (failCount < saveable) setSaveSummary('partial');
      else setSaveSummary('failed');
    };

    saveAll();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex-1 flex flex-col bg-slate-50">
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 py-8 flex flex-col gap-6">

        {/* Page header */}
        <div>
          <h1 className="text-sm font-semibold text-slate-500 uppercase tracking-widest">
            Batch Analysis Results
          </h1>
          <p className="text-2xl font-bold text-slate-900 tracking-tight mt-1">
            {results.length} image{results.length !== 1 ? 's' : ''} analyzed
          </p>
        </div>

        {/* Save status banner */}
        {saveSummary === 'saving' && (
          <div className="flex items-center gap-2 text-xs text-slate-500 bg-white border border-slate-200 rounded-xl px-4 py-3">
            <svg className="animate-spin w-4 h-4 text-teal-500" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Saving results to your history…
          </div>
        )}
        {saveSummary === 'done' && (
          <div className="text-xs text-teal-700 bg-teal-50 border border-teal-200 rounded-xl px-4 py-3 font-medium">
            ✓ All results saved to history.
          </div>
        )}
        {saveSummary === 'partial' && (
          <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 font-medium">
            Some results could not be saved. Please note any critical findings manually.
          </div>
        )}
        {saveSummary === 'failed' && (
          <div role="alert" className="text-xs text-red-700 bg-red-50 border border-red-200 rounded-xl px-4 py-3 font-medium">
            Results could not be saved to history. Please take a screenshot.
          </div>
        )}

        {/* Summary stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-4">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Total</p>
            <p className="text-2xl font-bold text-slate-900">{results.length}</p>
          </div>
          <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-4">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Classified</p>
            <p className="text-2xl font-bold text-teal-600">{classified.length}</p>
          </div>
          <div className={['rounded-2xl border shadow-sm p-4', needsReview.length > 0 ? 'bg-amber-50 border-amber-300' : 'bg-white border-slate-200'].join(' ')}>
            <p className="text-[10px] font-bold text-amber-500 uppercase tracking-widest mb-1">Needs Review</p>
            <p className="text-2xl font-bold text-amber-700">{needsReview.length}</p>
          </div>
          <div className={['rounded-2xl border shadow-sm p-4', (rejected.length + errors.length) > 0 ? 'bg-red-50 border-red-200' : 'bg-white border-slate-200'].join(' ')}>
            <p className="text-[10px] font-bold text-red-400 uppercase tracking-widest mb-1">Failed / Rejected</p>
            <p className="text-2xl font-bold text-red-600">{rejected.length + errors.length}</p>
          </div>
        </div>

        {/* Results table */}
        <div className="bg-white rounded-2xl border border-slate-300 shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50">
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest w-16">#</th>
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Image</th>
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Prediction</th>
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Confidence</th>
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Risk</th>
                  <th className="text-left px-4 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Trust</th>
                  <th className="text-right px-4 py-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {results.map((item, idx) => {
                  const pred = topClass(item.classes);
                  const risk = pred ? (RISK[pred.id] ?? { label: '—', cls: 'text-slate-400 bg-slate-50 border-slate-200' }) : null;
                  const trust = item.trustResult?.recommendation
                    ? (TRUST_BADGE[item.trustResult.recommendation] ?? null)
                    : null;

                  return (
                    <tr key={item.caseId} className="hover:bg-slate-50 transition-colors">
                      <td className="px-4 py-3 tabular-nums text-slate-400 font-medium">{idx + 1}</td>
                      <td className="px-4 py-3">
                        <img
                          src={item.image}
                          alt={`Image ${idx + 1}`}
                          className="w-10 h-10 rounded-lg object-cover border border-slate-200"
                        />
                      </td>
                      <td className="px-4 py-3">
                        {item.status === 'error' ? (
                          <span className="text-amber-600 font-medium text-xs">Error</span>
                        ) : item.status === 'rejected' ? (
                          <span className="text-red-600 font-medium text-xs">Rejected</span>
                        ) : pred ? (
                          <span className="font-semibold text-slate-800">{pred.name}</span>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                        {item.status === 'error' && item.errorMessage && (
                          <p className="text-[11px] text-slate-400 mt-0.5 max-w-[180px] truncate">{item.errorMessage}</p>
                        )}
                      </td>
                      <td className="px-4 py-3 tabular-nums">
                        {pred && item.status !== 'rejected' ? (
                          <span className="font-semibold text-slate-700">{pred.score.toFixed(1)}%</span>
                        ) : (
                          <span className="text-slate-300">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        {risk ? (
                          <span className={`inline-flex items-center px-2 py-0.5 rounded border text-[10px] font-bold ${risk.cls}`}>
                            {risk.label}
                          </span>
                        ) : (
                          <span className="text-slate-300">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        {trust ? (
                          <span className={`inline-flex items-center px-2 py-0.5 rounded border text-[10px] font-bold ${trust.cls}`}>
                            {trust.label}
                          </span>
                        ) : (
                          <span className="text-slate-300">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        {(item.status === 'success' || item.status === 'rejected') ? (
                          <button
                            onClick={() => onViewDetail(item)}
                            className="text-xs font-semibold text-teal-600 hover:text-teal-800 transition-colors flex items-center gap-1 ml-auto"
                          >
                            Details
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                          </button>
                        ) : (
                          <span className="text-slate-300 text-xs">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* CTAs */}
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            onClick={onAnalyzeAnother}
            className="flex-1 py-2.5 px-4 rounded-full font-semibold text-sm bg-teal-600 text-white hover:bg-teal-700 active:bg-teal-800 transition-colors flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            New Analysis
          </button>
          <button
            onClick={onNavigateToHistory}
            className="flex-1 py-2.5 px-4 rounded-full font-semibold text-sm border border-slate-300 text-slate-600 hover:bg-slate-100 transition-colors flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-slate-400"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            View History
          </button>
        </div>
      </main>

      <footer className="py-6 text-center border-t border-slate-100 bg-white mt-auto">
        <p className="text-[11px] text-slate-400 leading-relaxed text-center">
          This tool generates probabilistic outputs from a machine learning model. It is designed
          to assist clinical decision-making and does not replace professional medical diagnosis.
        </p>
      </footer>
    </div>
  );
};

export default BatchResultsScreen;
