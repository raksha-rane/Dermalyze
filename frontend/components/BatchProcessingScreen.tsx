import React, { useEffect, useRef, useState } from 'react';
import { classifyImage, ApiError } from '../lib/api';
import type { BatchItemResult, InferenceMetadata } from '../lib/types';

interface BatchProcessingScreenProps {
  images: string[];
  metadata: InferenceMetadata;
  onComplete: (results: BatchItemResult[]) => void;
  onError: (message?: string) => void;
}

const BatchProcessingScreen: React.FC<BatchProcessingScreenProps> = ({
  images,
  metadata,
  onComplete,
  onError,
}) => {
  const calledRef = useRef(false);
  const [items, setItems] = useState<BatchItemResult[]>(() =>
    images.map((img, i) => ({
      imageIndex: i,
      image: img,
      caseId: crypto.randomUUID(),
      status: 'pending',
      metadata,
    }))
  );
  const [currentIndex, setCurrentIndex] = useState(-1);

  // Update a single item by index, immutably
  const updateItem = (index: number, patch: Partial<BatchItemResult>) => {
    setItems((prev) => prev.map((it) => (it.imageIndex === index ? { ...it, ...patch } : it)));
  };

  useEffect(() => {
    if (calledRef.current) return;
    calledRef.current = true;

    if (!images.length) {
      onError('No images were provided for batch classification.');
      return;
    }

    const runBatch = async () => {
      const results: BatchItemResult[] = images.map((img, i) => ({
        imageIndex: i,
        image: img,
        caseId: crypto.randomUUID(),
        status: 'pending',
        metadata,
      }));

      for (let i = 0; i < results.length; i++) {
        setCurrentIndex(i);
        // Mark as processing
        results[i] = { ...results[i], status: 'processing' };
        setItems([...results]);

        try {
          const result = await classifyImage(results[i].image, false, metadata);

          const status =
            result.trustResult.recommendation === 'reject' ? 'rejected' : 'success';

          results[i] = {
            ...results[i],
            status,
            classes: result.classes,
            gradcamImage: result.gradcamImage,
            trustResult: result.trustResult,
          };
        } catch (err: unknown) {
          let msg =
            err instanceof Error
              ? err.message
              : 'Classification failed.';
              
          if (err instanceof ApiError && err.status === 500 && msg.includes('500')) {
            msg = 'The image could not be processed due to a server error. The image may be corrupted or in an unsupported format.';
          }
          results[i] = { ...results[i], status: 'error', errorMessage: msg };
        }

        setItems([...results]);

        // Brief pause between requests to be kind to the rate limiter
        if (i < results.length - 1) {
          await new Promise((r) => setTimeout(r, 300));
        }
      }

      setCurrentIndex(-1);
      onComplete(results);
    };

    runBatch();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const done = items.filter((it) => it.status !== 'pending' && it.status !== 'processing').length;
  const total = items.length;
  const progressPct = total > 0 ? Math.round((done / total) * 100) : 0;

  const statusIcon = (status: BatchItemResult['status'], isActive: boolean) => {
    if (isActive) {
      return (
        <span className="inline-flex items-center justify-center w-6 h-6 rounded-full border-2 border-teal-600 border-t-transparent animate-spin shrink-0" />
      );
    }
    switch (status) {
      case 'success':
        return (
          <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-teal-100 text-teal-700 shrink-0">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
            </svg>
          </span>
        );
      case 'rejected':
        return (
          <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-red-100 text-red-600 shrink-0">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </span>
        );
      case 'error':
        return (
          <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-amber-100 text-amber-600 shrink-0">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 9v4m0 4h.01" />
            </svg>
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center justify-center w-6 h-6 rounded-full border-2 border-slate-200 shrink-0" />
        );
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-50">
      <main className="flex-1 flex items-center justify-center p-6 sm:p-12">
        <div className="max-w-lg w-full">
          <div className="bg-white rounded-3xl border border-slate-300 p-8 sm:p-10 shadow-sm">
            {/* Header */}
            <div className="text-center mb-8">
              <div className="relative inline-flex items-center justify-center w-16 h-16 mb-4">
                <div className="absolute inset-0 rounded-full border-4 border-slate-100" />
                <div
                  className="absolute inset-0 rounded-full border-4 border-teal-500 border-t-transparent animate-spin"
                  style={{ animationDuration: '1.2s' }}
                />
                <span className="text-sm font-bold text-teal-600 tabular-nums">
                  {progressPct}%
                </span>
              </div>
              <h2 className="text-xl font-bold text-slate-900 tracking-tight">
                Analyzing Batch…
              </h2>
              <p className="text-sm text-slate-400 mt-1">
                {done} of {total} image{total !== 1 ? 's' : ''} processed
              </p>
            </div>

            {/* Progress bar */}
            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden mb-6">
              <div
                className="h-full bg-teal-500 rounded-full transition-all duration-500"
                style={{ width: `${progressPct}%` }}
              />
            </div>

            {/* Per-item tracker */}
            <ul className="space-y-2 max-h-64 overflow-y-auto pr-1">
              {items.map((item, idx) => {
                const isActive = idx === currentIndex;
                return (
                  <li
                    key={item.caseId}
                    className={[
                      'flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors',
                      isActive ? 'bg-teal-50 border border-teal-200' : 'border border-transparent',
                    ].join(' ')}
                  >
                    {statusIcon(item.status, isActive)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-slate-700 truncate">
                        Image {idx + 1}
                      </p>
                      {item.status === 'error' && item.errorMessage && (
                        <p className="text-xs text-amber-600 truncate">{item.errorMessage}</p>
                      )}
                      {item.status === 'rejected' && (
                        <p className="text-xs text-red-500">Rejected by trust layer</p>
                      )}
                      {item.status === 'success' && item.classes?.[0] && (
                        <p className="text-xs text-teal-600">
                          {item.classes[0].name} — {item.classes[0].score.toFixed(1)}%
                        </p>
                      )}
                    </div>
                    <div className="shrink-0">
                      <img
                        src={item.image}
                        alt={`Image ${idx + 1}`}
                        className="w-9 h-9 rounded-lg object-cover border border-slate-200"
                      />
                    </div>
                  </li>
                );
              })}
            </ul>

            {/* Footer note */}
            <p className="text-[11px] text-slate-400 text-center mt-6 leading-relaxed">
              Images are analyzed sequentially. Please keep this page open.
            </p>
          </div>
        </div>
      </main>

      <footer className="py-8 text-center bg-slate-50">
        <p className="text-[11px] font-medium text-slate-400 uppercase tracking-widest leading-relaxed px-6">
          Designed to assist medical professionals. Not a replacement for clinical diagnosis.
        </p>
      </footer>
    </div>
  );
};

export default BatchProcessingScreen;
