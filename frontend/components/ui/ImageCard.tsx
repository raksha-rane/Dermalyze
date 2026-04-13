import React from 'react';

interface ImageCardProps {
  imageUrl: string | null | undefined;
  alt?: string;
}

/**
 * ImageCard — Displays the analyzed dermatoscopy image.
 *
 * UX Laws applied:
 * - Aesthetic-Usability Effect: clean border, consistent aspect ratio, and
 *   a subtle background create a polished clinical feel.
 * - Law of Common Region: the bordered card groups the image as a distinct
 *   visual unit, preventing it from bleeding into surrounding content.
 * - Tesler's Law: no caption clutter — the image speaks for itself in
 *   context. Complexity that cannot be removed is hidden (metadata).
 */
const ImageCard: React.FC<ImageCardProps> = ({ imageUrl, alt = 'Analyzed lesion' }) => {
  return (
    <section className="bg-white rounded-xl border border-slate-300 shadow-sm p-5">
      <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-4">
        Analyzed Image
      </h3>
      <div className="aspect-square w-full bg-slate-50 rounded-lg overflow-hidden border border-slate-100 flex items-center justify-center">
        {imageUrl ? (
          <img src={imageUrl} alt={alt} className="w-full h-full object-cover" draggable={false} />
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
            <span className="text-xs">No image available</span>
          </div>
        )}
      </div>
    </section>
  );
};

export default ImageCard;
