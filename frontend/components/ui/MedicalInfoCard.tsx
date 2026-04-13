import React from 'react';
import { ClassInfo, getRiskSeverity, getRiskBadgeStyles } from '../../lib/classInfo';

interface MedicalInfoCardProps {
  info: ClassInfo;
}

/**
 * MedicalInfoCard — Structured clinical information about the predicted condition.
 *
 * UX Laws applied:
 * - Law of Proximity: related fields (Description, Risk, Demographics,
 *   Features) are grouped with consistent spacing so each reads as a unit.
 * - Law of Common Region: each field has a subtle label + value pair inside
 *   a shared container, signaling that they belong together.
 * - Miller's Law: information is chunked into 4 scannable sections rather
 *   than presented as continuous prose.
 * - Jakob's Law: field layout mirrors EHR/clinical-report conventions that
 *   dermatologists already recognise (label-above-value pattern).
 */
const MedicalInfoCard: React.FC<MedicalInfoCardProps> = ({ info }) => {
  const severity = getRiskSeverity(info.riskLevel);
  const riskStyle = getRiskBadgeStyles(severity);

  const fields: { label: string; value: string; highlight?: boolean }[] = [
    { label: 'Description', value: info.description },
    { label: 'Risk Level', value: info.riskLevel, highlight: true },
    { label: 'Common In', value: info.commonIn },
    { label: 'Clinical Features', value: info.keyFeatures },
  ];

  return (
    <section className="bg-white rounded-xl border border-slate-300 shadow-sm p-5">
      <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-5">
        Condition Information
      </h3>

      <div className="space-y-5">
        {fields.map((field) => (
          <div key={field.label}>
            <dt className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">
              {field.label}
            </dt>
            <dd
              className={`text-sm leading-relaxed ${
                field.highlight ? `font-semibold ${riskStyle.text}` : 'text-slate-600'
              }`}
            >
              {field.value}
            </dd>
          </div>
        ))}
      </div>
    </section>
  );
};

export default MedicalInfoCard;
