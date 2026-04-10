import React from 'react';
import {
  ChartBarSquareIcon,
  ClipboardDocumentCheckIcon,
  LockClosedIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';

interface AboutScreenProps {
  onBack: () => void;
}

type IconComponent = React.ComponentType<React.SVGProps<SVGSVGElement>>;

const ABOUT_PILLARS: Array<{ title: string; description: string; Icon: IconComponent }> = [
  {
    title: 'Clinical decision support',
    description:
      'Dermalyze helps you review dermatoscopic cases with a consistent seven-class output so you can triage and document faster.',
    Icon: ShieldCheckIcon,
  },
  {
    title: 'Interpretable output',
    description:
      'Each result includes top prediction, confidence score, full score distribution, and heatmap overlay to support transparent review.',
    Icon: ChartBarSquareIcon,
  },
  {
    title: 'Workflow-ready records',
    description:
      'Save clinician notes, revisit prior analyses in History, and export PDF summaries for case handoff, teaching, or charting.',
    Icon: ClipboardDocumentCheckIcon,
  },
];

const ANALYSIS_FLOW = [
  {
    step: '1',
    title: 'Upload',
    detail: 'JPEG or PNG dermatoscopic image (up to 10 MB).',
  },
  {
    step: '2',
    title: 'Infer',
    detail: 'Model evaluates lesion patterns across seven supported classes.',
  },
  {
    step: '3',
    title: 'Interpret',
    detail: 'Review confidence, class ranking, and Grad-CAM visual evidence.',
  },
  {
    step: '4',
    title: 'Document',
    detail: 'Add clinical notes and save to History or PDF report.',
  },
];

const PRIVACY_COMMITMENTS = [
  'Images are encrypted before storage.',
  'Case history is scoped to your authenticated account.',
  'No public image URLs are used for medical records.',
  'Account deletion removes your associated analysis data.',
];

const RESPONSIBLE_USE = {
  do: [
    'Use outputs alongside dermoscopic exam and patient history.',
    'Prioritize immediate follow-up for critical and high-risk predictions.',
    'Track lesion change over time using History records.',
  ],
  avoid: [
    'Using model output as a standalone diagnosis.',
    'Relying on non-dermatoscopic or poor-quality images.',
    'Delaying referral or biopsy when clinical concern is high.',
  ],
};

const AboutScreen: React.FC<AboutScreenProps> = ({ onBack }) => {
  return (
    <div className="flex-1 flex flex-col bg-slate-50 text-slate-900">
      <section className="bg-gradient-to-br from-teal-600 to-teal-700 px-6 py-12 sm:py-16">
        <div className="max-w-6xl mx-auto">
          <span className="inline-block px-3 py-1.5 bg-white/20 backdrop-blur-sm text-white text-[11px] font-bold uppercase tracking-wider rounded-lg border border-white/30 mb-4">
            Product Overview
          </span>
          <h1 className="text-3xl sm:text-4xl font-bold text-white tracking-tight mb-4">
            About Dermalyze
          </h1>
          <p className="text-teal-50 text-base leading-relaxed max-w-3xl mb-8">
            Dermalyze is built for clinicians who need faster lesion review, consistent
            risk-oriented outputs, and clear documentation support without replacing professional
            judgment.
          </p>
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white border-2 border-white/30 hover:bg-white/10 rounded-xl transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
            Back to Dashboard
          </button>
        </div>
      </section>

      <main className="flex-1">
        <section className="bg-slate-100 px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">
                What this platform is designed to do
              </h2>
              <p className="text-sm text-slate-500">
                Core capabilities focused on day-to-day clinical workflow.
              </p>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              {ABOUT_PILLARS.map(({ title, description, Icon }) => (
                <article
                  key={title}
                  className="bg-white rounded-2xl border-2 border-slate-300 p-6 shadow-sm"
                >
                  <div className="w-12 h-12 bg-teal-600 rounded-xl flex items-center justify-center mb-4 shadow-sm">
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 mb-2">{title}</h3>
                  <p className="text-sm text-slate-600 leading-relaxed">{description}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="bg-white px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">
                What happens during each analysis
              </h2>
              <p className="text-sm text-slate-500">
                A predictable flow from upload to clinical documentation.
              </p>
            </div>
            <div className="grid md:grid-cols-4 gap-4">
              {ANALYSIS_FLOW.map(({ step, title, detail }) => (
                <article
                  key={step}
                  className="bg-slate-50 rounded-xl border-2 border-slate-300 p-5"
                >
                  <div className="w-10 h-10 bg-teal-600 text-white rounded-full flex items-center justify-center font-bold mb-3">
                    {step}
                  </div>
                  <h3 className="text-base font-semibold text-slate-900 mb-2">{title}</h3>
                  <p className="text-sm text-slate-600 leading-relaxed">{detail}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="bg-slate-50 px-6 py-12">
          <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-6">
            <article className="bg-white rounded-2xl border-2 border-slate-300 p-6 shadow-sm">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-teal-100 rounded-xl flex items-center justify-center">
                  <LockClosedIcon className="w-5 h-5 text-teal-700" />
                </div>
                <h2 className="text-lg font-bold text-slate-900">Privacy and data handling</h2>
              </div>
              <ul className="space-y-3">
                {PRIVACY_COMMITMENTS.map((item) => (
                  <li key={item} className="flex items-start gap-2 text-sm text-slate-700">
                    <span className="w-1.5 h-1.5 bg-teal-500 rounded-full mt-2 shrink-0"></span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </article>

            <article className="bg-white rounded-2xl border-2 border-slate-300 p-6 shadow-sm">
              <h2 className="text-lg font-bold text-slate-900 mb-4">Responsible clinical use</h2>
              <div className="grid sm:grid-cols-2 gap-4">
                <div className="bg-emerald-50 rounded-xl border border-emerald-200 p-4">
                  <p className="text-xs font-bold text-emerald-700 uppercase tracking-wider mb-2">
                    Do
                  </p>
                  <ul className="space-y-2">
                    {RESPONSIBLE_USE.do.map((item) => (
                      <li key={item} className="text-sm text-emerald-800 leading-relaxed">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="bg-rose-50 rounded-xl border border-rose-200 p-4">
                  <p className="text-xs font-bold text-rose-700 uppercase tracking-wider mb-2">
                    Avoid
                  </p>
                  <ul className="space-y-2">
                    {RESPONSIBLE_USE.avoid.map((item) => (
                      <li key={item} className="text-sm text-rose-800 leading-relaxed">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </article>
          </div>
        </section>
      </main>

      <footer className="bg-white border-t-2 border-slate-300 py-8 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-xs text-slate-400 uppercase tracking-widest font-bold">
            Dermalyze supports clinical review and documentation workflows
          </p>
          <p className="text-xs text-slate-400 mt-2">Not a standalone diagnostic system</p>
        </div>
      </footer>
    </div>
  );
};

export default AboutScreen;
