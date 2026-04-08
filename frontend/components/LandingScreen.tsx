
import React from 'react';
import { ShieldCheckIcon } from '@heroicons/react/24/outline';

interface LandingScreenProps {
  onNavigateToLogin: () => void;
  onNavigateToSignup: () => void;
}

const LandingScreen: React.FC<LandingScreenProps> = ({ onNavigateToLogin, onNavigateToSignup }) => {
  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Header/Nav */}
      <header className="w-full bg-white border-b border-slate-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 bg-teal-600 rounded-xl flex items-center justify-center shadow-sm">
              <ShieldCheckIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 tracking-tight leading-tight">Dermalyze</h1>
              <p className="text-[9px] text-slate-500 uppercase tracking-wider font-semibold leading-tight">Clinical Decision Support</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onNavigateToLogin}
              className="px-5 py-2.5 text-sm font-semibold text-slate-700 hover:text-slate-900 hover:bg-slate-100 rounded-xl transition-colors"
            >
              Sign In
            </button>
            <button
              onClick={onNavigateToSignup}
              className="px-5 py-2.5 text-sm font-semibold text-white bg-teal-600 hover:bg-teal-700 rounded-xl shadow-sm transition-colors"
            >
              Create Account
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section with Teal Background */}
      <section className="bg-gradient-to-br from-teal-600 to-teal-700 text-white px-6 py-20">
        <div className="max-w-6xl mx-auto">
          <div className="max-w-3xl">
            <div className="mb-6">
              <span className="inline-block px-3 py-1.5 bg-white/20 backdrop-blur-sm text-white text-[11px] font-bold uppercase tracking-wider rounded-lg border border-white/30">
                For Medical Professionals
              </span>
            </div>

            <h1 className="text-4xl sm:text-5xl font-bold tracking-normal mb-6 leading-[1.25]">
              Dermatoscopic Image Analysis for Clinical Decision Support
            </h1>

            <p className="text-lg text-teal-50 mb-8 leading-relaxed">
              A diagnostic assistance tool for evaluating dermatoscopic images of pigmented and non-pigmented skin lesions. Provides classification support across seven common dermatological conditions to supplement your clinical assessment.
            </p>

            <div className="flex flex-wrap items-center gap-4">
              <button
                onClick={onNavigateToSignup}
                className="px-6 py-3 text-base font-semibold text-teal-700 bg-white hover:bg-teal-50 rounded-xl shadow-lg transition-colors flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                </svg>
                Create Account
              </button>
              <button
                onClick={onNavigateToLogin}
                className="px-6 py-3 text-base font-semibold text-white border-2 border-white/30 hover:bg-white/10 rounded-xl transition-colors"
              >
                Sign In
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <section className="flex-1">

        {/* Section 1: How It Works - Light gray background */}
        <div className="bg-slate-100 px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-10">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">How It Works</h2>
              <p className="text-sm text-slate-500">Four simple steps from image to documentation</p>
            </div>
            <div className="grid md:grid-cols-4 gap-6">
              {[
                { step: '1', title: 'Upload', desc: 'Upload a dermatoscopic image from your device or practice system', icon: 'M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z' },
                { step: '2', title: 'Analyze', desc: 'System classifies across seven diagnostic categories with confidence scores', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
                { step: '3', title: 'Review', desc: 'Examine results with visual overlay highlighting areas of significance', icon: 'M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z' },
                { step: '4', title: 'Document', desc: 'Add clinical notes and save to patient record or generate PDF report', icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z' },
              ].map((item, index) => (
                <div key={item.step} className="relative">
                  {index < 3 && (
                    <div className="hidden md:block absolute top-8 left-[60%] w-full h-0.5 bg-slate-300"></div>
                  )}
                  <div className="bg-white rounded-2xl border-2 border-slate-300 p-6 shadow-sm relative">
                    <div className="w-12 h-12 bg-teal-600 text-white rounded-full flex items-center justify-center text-lg font-bold mx-auto mb-4 shadow-md">
                      {item.step}
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900 text-center mb-2">{item.title}</h3>
                    <p className="text-sm text-slate-600 text-center leading-relaxed">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Section 2: Clinical Guidelines - White background */}
        <div className="bg-white px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <div className="mb-10">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Clinical Guidelines</h2>
              <p className="text-sm text-slate-500">Understand appropriate use and limitations</p>
            </div>
            <div className="grid md:grid-cols-2 gap-8">
              {/* Appropriate Use */}
              <div className="bg-emerald-50 rounded-2xl border-2 border-emerald-300 p-8 shadow-sm">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 bg-emerald-600 rounded-xl flex items-center justify-center shadow-sm">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-emerald-900">Appropriate Use Cases</h3>
                </div>
                <ul className="space-y-4">
                  {[
                    'Supplementary assessment during routine skin screening',
                    'Second opinion support for ambiguous clinical presentations',
                    'Documentation and longitudinal tracking of lesions',
                    'Educational reference for dermatology trainees',
                  ].map((item, i) => (
                    <li key={i} className="flex items-start gap-3 text-sm text-emerald-800">
                      <svg className="w-5 h-5 text-emerald-600 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Not Intended For */}
              <div className="bg-rose-50 rounded-2xl border-2 border-rose-300 p-8 shadow-sm">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 bg-rose-600 rounded-xl flex items-center justify-center shadow-sm">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-rose-900">Not Intended For</h3>
                </div>
                <ul className="space-y-4">
                  {[
                    'Sole basis for diagnosis or treatment decisions',
                    'Substitute for histopathological confirmation',
                    'Analysis of non-dermatoscopic clinical photographs',
                    'Lesions outside the seven supported categories',
                  ].map((item, i) => (
                    <li key={i} className="flex items-start gap-3 text-sm text-rose-800">
                      <svg className="w-5 h-5 text-rose-600 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Section 3: Diagnostic Categories - Teal gradient background */}
        <div className="bg-gradient-to-br from-teal-600 to-teal-700 px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-white mb-2">Supported Diagnostic Categories</h2>
              <p className="text-sm text-teal-100">Seven conditions with clinical risk stratification</p>
            </div>

            {/* High Priority Conditions - 3 larger cards */}
            <div className="grid md:grid-cols-3 gap-4 mb-4">
              {[
                { name: 'Melanoma', risk: 'Critical', desc: 'Requires immediate specialist evaluation', color: 'bg-red-500', border: 'border-red-400', text: 'text-red-700', bg: 'bg-red-50' },
                { name: 'Basal Cell Carcinoma', risk: 'High', desc: 'Requires prompt dermatology referral', color: 'bg-orange-500', border: 'border-orange-400', text: 'text-orange-700', bg: 'bg-orange-50' },
                { name: 'Actinic Keratoses', risk: 'Moderate', desc: 'Requires monitoring and follow-up', color: 'bg-amber-500', border: 'border-amber-400', text: 'text-amber-700', bg: 'bg-amber-50' },
              ].map((cat) => (
                <div key={cat.name} className={`${cat.bg} rounded-xl border-2 ${cat.border} p-6 shadow-sm`}>
                  <div className="flex items-center gap-2 mb-3">
                    <div className={`w-3.5 h-3.5 ${cat.color} rounded-full`}></div>
                    <span className={`text-xs font-bold ${cat.text} uppercase tracking-wide`}>{cat.risk} Risk</span>
                  </div>
                  <h4 className="text-lg font-bold text-slate-900 mb-1">{cat.name}</h4>
                  <p className="text-sm text-slate-600">{cat.desc}</p>
                </div>
              ))}
            </div>

            {/* Low Risk Conditions - 4 compact cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { name: 'Benign Keratosis', desc: 'Routine follow-up only', color: 'bg-emerald-500', border: 'border-emerald-300' },
                { name: 'Dermatofibroma', desc: 'No intervention typically', color: 'bg-cyan-500', border: 'border-cyan-300' },
                { name: 'Melanocytic Nevi', desc: 'Periodic observation only', color: 'bg-violet-500', border: 'border-violet-300' },
                { name: 'Vascular Lesions', desc: 'No treatment typically', color: 'bg-pink-500', border: 'border-pink-300' },
              ].map((cat) => (
                <div key={cat.name} className={`bg-white rounded-xl border-2 ${cat.border} p-4 shadow-sm`}>
                  <div className="flex items-center gap-2 mb-3">
                    <div className={`w-3.5 h-3.5 ${cat.color} rounded-full`}></div>
                    <span className="text-xs font-bold text-emerald-700 uppercase tracking-wide">Low Risk</span>
                  </div>
                  <h4 className="text-lg font-bold text-slate-900 mb-1">{cat.name}</h4>
                  <p className="text-sm text-slate-600">{cat.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Section 4: Image Requirements - Light background */}
        <div className="bg-slate-50 px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Image Requirements</h2>
              <p className="text-sm text-slate-500">Guidelines for optimal analysis results</p>
            </div>
            <div className="bg-white rounded-2xl border-2 border-slate-300 shadow-sm overflow-hidden">
              <div className="grid md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-300">
                <div className="p-8">
                  <div className="flex items-center gap-3 mb-5">
                    <div className="w-10 h-10 bg-teal-100 rounded-xl flex items-center justify-center">
                      <svg className="w-5 h-5 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900">Optimal Characteristics</h3>
                  </div>
                  <ul className="space-y-3">
                    {[
                      'Dermatoscopic images from standard polarized or non-polarized devices',
                      'Lesion centered in frame with clear, sharp focus',
                      'Uniform lighting without significant glare or shadows',
                      'Minimal surrounding normal skin for accurate analysis',
                    ].map((item, i) => (
                      <li key={i} className="flex items-start gap-3 text-sm text-slate-700">
                        <span className="w-1.5 h-1.5 bg-teal-500 rounded-full mt-2 shrink-0"></span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="p-8">
                  <div className="flex items-center gap-3 mb-5">
                    <div className="w-10 h-10 bg-amber-100 rounded-xl flex items-center justify-center">
                      <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-semibold text-slate-900">Known Limitations</h3>
                  </div>
                  <ul className="space-y-3">
                    {[
                      'Not designed for standard clinical photographs or smartphone images',
                      'May perform suboptimally on heavily obscured or artifact-laden images',
                      'Results should be interpreted in context of patient history',
                      'Performance varies with image quality and lesion presentation',
                    ].map((item, i) => (
                      <li key={i} className="flex items-start gap-3 text-sm text-slate-700">
                        <span className="w-1.5 h-1.5 bg-amber-500 rounded-full mt-2 shrink-0"></span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Section 5: System Features - White background */}
        <div className="bg-white px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-slate-900 mb-2">System Features</h2>
              <p className="text-sm text-slate-500">Built for clinical workflow integration</p>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                {
                  title: 'Visual Explanation',
                  desc: 'Heatmap overlay highlights diagnostically significant regions, helping you understand which areas most influenced the classification result.',
                  icon: 'M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z',
                  color: 'bg-teal-600',
                  iconColor: 'text-white',
                },
                {
                  title: 'Case Archive',
                  desc: 'Maintain comprehensive records with clinical notes for longitudinal tracking, audits, and building your diagnostic reference library.',
                  icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2',
                  color: 'bg-teal-600',
                  iconColor: 'text-white',
                },
                {
                  title: 'Data Security',
                  desc: 'End-to-end encryption ensures patient images are protected. Only you can access your uploaded medical images—not even administrators.',
                  icon: 'M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z',
                  color: 'bg-teal-600',
                  iconColor: 'text-white',
                },
              ].map((feature) => (
                <div key={feature.title} className="text-center p-6 bg-white rounded-2xl border-2 border-slate-300 shadow-sm">
                  <div className={`w-14 h-14 ${feature.color} rounded-xl flex items-center justify-center mx-auto mb-5 shadow-md`}>
                    <svg className={`w-7 h-7 ${feature.iconColor}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={feature.icon} />
                    </svg>
                  </div>
                  <h4 className="text-lg font-semibold text-slate-900 mb-3">{feature.title}</h4>
                  <p className="text-sm text-slate-600 leading-relaxed">{feature.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

      </section>

      {/* Clinical Disclaimer */}
      <div className="bg-red-50 border-y-2 border-red-300 py-6 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-start gap-4">
            <svg className="w-6 h-6 text-red-600 shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <p className="text-sm font-bold text-red-900 uppercase tracking-wider mb-2">Medical Device Disclaimer</p>
              <p className="text-sm text-red-800 leading-relaxed mb-3">
                <strong>This is a decision support tool only.</strong> It is not a medical device and does not provide diagnoses. All output must be independently verified by a qualified dermatologist or healthcare professional. Clinical diagnosis requires correlation with patient history, physical examination, and when indicated, histopathological confirmation.
              </p>
              <p className="text-sm text-red-800 leading-relaxed">
                <strong>No warranty is provided</strong> regarding diagnostic accuracy. Users assume full professional responsibility for all clinical decisions. This tool should never delay appropriate patient care or specialist referral.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t-2 border-slate-300 py-8 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-xs text-slate-400 uppercase tracking-widest font-bold">
            © {new Date().getFullYear()} Dermalyze · Clinical Decision Support Tool for Dermatology
          </p>
          <p className="text-xs text-slate-400 mt-2">
            For use by qualified medical professionals only
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LandingScreen;
