
import React from 'react';
import { ShieldCheckIcon } from '@heroicons/react/24/outline';
import Button from './ui/Button';

interface EmailVerificationScreenProps {
  email: string;
  onNavigateToLogin: () => void;
  onNavigateToHome: () => void;
  onResendEmail: () => Promise<void>;
}

const COOLDOWN_SECONDS = 60;

const EmailVerificationScreen: React.FC<EmailVerificationScreenProps> = ({
  email,
  onNavigateToLogin,
  onNavigateToHome,
  onResendEmail,
}) => {
  const [resending,  setResending]  = React.useState(false);
  const [resendMsg,  setResendMsg]  = React.useState<string | null>(null);
  const [cooldown,   setCooldown]   = React.useState(0); // seconds remaining

  // Tick the cooldown down every second
  React.useEffect(() => {
    if (cooldown <= 0) return;
    const t = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  const handleResend = async () => {
    setResending(true);
    setResendMsg(null);
    try {
      await onResendEmail();
      setResendMsg('Verification email resent. Please check your inbox.');
      setCooldown(COOLDOWN_SECONDS); // start cooldown only on success
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to resend. Please try again.';
      setResendMsg(message);
      // no cooldown on failure — let the user retry
    } finally {
      setResending(false);
    }
  };

  const resendDisabled = resending || cooldown > 0;

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <header className="w-full bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <button onClick={onNavigateToHome} className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <div className="w-11 h-11 bg-teal-600 rounded-xl flex items-center justify-center shadow-sm">
              <ShieldCheckIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 tracking-tight leading-tight">Dermalyze</h1>
              <p className="text-[9px] text-slate-500 uppercase tracking-wider font-semibold leading-tight">Clinical Decision Support</p>
            </div>
          </button>
          <div className="flex items-center gap-3">
            <button
              onClick={onNavigateToHome}
              className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
            >
              Home
            </button>
            <button
              onClick={onNavigateToLogin}
              className="px-4 py-2 text-sm font-semibold text-white bg-teal-600 hover:bg-teal-700 rounded-lg shadow-sm transition-colors"
            >
              Sign In
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          <div className="bg-white rounded-2xl border border-slate-300 shadow-sm p-10">
            <div className="text-center">
              {/* Mail icon */}
              <div className="inline-flex items-center justify-center w-16 h-16 bg-teal-50 rounded-2xl mb-6">
                <svg className="w-10 h-10 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
              </div>

              <h2 className="text-2xl font-bold text-slate-900 mb-3">
                Verify Your Email
              </h2>

              <p className="text-slate-600 mb-2 leading-relaxed">
                We've sent a verification link to:
              </p>
              <p className="text-teal-600 font-semibold mb-6 break-all">
                {email || 'your email address'}
              </p>

              <div className="bg-teal-50 border border-teal-200 rounded-xl p-4 mb-6 text-left">
                <p className="text-xs font-bold text-teal-700 uppercase tracking-widest mb-3">Next Steps</p>
                <ol className="space-y-2 text-sm text-teal-800 list-decimal list-inside">
                  <li>Open your email inbox</li>
                  <li>Click the verification link in the email from Dermalyze</li>
                  <li>Return here and sign in to your account</li>
                </ol>
              </div>

              <div className="space-y-3">
                <Button onClick={onNavigateToLogin} className="shadow-sm">
                  Continue to Sign In
                </Button>

                <button
                  onClick={handleResend}
                  disabled={resendDisabled}
                  className="w-full text-sm font-medium text-slate-500 hover:text-teal-600 transition-colors text-center py-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {resending
                    ? 'Resending…'
                    : cooldown > 0
                      ? `Resend available in ${cooldown}s`
                      : "Didn't receive it? Resend email"}
                </button>
              </div>

              {resendMsg && (
                <p className="mt-4 text-xs text-teal-600 font-medium">{resendMsg}</p>
              )}

              <div className="mt-6 p-4 bg-amber-50 border border-amber-200 rounded-xl">
                <p className="text-sm text-amber-800 leading-relaxed">
                  <strong>Tip:</strong> If you don't see the email, check your spam or junk folder.
                  The email may take a minute or two to arrive.
                </p>
              </div>
            </div>
          </div>

          {/* Footer Note */}
          <p className="mt-8 text-center text-xs text-slate-400 leading-relaxed px-4">
            By using this service, you confirm that you are a qualified medical professional using this system in accordance with professional guidelines.
          </p>
        </div>
      </div>
    </div>
  );
};

export default EmailVerificationScreen;
