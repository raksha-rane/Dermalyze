import React from 'react';
import Button from './ui/Button';

interface LogoutConfirmScreenProps {
  onConfirm: () => void;
  onCancel: () => void;
}

const LogoutConfirmScreen: React.FC<LogoutConfirmScreenProps> = ({ onConfirm, onCancel }) => {
  return (
    <div className="w-full max-w-md">
      <div className="bg-white rounded-3xl border border-slate-200 p-10 sm:p-14 shadow-sm text-center">
        {/* Professional Logout Icon */}
        <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 mx-auto mb-8">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
            />
          </svg>
        </div>

        <h2 className="text-2xl font-bold text-slate-900 mb-3 tracking-tight">Confirm Logout</h2>

        <p className="text-slate-500 text-sm mb-10 leading-relaxed px-2">
          Are you sure you want to log out? Your current session will be terminated and you will
          need to log in again to access the system.
        </p>

        <div className="flex flex-col gap-3">
          <Button onClick={onConfirm}>Confirm Logout</Button>
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
        </div>
      </div>

      <div className="mt-6 text-center">
        <p className="text-[10px] font-bold text-slate-700 uppercase tracking-widest leading-relaxed drop-shadow-sm">
          Dermalyze Session Management
        </p>
      </div>
    </div>
  );
};

export default LogoutConfirmScreen;
