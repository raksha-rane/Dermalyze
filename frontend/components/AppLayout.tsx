import React, { useState } from 'react';
import { Bars3Icon, ShieldCheckIcon } from '@heroicons/react/24/outline';
import { useNavigate } from 'react-router-dom';
import Sidebar from './Sidebar';
import { ROUTES } from '../App';

interface AppLayoutProps {
  children: React.ReactNode;
  onLogout: () => void;
}

/**
 * AppLayout — wraps all protected routes with the dual-tier sidebar.
 *
 * Desktop (≥ lg): Persistent sticky sidebar beside scrollable content.
 * Mobile  (<  lg): Hidden sidebar; hamburger in a top bar opens it as
 *                  a fixed overlay with a darkened backdrop.
 */
const AppLayout: React.FC<AppLayoutProps> = ({ children, onLogout }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen bg-slate-50">
      {/* ── Sidebar ── */}
      <Sidebar
        onLogout={onLogout}
        mobileOpen={mobileOpen}
        onMobileClose={() => setMobileOpen(false)}
      />

      {/* ── Main content area ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-auto">
        {/* Mobile-only top bar */}
        <div className="lg:hidden flex items-center gap-3 px-4 py-3 bg-white border-b border-slate-200 shadow-sm sticky top-0 z-20">
          <button
            onClick={() => setMobileOpen(true)}
            className="p-2 rounded-lg hover:bg-slate-100 text-slate-500 transition-colors"
            aria-label="Open navigation menu"
          >
            <Bars3Icon className="w-5 h-5" />
          </button>
          <button onClick={() => navigate(ROUTES.dashboard)} className="flex items-center gap-2">
            <div className="bg-teal-600 rounded-lg p-1.5">
              <ShieldCheckIcon className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-slate-900 tracking-tight text-base">Dermalyze</span>
          </button>
        </div>

        {/* Page content */}
        {children}
      </div>
    </div>
  );
};

export default AppLayout;
