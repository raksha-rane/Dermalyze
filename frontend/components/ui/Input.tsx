import React, { useId } from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
}

const Input: React.FC<InputProps> = ({ label, error, ...props }) => {
  const uid = useId();
  const inputId = props.id ?? uid;
  return (
    <div className="flex flex-col gap-1.5 mb-4">
      <label htmlFor={inputId} className="text-sm font-medium text-slate-600 px-0.5">
        {label}
      </label>
      <input
        id={inputId}
        aria-invalid={!!error}
        aria-describedby={error ? `${inputId}-error` : undefined}
        {...props}
        className="w-full px-4 py-2.5 rounded-lg border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 transition-all text-slate-800 placeholder:text-slate-400"
      />
      {error && (
        <p id={`${inputId}-error`} role="alert" className="text-xs text-red-600 px-0.5">
          {error}
        </p>
      )}
    </div>
  );
};

export default Input;
