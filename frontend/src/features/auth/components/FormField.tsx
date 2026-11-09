'use client';

import { useState, type InputHTMLAttributes, type Ref } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { cn } from '@/lib/utils';

export const inputClass = cn(
  'w-full h-11 rounded-xl px-3.5 text-base sm:text-sm outline-none transition-colors',
  'bg-[var(--color-input-bg)] border border-[var(--color-border)] text-[var(--color-text-primary)]',
  'placeholder:text-[var(--color-text-disabled)] hover:border-[var(--color-border-active)]',
  'focus-visible:border-[var(--color-focus)] focus-visible:ring-2 focus-visible:ring-[var(--color-focus)]/25 focus-visible:outline-none',
  'aria-[invalid=true]:border-[var(--color-error)]',
);

interface FormFieldProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'id'> {
  id: string;
  label: string;
  error?: string;
  hint?: string;
  inputRef?: Ref<HTMLInputElement>;
  /** Adds a show/hide toggle (password fields). */
  revealable?: boolean;
}

/** Labelled input with accessible error/hint wiring and optional show/hide. */
export function FormField({ id, label, error, hint, inputRef, revealable, type, className, ...rest }: FormFieldProps) {
  const [revealed, setRevealed] = useState(false);
  const describedBy = [hint && `${id}-hint`, error && `${id}-error`].filter(Boolean).join(' ') || undefined;

  return (
    <div className={className}>
      <label htmlFor={id} className="block mb-1.5 text-sm font-medium text-[var(--color-text-primary)]">
        {label}
      </label>
      <div className="relative">
        <input
          id={id}
          ref={inputRef}
          type={revealable ? (revealed ? 'text' : 'password') : type}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          className={cn(inputClass, revealable && 'pr-11')}
          {...rest}
        />
        {revealable && (
          <button
            type="button"
            onClick={() => setRevealed((v) => !v)}
            aria-label={revealed ? 'Hide password' : 'Show password'}
            aria-pressed={revealed}
            aria-controls={id}
            className="absolute right-1.5 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg flex items-center justify-center text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)]"
          >
            {revealed ? <EyeOff className="w-4 h-4" aria-hidden="true" /> : <Eye className="w-4 h-4" aria-hidden="true" />}
          </button>
        )}
      </div>
      {hint && !error && (
        <p id={`${id}-hint`} className="mt-1.5 text-xs text-[var(--color-text-muted)]">
          {hint}
        </p>
      )}
      {error && (
        <p id={`${id}-error`} className="mt-1.5 text-xs text-[var(--color-error)]">
          {error}
        </p>
      )}
    </div>
  );
}
