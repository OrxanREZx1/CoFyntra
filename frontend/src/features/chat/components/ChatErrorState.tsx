'use client';

import { AlertTriangle, RotateCw } from 'lucide-react';
import type { ApiError } from '@/lib/api/errors';

/** Shown when the conversation history fails to load. */
export function ChatErrorState({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 text-center" role="alert">
      <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-3 bg-[var(--color-error-muted)]">
        <AlertTriangle className="w-5 h-5 text-[var(--color-error)]" />
      </div>
      <h2 className="text-sm font-semibold text-[var(--color-text-primary)] mb-1">
        Couldn&apos;t load your conversation
      </h2>
      <p className="text-xs text-[var(--color-text-muted)] max-w-xs mb-4">{error.userMessage}</p>
      <button
        type="button"
        onClick={onRetry}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-[var(--color-text-primary)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] hover:bg-[var(--color-bg-hover)] hover:border-[var(--color-border-active)] transition-colors"
      >
        <RotateCw className="w-3.5 h-3.5" /> Try again
      </button>
    </div>
  );
}
