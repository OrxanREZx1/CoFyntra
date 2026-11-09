'use client';

import { Bot } from 'lucide-react';

/** Shown while the assistant reply for the latest message is pending. */
export function ChatLoadingIndicator({ label }: { label: string }) {
  return (
    <div className="flex items-end gap-2.5" role="status" aria-live="polite">
      <div
        className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
        style={{ background: 'var(--gradient-brand)' }}
        aria-hidden="true"
      >
        <Bot className="w-3.5 h-3.5 text-[var(--color-on-brand)]" />
      </div>
      <div className="flex items-center gap-1.5 px-3.5 py-3 rounded-2xl rounded-bl-sm bg-[var(--color-bg-elevated)] border border-[var(--color-border)]">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            aria-hidden="true"
            className="w-1.5 h-1.5 rounded-full bg-[var(--color-text-muted)] motion-safe:animate-[chat-dot_1.2s_ease-in-out_infinite]"
            style={{ animationDelay: `${i * 0.2}s` }}
          />
        ))}
        <span className="sr-only">{label}</span>
      </div>
    </div>
  );
}
