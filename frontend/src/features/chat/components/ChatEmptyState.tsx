'use client';

import { Bot, Sparkles } from 'lucide-react';

const PROMPTS = [
  'Who should be our first customers?',
  'What assumptions are we making about the market?',
  'Help me synthesize my recent customer interviews',
  'What are the riskiest parts of our hypothesis?',
];

export function ChatEmptyState({ onSelectPrompt }: { onSelectPrompt: (prompt: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-full px-6 py-8 text-center">
      <div
        className="w-14 h-14 rounded-2xl flex items-center justify-center mb-4"
        style={{ background: 'var(--gradient-brand)' }}
        aria-hidden="true"
      >
        <Bot className="w-7 h-7 text-[var(--color-on-brand)]" />
      </div>

      <h2 className="text-base font-semibold text-[var(--color-text-primary)] mb-1">Your private AI workspace</h2>
      <p className="text-sm text-[var(--color-text-muted)] max-w-xs mb-6">
        Think through your startup questions freely. Your raw conversation is never shown to your co-founder.
      </p>

      <div className="w-full max-w-sm space-y-2">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="w-3.5 h-3.5 text-[var(--color-primary)]" />
          <span className="text-xs text-[var(--color-text-muted)]">Try asking</span>
        </div>
        {PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => onSelectPrompt(prompt)}
            className="w-full text-left px-3 py-2.5 rounded-xl text-sm text-[var(--color-text-secondary)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] hover:bg-[var(--color-bg-hover)] hover:border-[var(--color-border-active)] hover:text-[var(--color-text-primary)] transition-all"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
