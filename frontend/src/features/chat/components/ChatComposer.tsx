'use client';

import { useLayoutEffect, type RefObject } from 'react';
import { ArrowUp } from 'lucide-react';
import { cn } from '@/lib/utils';

const MAX_HEIGHT_PX = 180;

interface ChatComposerProps {
  value: string;
  onChange: (value: string) => void;
  /** Called with the current text when the user submits a non-empty message. */
  onSubmit: () => void;
  /** A reply is in flight: typing is allowed, submitting is not. */
  isSending?: boolean;
  /** Composer fully unavailable (e.g. history not loaded). */
  disabled?: boolean;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
}

export function ChatComposer({
  value,
  onChange,
  onSubmit,
  isSending = false,
  disabled = false,
  textareaRef,
}: ChatComposerProps) {
  const canSend = value.trim().length > 0 && !isSending && !disabled;

  // Grow with content up to MAX_HEIGHT_PX, then scroll.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [value, textareaRef]);

  const submit = () => {
    if (canSend) onSubmit();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Ignore Enter while an IME composition (e.g. CJK input) is active.
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form
      className="flex-shrink-0 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] border-t border-[var(--color-border)] bg-[var(--color-bg-base)]"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <div
        className={cn(
          'mx-auto max-w-3xl flex items-end gap-2 px-3 py-2 rounded-xl',
          'bg-[var(--color-bg-elevated)] border border-[var(--color-border)]',
          'focus-within:border-[var(--color-border-active)] transition-colors',
          disabled && 'opacity-50',
        )}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask your AI anything…"
          disabled={disabled}
          rows={1}
          enterKeyHint="send"
          aria-label="Message"
          aria-describedby="chat-composer-hint"
          className={cn(
            // 16px on mobile prevents iOS from zooming into the field on focus.
            'flex-1 resize-none bg-transparent text-base sm:text-sm text-[var(--color-text-primary)]',
            'placeholder:text-[var(--color-text-disabled)] outline-none',
            'min-h-[24px] leading-6 py-0.5 disabled:cursor-not-allowed',
          )}
          style={{ maxHeight: MAX_HEIGHT_PX }}
        />
        <button
          type="submit"
          disabled={!canSend}
          aria-label={isSending ? 'Waiting for reply' : 'Send message'}
          aria-busy={isSending}
          className={cn(
            'w-8 h-8 sm:w-7 sm:h-7 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors',
            canSend
              ? 'bg-[var(--color-primary)] text-[var(--color-on-primary)] hover:bg-[var(--color-primary-hover)] shadow-sm'
              : 'bg-[var(--color-bg-hover)] text-[var(--color-text-disabled)] cursor-not-allowed',
          )}
        >
          {isSending ? (
            <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
          ) : (
            <ArrowUp className="w-3.5 h-3.5" />
          )}
        </button>
      </div>
      <p id="chat-composer-hint" className="mt-1.5 text-center text-[10px] text-[var(--color-text-disabled)]">
        <span className="hidden sm:inline">Enter to send · Shift+Enter for a new line · </span>
        AI responses are for discovery exploration only — not financial, legal, or professional advice.
      </p>
    </form>
  );
}
