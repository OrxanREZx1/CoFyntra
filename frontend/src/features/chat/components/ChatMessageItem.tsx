'use client';

import { memo } from 'react';
import { AlertCircle, Bot, Pencil, RotateCw } from 'lucide-react';
import { cn, formatFullTimestamp } from '@/lib/utils';
import { FounderAvatar } from '@/components/ui/FounderAvatar';
import type { FounderId } from '@/types';
import type { DisplayChatMessage } from '../hooks/useFounderChat';
import { MessageContent } from './MessageContent';

interface ChatMessageItemProps {
  message: DisplayChatMessage;
  founderId: FounderId;
  isDemo: boolean;
  onRetry: (messageId: string) => void;
  onEdit: (messageId: string) => void;
}

function MessageMeta({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        'flex items-center gap-1.5 text-[10px] text-[var(--color-text-disabled)] transition-opacity',
        // Always faintly visible on touch screens; revealed on hover/focus on larger screens.
        'sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100',
        className,
      )}
    >
      {children}
    </div>
  );
}

function ChatMessageItemImpl({ message, founderId, isDemo, onRetry, onEdit }: ChatMessageItemProps) {
  const time = <time dateTime={message.createdAt}>{formatFullTimestamp(message.createdAt)}</time>;

  if (message.role === 'user') {
    const failed = message.status === 'failed';
    const sending = message.status === 'sending';

    return (
      <div className="flex items-end gap-2.5 justify-end group">
        <div className="flex flex-col items-end gap-1 min-w-0 max-w-[85%] sm:max-w-[75%]">
          <div
            className={cn(
              'px-3.5 py-2.5 rounded-2xl rounded-br-sm text-sm leading-relaxed',
              'whitespace-pre-wrap max-w-full wrap-anywhere',
              'bg-[var(--color-primary)] text-[var(--color-on-primary)] transition-opacity',
              sending && 'opacity-70',
              failed && 'opacity-60 ring-1 ring-[var(--color-error)]',
            )}
          >
            {message.content}
          </div>

          {failed ? (
            <div className="flex flex-wrap items-center justify-end gap-x-3 gap-y-1 text-[11px]" role="alert">
              <span className="flex items-center gap-1 text-[var(--color-error)]">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {'error' in message && message.error ? message.error : 'Not sent.'}
              </span>
              <button
                type="button"
                onClick={() => onRetry(message.id)}
                className="flex items-center gap-1 font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
              >
                <RotateCw className="w-3 h-3" /> Retry
              </button>
              <button
                type="button"
                onClick={() => onEdit(message.id)}
                className="flex items-center gap-1 font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
              >
                <Pencil className="w-3 h-3" /> Edit
              </button>
            </div>
          ) : sending ? (
            <span className="text-[10px] text-[var(--color-text-disabled)]">Sending…</span>
          ) : (
            <MessageMeta>{time}</MessageMeta>
          )}
        </div>
        <FounderAvatar founderId={founderId} size="sm" className="mb-5" />
      </div>
    );
  }

  return (
    <div className="flex items-end gap-2.5 group">
      <div
        className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mb-5"
        style={{ background: 'var(--gradient-brand)' }}
        aria-hidden="true"
      >
        <Bot className="w-3.5 h-3.5 text-[var(--color-on-brand)]" />
      </div>
      <div className="flex flex-col gap-1 min-w-0 max-w-[85%] sm:max-w-[80%]">
        <div
          className={cn(
            'px-3.5 py-2.5 rounded-2xl rounded-bl-sm text-sm leading-relaxed max-w-full wrap-anywhere',
            'bg-[var(--color-bg-elevated)] border border-[var(--color-border)] text-[var(--color-text-primary)]',
          )}
        >
          <MessageContent text={message.content} />
        </div>
        <MessageMeta>
          {time}
          {isDemo && <span>· Demo reply (scripted, not AI)</span>}
        </MessageMeta>
      </div>
    </div>
  );
}

/** Memoized: history items don't re-render while the user types or a reply is pending. */
export const ChatMessageItem = memo(ChatMessageItemImpl);
