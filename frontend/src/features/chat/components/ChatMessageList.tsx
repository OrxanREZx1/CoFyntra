'use client';

import { memo, useCallback, useLayoutEffect, useRef, useState } from 'react';
import { ArrowDown } from 'lucide-react';
import type { FounderId } from '@/types';
import type { DisplayChatMessage } from '../hooks/useFounderChat';
import { ChatLoadingIndicator } from './ChatLoadingIndicator';
import { ChatMessageItem } from './ChatMessageItem';

/** Within this many px of the bottom counts as "following" the conversation. */
const NEAR_BOTTOM_PX = 120;

interface ChatMessageListProps {
  messages: DisplayChatMessage[];
  founderId: FounderId;
  isDemo: boolean;
  isAwaitingReply: boolean;
  onRetry: (messageId: string) => void;
  onEdit: (messageId: string) => void;
}

function ChatMessageListImpl({
  messages,
  founderId,
  isDemo,
  isAwaitingReply,
  onRetry,
  onEdit,
}: ChatMessageListProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);
  const hasInitialScrollRef = useRef(false);
  const [showJumpButton, setShowJumpButton] = useState(false);

  const scrollToBottom = useCallback((behavior: ScrollBehavior) => {
    const el = containerRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior });
  }, []);

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
    isNearBottomRef.current = nearBottom;
    setShowJumpButton(!nearBottom);
  }, []);

  const last = messages[messages.length - 1];
  const lastId = last?.id;
  const userJustSent = last?.role === 'user' && last.status === 'sending';

  useLayoutEffect(() => {
    if (!hasInitialScrollRef.current) {
      // Open the conversation at the latest message.
      if (lastId) {
        scrollToBottom('auto');
        hasInitialScrollRef.current = true;
      }
      return;
    }
    // Always reveal the user's own new message; otherwise only follow new
    // content when they are already at the bottom (not reading history).
    if (userJustSent || isNearBottomRef.current) scrollToBottom('smooth');
  }, [lastId, userJustSent, isAwaitingReply, scrollToBottom]);

  return (
    <div className="relative flex-1 min-h-0">
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="h-full overflow-y-auto overscroll-contain px-4 py-5"
        role="log"
        aria-label="Conversation"
        aria-relevant="additions"
      >
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((message) => (
            <ChatMessageItem
              key={message.id}
              message={message}
              founderId={founderId}
              isDemo={isDemo}
              onRetry={onRetry}
              onEdit={onEdit}
            />
          ))}
          {isAwaitingReply && (
            <ChatLoadingIndicator label={isDemo ? 'Demo assistant is replying' : 'Assistant is replying'} />
          )}
        </div>
      </div>

      {showJumpButton && (
        <button
          type="button"
          onClick={() => scrollToBottom('smooth')}
          className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium shadow-lg text-[var(--color-text-secondary)] bg-[var(--color-bg-raised)] border border-[var(--color-border)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-border-active)] transition-colors"
        >
          <ArrowDown className="w-3 h-3" /> Latest
        </button>
      )}
    </div>
  );
}

/** Memoized so typing in the composer doesn't re-render the conversation. */
export const ChatMessageList = memo(ChatMessageListImpl);
