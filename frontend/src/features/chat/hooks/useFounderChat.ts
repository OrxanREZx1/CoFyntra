'use client';

import { useCallback, useMemo } from 'react';
import useSWR, { useSWRConfig } from 'swr';
import { toast } from 'sonner';
import { normalizeError } from '@/lib/api/errors';
import type { FounderId } from '@/types';
import { chatService } from '../services/chatService';
import type { ChatMessage } from '../types';

/** A user message not yet confirmed by the chat service. */
export type PendingChatMessage = ChatMessage & {
  status: 'sending' | 'failed';
  error?: string;
};

export type DisplayChatMessage = ChatMessage | PendingChatMessage;

/**
 * SWR cache keys. Every key includes project AND founder, so one founder's
 * private messages can never be read from, or written to, another's entry.
 */
export const chatKeys = {
  messages: (projectId: string, founderId: FounderId) => ['chat', 'messages', projectId, founderId] as const,
  pending: (projectId: string, founderId: FounderId) => ['chat', 'pending', projectId, founderId] as const,
};

const EMPTY: PendingChatMessage[] = [];

function createClientMessageId(): string {
  // randomUUID is unavailable in insecure contexts (e.g. testing over a LAN IP).
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `client-${crypto.randomUUID()}`;
  }
  return `client-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** Appends messages, skipping any whose ID is already present. */
function mergeMessages(existing: ChatMessage[], incoming: ChatMessage[]): ChatMessage[] {
  const ids = new Set(existing.map((m) => m.id));
  return [...existing, ...incoming.filter((m) => !ids.has(m.id))];
}

export function useFounderChat(projectId: string, founderId: FounderId) {
  const { mutate } = useSWRConfig();

  // Confirmed history. Private chats are never polled or revalidated on focus.
  const history = useSWR(chatKeys.messages(projectId, founderId), () => chatService.getMessages(projectId, founderId), {
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
    shouldRetryOnError: false, // the user retries explicitly from the error state
  });

  // Unconfirmed sends, kept in the SWR cache (not component state) so an
  // in-flight or failed message survives switching founders and back.
  const { data: pending = EMPTY } = useSWR<PendingChatMessage[]>(chatKeys.pending(projectId, founderId), null);

  const deliver = useCallback(
    async (message: PendingChatMessage) => {
      // Keys are captured per call, so a reply that lands after the user has
      // switched founders is still written to the correct founder's cache.
      const msgKey = chatKeys.messages(message.projectId, message.founderId);
      const pendKey = chatKeys.pending(message.projectId, message.founderId);
      const updatePending = (fn: (prev: PendingChatMessage[]) => PendingChatMessage[]) =>
        mutate(pendKey, (prev: PendingChatMessage[] = []) => fn(prev), { revalidate: false });

      updatePending((prev) =>
        prev.map((p) => (p.id === message.id ? { ...p, status: 'sending', error: undefined } : p)),
      );

      try {
        const { userMessage, assistantMessage } = await chatService.sendMessage(
          message.projectId,
          message.founderId,
          { content: message.content, clientMessageId: message.id },
        );
        // Both updates run in the same tick, so React renders them together
        // and the optimistic copy never appears alongside the confirmed one.
        mutate(msgKey, (prev: ChatMessage[] = []) => mergeMessages(prev, [userMessage, assistantMessage]), {
          revalidate: false,
        });
        updatePending((prev) => prev.filter((p) => p.id !== message.id));
      } catch (err) {
        const error = normalizeError(err);
        updatePending((prev) =>
          prev.map((p) => (p.id === message.id ? { ...p, status: 'failed', error: error.userMessage } : p)),
        );
        // Toast copy never includes message content.
        toast.error('Message not sent', { id: `chat-send-${message.id}`, description: error.userMessage });
      }
    },
    [mutate],
  );

  const send = useCallback(
    (content: string): boolean => {
      const trimmed = content.trim();
      if (!trimmed) return false;

      const message: PendingChatMessage = {
        id: createClientMessageId(),
        projectId,
        founderId,
        role: 'user',
        content: trimmed,
        createdAt: new Date().toISOString(),
        status: 'sending',
      };
      mutate(chatKeys.pending(projectId, founderId), (prev: PendingChatMessage[] = []) => [...prev, message], {
        revalidate: false,
      });
      void deliver(message);
      return true;
    },
    [projectId, founderId, mutate, deliver],
  );

  /** Re-sends a failed message with the same client ID (no duplicate). */
  const retry = useCallback(
    (messageId: string) => {
      const message = pending.find((p) => p.id === messageId && p.status === 'failed');
      if (message) void deliver(message);
    },
    [pending, deliver],
  );

  /** Removes a failed message and returns its content (e.g. to edit it). */
  const takeFailed = useCallback(
    (messageId: string): string | undefined => {
      const message = pending.find((p) => p.id === messageId && p.status === 'failed');
      if (!message) return undefined;
      toast.dismiss(`chat-send-${messageId}`);
      mutate(
        chatKeys.pending(message.projectId, message.founderId),
        (prev: PendingChatMessage[] = []) => prev.filter((p) => p.id !== messageId),
        { revalidate: false },
      );
      return message.content;
    },
    [pending, mutate],
  );

  const confirmed = history.data;
  const messages = useMemo<DisplayChatMessage[]>(() => {
    const list = confirmed ?? [];
    const ids = new Set(list.map((m) => m.id));
    return [...list, ...pending.filter((p) => !ids.has(p.id))];
  }, [confirmed, pending]);

  return {
    mode: chatService.mode,
    messages,
    isLoadingHistory: confirmed === undefined && !history.error,
    historyError: history.error ? normalizeError(history.error) : undefined,
    reloadHistory: () => history.mutate(),
    isAwaitingReply: pending.some((p) => p.status === 'sending'),
    send,
    retry,
    takeFailed,
  };
}
