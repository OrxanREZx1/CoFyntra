/**
 * ⚠️ DEMO-ONLY chat adapter. NOT a backend, NOT secure, NOT synced.
 *
 * - Messages persist in this browser's localStorage, namespaced by project and
 *   founder. Anyone with access to this browser can read every key; the
 *   founder separation here is a UI convention, not access control.
 * - Nothing leaves the browser: no cross-device sync, no server persistence.
 * - Assistant replies are scripted templates (see demoReplies.ts), not an LLM.
 *
 * Testing aid: include "#fail" in a message to make its first send attempt
 * fail with a simulated network error, so retry behaviour can be exercised.
 */
import { ApiError } from '@/lib/api/errors';
import { CHAT_MESSAGES_BY_FOUNDER } from '@/mocks';
import type { FounderId } from '@/types';
import type { ChatMessage, ChatService } from '../types';
import { getDemoReply } from './demoReplies';
import { parseMessageList } from './validation';

const STORAGE_PREFIX = 'confyntra.demo.chat.v1';

/**
 * Small simulated round-trip so loading states are visible in the demo.
 * This stands in for network latency only — it is not "thinking time".
 */
const DEMO_LATENCY_MS = 400;

/** Fallback when localStorage is unavailable (private mode, blocked storage). */
const memoryStore = new Map<string, ChatMessage[]>();

/** clientMessageIds whose simulated failure has already fired. */
const simulatedFailures = new Set<string>();

function storageKey(projectId: string, founderId: FounderId) {
  return `${STORAGE_PREFIX}:${projectId}:${founderId}`;
}

function seedMessages(projectId: string, founderId: FounderId): ChatMessage[] {
  // Defensive filter: a founder's seed must only contain that founder's messages.
  return (CHAT_MESSAGES_BY_FOUNDER[founderId] ?? []).filter(
    (m) => m.founderId === founderId && m.projectId === projectId,
  );
}

function load(projectId: string, founderId: FounderId): ChatMessage[] {
  const key = storageKey(projectId, founderId);
  try {
    const raw = window.localStorage.getItem(key);
    if (raw !== null) return parseMessageList(JSON.parse(raw), projectId, founderId);
  } catch {
    // Unavailable or corrupt storage — fall through to memory/seed.
  }
  return memoryStore.get(key) ?? seedMessages(projectId, founderId);
}

function save(projectId: string, founderId: FounderId, messages: ChatMessage[]) {
  const key = storageKey(projectId, founderId);
  memoryStore.set(key, messages);
  try {
    window.localStorage.setItem(key, JSON.stringify(messages));
  } catch {
    // Quota exceeded or storage blocked — the in-memory copy still works for this session.
  }
}

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export const demoChatAdapter: ChatService = {
  mode: 'demo',

  async getMessages(projectId, founderId) {
    return load(projectId, founderId);
  },

  async sendMessage(projectId, founderId, { content, clientMessageId }) {
    const trimmed = content.trim();
    if (!trimmed) throw new ApiError('unknown', { message: 'Message is empty' });

    await wait(DEMO_LATENCY_MS);

    if (typeof navigator !== 'undefined' && !navigator.onLine) throw new ApiError('network');

    if (trimmed.includes('#fail') && !simulatedFailures.has(clientMessageId)) {
      simulatedFailures.add(clientMessageId);
      throw new ApiError('network');
    }

    const messages = load(projectId, founderId);

    // Idempotency: a retry of an already-stored send returns the original pair.
    const existingIndex = messages.findIndex((m) => m.id === clientMessageId);
    if (existingIndex !== -1) {
      const reply = messages[existingIndex + 1];
      if (reply?.role === 'assistant') {
        return { userMessage: messages[existingIndex], assistantMessage: reply };
      }
    }

    const now = Date.now();
    const userMessage: ChatMessage = {
      id: clientMessageId,
      projectId,
      founderId,
      role: 'user',
      content: trimmed,
      createdAt: new Date(now).toISOString(),
      status: 'sent',
    };
    const assistantMessage: ChatMessage = {
      id: `${clientMessageId}-reply`,
      projectId,
      founderId,
      role: 'assistant',
      content: getDemoReply(trimmed),
      createdAt: new Date(now + 1).toISOString(),
      status: 'sent',
    };

    const base = existingIndex === -1 ? messages : messages.filter((m) => m.id !== clientMessageId);
    save(projectId, founderId, [...base, userMessage, assistantMessage]);
    return { userMessage, assistantMessage };
  },
};
