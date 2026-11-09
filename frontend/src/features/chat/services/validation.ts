import { ApiError } from '@/lib/api/errors';
import type { FounderId } from '@/types';
import type { ChatMessage, SendMessageResult } from '../types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

/**
 * Validates one message from the service and checks it belongs to the
 * expected project/founder scope, so a misrouted response can never be cached
 * under (and shown to) the wrong founder.
 */
export function parseChatMessage(raw: unknown, projectId: string, founderId: FounderId): ChatMessage {
  if (
    !isRecord(raw) ||
    typeof raw.id !== 'string' ||
    (raw.role !== 'user' && raw.role !== 'assistant') ||
    typeof raw.content !== 'string' ||
    typeof raw.createdAt !== 'string' ||
    Number.isNaN(Date.parse(raw.createdAt))
  ) {
    throw new ApiError('invalid_response');
  }
  if (raw.projectId !== projectId || raw.founderId !== founderId) {
    throw new ApiError('invalid_response', { message: 'Message scope mismatch' });
  }
  return {
    id: raw.id,
    projectId,
    founderId,
    role: raw.role,
    content: raw.content,
    createdAt: raw.createdAt,
    status: 'sent',
  };
}

export function parseMessageList(raw: unknown, projectId: string, founderId: FounderId): ChatMessage[] {
  if (!Array.isArray(raw)) throw new ApiError('invalid_response');
  return raw.map((item) => parseChatMessage(item, projectId, founderId));
}

export function parseSendResult(raw: unknown, projectId: string, founderId: FounderId): SendMessageResult {
  if (!isRecord(raw)) throw new ApiError('invalid_response');
  const userMessage = parseChatMessage(raw.userMessage, projectId, founderId);
  const assistantMessage = parseChatMessage(raw.assistantMessage, projectId, founderId);
  if (userMessage.role !== 'user' || assistantMessage.role !== 'assistant') {
    throw new ApiError('invalid_response');
  }
  if (assistantMessage.content.trim() === '') throw new ApiError('empty_response');
  return { userMessage, assistantMessage };
}
