import { httpClient } from '@/lib/api/httpClient';
import { BACKEND_MODE } from '@/lib/config';
import type { ChatService } from '../types';
import { demoChatAdapter } from './demoChatAdapter';
import { parseMessageList, parseSendResult } from './validation';

/**
 * HTTP implementation of the chat service.
 *
 * ⚠️ PLACEHOLDER ENDPOINTS: there is no agreed backend contract yet
 * (docs/sync_contract.md does not exist). These paths are assumptions kept in
 * one place so they can be swapped when the contract is finalized. This
 * adapter is inactive unless NEXT_PUBLIC_BACKEND_MODE=api.
 *
 * Note that `founderId` is intentionally NOT sent: the backend must identify
 * the founder from the authenticated session. It is only used to validate
 * that responses belong to the scope we cache them under.
 */
const endpoints = {
  messages: (projectId: string) => `/projects/${encodeURIComponent(projectId)}/chat/messages`,
};

const httpChatService: ChatService = {
  mode: 'api',

  async getMessages(projectId, founderId) {
    const { data } = await httpClient.get<unknown>(endpoints.messages(projectId));
    return parseMessageList(data, projectId, founderId);
  },

  async sendMessage(projectId, founderId, request) {
    const { data } = await httpClient.post<unknown>(endpoints.messages(projectId), request, {
      // AI replies can be slow; allow longer than the default timeout.
      timeout: 60_000,
      headers: { 'Idempotency-Key': request.clientMessageId },
    });
    return parseSendResult(data, projectId, founderId);
  },
};

/** The active chat service. Defaults to the demo adapter. */
export const chatService: ChatService = BACKEND_MODE === 'api' ? httpChatService : demoChatAdapter;
