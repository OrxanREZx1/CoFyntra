import type { ChatMessage, FounderId, SendMessageRequest, SendMessageResult } from '@/types';

export type { ChatMessage, SendMessageRequest, SendMessageResult };

/**
 * Frontend abstraction over the private (L1) chat backend.
 *
 * `founderId` scopes client-side state and caching only. A real backend must
 * derive the founder from the authenticated session and must never let a
 * client-supplied founder ID grant access to another founder's messages.
 */
export interface ChatService {
  /** 'demo' = in-browser mock adapter; 'api' = real backend. */
  readonly mode: 'demo' | 'api';

  getMessages(projectId: string, founderId: FounderId): Promise<ChatMessage[]>;

  /**
   * Must be idempotent on `request.clientMessageId`: retrying a send with the
   * same ID must not create a second user message.
   */
  sendMessage(
    projectId: string,
    founderId: FounderId,
    request: SendMessageRequest,
  ): Promise<SendMessageResult>;
}
