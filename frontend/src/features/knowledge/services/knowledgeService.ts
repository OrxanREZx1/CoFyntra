import { ApiError } from '@/lib/api/errors';
import { httpClient } from '@/lib/api/httpClient';
import { BACKEND_MODE } from '@/lib/config';
import { demoLatency, isDemoScenario } from '@/lib/demoScenario';
import { MOCK_KNOWLEDGE_CONFLICTS, MOCK_KNOWLEDGE_ITEMS } from '@/mocks';
import type { SharedKnowledgeSnapshot } from '@/types';
import { parseKnowledgeSnapshot } from './validation';

export interface KnowledgeService {
  readonly mode: 'demo' | 'api';
  /** Project-level shared (L2) knowledge — identical for both founders. */
  getSharedKnowledge(projectId: string): Promise<SharedKnowledgeSnapshot>;
}

/**
 * ⚠️ DEMO adapter: serves the predefined mock findings and conflict
 * relationships from src/mocks. No extraction, no AI, no persistence.
 */
export const demoKnowledgeAdapter: KnowledgeService = {
  mode: 'demo',
  async getSharedKnowledge(projectId) {
    await demoLatency();
    if (isDemoScenario('knowledge-error')) throw new ApiError('unavailable');
    const empty = isDemoScenario('knowledge-empty');
    // Run mocks through the same validation as real responses.
    return parseKnowledgeSnapshot(
      { items: empty ? [] : MOCK_KNOWLEDGE_ITEMS, conflicts: empty ? [] : MOCK_KNOWLEDGE_CONFLICTS },
      projectId,
    );
  },
};

/**
 * HTTP adapter. ⚠️ PLACEHOLDER ENDPOINT — no backend contract exists yet.
 * The backend must authorize the session for this project and return only
 * shareable L2 material (never raw private transcripts).
 */
const httpKnowledgeService: KnowledgeService = {
  mode: 'api',
  async getSharedKnowledge(projectId) {
    const { data } = await httpClient.get<unknown>(`/projects/${encodeURIComponent(projectId)}/knowledge`);
    return parseKnowledgeSnapshot(data, projectId);
  },
};

export const knowledgeService: KnowledgeService =
  BACKEND_MODE === 'api' ? httpKnowledgeService : demoKnowledgeAdapter;
