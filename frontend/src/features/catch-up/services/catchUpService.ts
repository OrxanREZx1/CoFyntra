import { ApiError } from '@/lib/api/errors';
import { httpClient } from '@/lib/api/httpClient';
import { BACKEND_MODE } from '@/lib/config';
import { isDemoScenario } from '@/lib/demoScenario';
import { FOUNDERS } from '@/mocks';
import type { CatchUpSummary, FounderId } from '@/types';
import { demoKnowledgeAdapter } from '@/features/knowledge/services/knowledgeService';
import { demoProposalAdapter } from '@/features/proposals/services/demoProposalAdapter';
import { decisionFor } from '@/features/proposals/lib/proposalMachine';
import { buildDemoCatchUp } from '../lib/catchUpLogic';

export interface CatchUpService {
  readonly mode: 'demo' | 'api';
  /**
   * Catch-up for the requesting founder about their co-founder's shared
   * findings. `founderId` scopes client caching; a real backend must take the
   * founder from the authenticated session.
   */
  getCatchUp(projectId: string, founderId: FounderId): Promise<CatchUpSummary>;
}

/** MVP has exactly two founders. */
export function getCoFounderId(founderId: FounderId): FounderId {
  return founderId === 'founder-a' ? 'founder-b' : 'founder-a';
}

/**
 * ⚠️ DEMO adapter: builds a deterministic summary from the demo shared
 * knowledge records only (never from private chats). Not AI-generated.
 */
const demoCatchUpAdapter: CatchUpService = {
  mode: 'demo',
  async getCatchUp(projectId, founderId) {
    const snapshot = await demoKnowledgeAdapter.getSharedKnowledge(projectId);
    if (isDemoScenario('catchup-error')) throw new ApiError('unavailable');
    const coFounderId = getCoFounderId(founderId);
    // Proposals awaiting THIS founder's decision on the current version.
    const proposals = await demoProposalAdapter.listProposals(projectId);
    const pendingProposals = proposals
      .filter((p) => p.status === 'PENDING_APPROVAL' && !decisionFor(p, founderId))
      .map(({ id, title }) => ({ id, title }));

    return buildDemoCatchUp({
      snapshot,
      requestingFounderId: founderId,
      coFounderId,
      coFounderName: FOUNDERS[coFounderId]?.name ?? 'Your co-founder',
      pendingProposals,
    });
  },
};

/** ⚠️ PLACEHOLDER ENDPOINT — no backend contract exists yet. */
const httpCatchUpService: CatchUpService = {
  mode: 'api',
  async getCatchUp(projectId) {
    const { data } = await httpClient.get<CatchUpSummary>(
      `/projects/${encodeURIComponent(projectId)}/catch-up`,
    );
    if (!data || typeof data !== 'object' || !Array.isArray(data.findings)) {
      throw new ApiError('invalid_response');
    }
    return data;
  },
};

export const catchUpService: CatchUpService = BACKEND_MODE === 'api' ? httpCatchUpService : demoCatchUpAdapter;
