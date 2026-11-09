import { httpClient } from '@/lib/api/httpClient';
import { BACKEND_MODE } from '@/lib/config';
import type { CommittedStartupState, DecisionProposal, FounderId, ProposalContent } from '@/types';
import { demoProposalAdapter } from './demoProposalAdapter';
import { parseCommittedState, parseProposal, parseProposalList } from './validation';

export type ProposalOutcome =
  | 'CREATED'
  | 'SUBMITTED'
  | 'DRAFT_UPDATED'
  | 'REVISED'
  | 'APPROVED'
  | 'ALREADY_APPROVED'
  | 'COMMITTED'
  | 'ALREADY_COMMITTED'
  | 'CHANGES_REQUESTED'
  | 'ALREADY_REQUESTED';

export interface ProposalMutationResult {
  proposal: DecisionProposal;
  outcome: ProposalOutcome;
  /** Present only when the service confirmed a commit (or returned current state). */
  committedState?: CommittedStartupState;
}

/**
 * Single interface for all L3 operations.
 *
 * `founderId` is the DEMO identity used by the demo adapter. A real backend
 * must take the acting founder from the authenticated session and must not
 * trust a client-supplied identity. `version` is always sent so the server can
 * reject actions against an outdated version (HTTP 409).
 */
export interface ProposalService {
  readonly mode: 'demo' | 'api';
  listProposals(projectId: string): Promise<DecisionProposal[]>;
  getCommittedState(projectId: string): Promise<CommittedStartupState>;
  createProposal(
    projectId: string,
    founderId: FounderId,
    content: ProposalContent,
    options: { submit: boolean },
  ): Promise<ProposalMutationResult>;
  submitDraft(projectId: string, founderId: FounderId, proposalId: string, version: number): Promise<ProposalMutationResult>;
  reviseProposal(
    projectId: string,
    founderId: FounderId,
    proposalId: string,
    version: number,
    content: ProposalContent,
    changeNote: string | null,
  ): Promise<ProposalMutationResult>;
  approve(projectId: string, founderId: FounderId, proposalId: string, version: number): Promise<ProposalMutationResult>;
  requestChanges(
    projectId: string,
    founderId: FounderId,
    proposalId: string,
    version: number,
    reason: string,
  ): Promise<ProposalMutationResult>;
}

/**
 * HTTP adapter. ⚠️ PLACEHOLDER ENDPOINTS AND RESPONSE SHAPES — there is no
 * backend contract yet (docs/sync_contract.md is missing). Inactive unless
 * NEXT_PUBLIC_BACKEND_MODE=api. Expected server behaviour is documented in
 * the Phase 4 report: atomic commit, 409 on stale version, idempotent approvals.
 */
const base = (projectId: string) => `/projects/${encodeURIComponent(projectId)}`;
const proposalPath = (projectId: string, id: string) => `${base(projectId)}/proposals/${encodeURIComponent(id)}`;

function parseMutation(data: unknown, projectId: string): ProposalMutationResult {
  const body = (data ?? {}) as { proposal?: unknown; outcome?: unknown; committedState?: unknown };
  return {
    proposal: parseProposal(body.proposal, projectId),
    outcome: (typeof body.outcome === 'string' ? body.outcome : 'APPROVED') as ProposalOutcome,
    committedState: body.committedState ? parseCommittedState(body.committedState, projectId) : undefined,
  };
}

const httpProposalService: ProposalService = {
  mode: 'api',
  async listProposals(projectId) {
    const { data } = await httpClient.get<unknown>(`${base(projectId)}/proposals`);
    return parseProposalList(data, projectId);
  },
  async getCommittedState(projectId) {
    const { data } = await httpClient.get<unknown>(`${base(projectId)}/committed-state`);
    return parseCommittedState(data, projectId);
  },
  async createProposal(projectId, _founderId, content, { submit }) {
    const { data } = await httpClient.post<unknown>(`${base(projectId)}/proposals`, { ...content, submit });
    return parseMutation(data, projectId);
  },
  async submitDraft(projectId, _founderId, proposalId, version) {
    const { data } = await httpClient.post<unknown>(`${proposalPath(projectId, proposalId)}/submit`, { version });
    return parseMutation(data, projectId);
  },
  async reviseProposal(projectId, _founderId, proposalId, version, content, changeNote) {
    const { data } = await httpClient.post<unknown>(`${proposalPath(projectId, proposalId)}/revisions`, {
      expectedVersion: version,
      ...content,
      changeNote,
    });
    return parseMutation(data, projectId);
  },
  async approve(projectId, _founderId, proposalId, version) {
    const { data } = await httpClient.post<unknown>(`${proposalPath(projectId, proposalId)}/approvals`, { version });
    return parseMutation(data, projectId);
  },
  async requestChanges(projectId, _founderId, proposalId, version, reason) {
    const { data } = await httpClient.post<unknown>(`${proposalPath(projectId, proposalId)}/change-requests`, {
      version,
      reason,
    });
    return parseMutation(data, projectId);
  },
};

export const proposalService: ProposalService = BACKEND_MODE === 'api' ? httpProposalService : demoProposalAdapter;
