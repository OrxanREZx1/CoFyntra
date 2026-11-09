/**
 * ⚠️ DEMO-ONLY decision adapter. NOT a backend, NOT a database transaction,
 * NOT secure, NOT synchronized across browsers or devices.
 *
 * - All state transitions come from the pure state machine (proposalMachine.ts);
 *   this adapter only loads, applies one transition, and saves.
 * - State lives in this browser's localStorage, so tabs of the SAME browser
 *   share it. A real two-browser demo requires the backend integration.
 * - "Founder identity" is the demo founder switcher — not authentication.
 */
import { ApiError } from '@/lib/api/errors';
import { demoLatency, isDemoScenario } from '@/lib/demoScenario';
import { MOCK_COMMITTED_STATE, MOCK_DECISION_PROPOSALS, MOCK_KNOWLEDGE_ITEMS } from '@/mocks';
import type { ProposalContent } from '@/types';
import {
  approveProposal,
  createProposal,
  emptyStore,
  ProposalRuleError,
  requestChanges,
  reviseProposal,
  submitDraft,
  type DecisionStore,
} from '../lib/proposalMachine';
import type { ProposalMutationResult, ProposalService } from './proposalService';
import { parseCommittedState, parseProposalList } from './validation';

export const DEMO_DECISIONS_STORAGE_PREFIX = 'confyntra.demo.decisions.v1';
const storageKey = (projectId: string) => `${DEMO_DECISIONS_STORAGE_PREFIX}:${projectId}`;

/** Fallback when localStorage is unavailable. */
const memoryStore = new Map<string, DecisionStore>();

function seed(projectId: string): DecisionStore {
  return {
    projectId,
    proposals: MOCK_DECISION_PROPOSALS.filter((p) => p.projectId === projectId),
    committed: MOCK_COMMITTED_STATE.projectId === projectId ? MOCK_COMMITTED_STATE : emptyStore(projectId).committed,
  };
}

function load(projectId: string): DecisionStore {
  try {
    const raw = window.localStorage.getItem(storageKey(projectId));
    if (raw !== null) {
      const parsed = JSON.parse(raw) as Partial<DecisionStore>;
      return {
        projectId,
        proposals: parseProposalList(parsed.proposals, projectId),
        committed: parseCommittedState(parsed.committed, projectId),
      };
    }
  } catch {
    // Missing, blocked or corrupt storage — fall back below.
  }
  return memoryStore.get(projectId) ?? seed(projectId);
}

function save(store: DecisionStore) {
  memoryStore.set(store.projectId, store);
  try {
    window.localStorage.setItem(storageKey(store.projectId), JSON.stringify(store));
  } catch {
    // Storage unavailable — in-memory copy still works for this tab.
  }
}

/** Maps state-machine rule violations onto the shared ApiError kinds. */
function toApiError(error: unknown): unknown {
  if (!(error instanceof ProposalRuleError)) return error;
  const kind = error.code === 'STALE_VERSION' ? 'conflict' : error.code === 'NOT_FOUND' ? 'not_found' : 'invalid_request';
  return new ApiError(kind, { detail: error.message, cause: error });
}

/** Rejects references to findings that don't exist in the project's shared knowledge. */
function assertKnownKnowledge(projectId: string, content: ProposalContent) {
  const known = new Set(MOCK_KNOWLEDGE_ITEMS.filter((i) => i.projectId === projectId).map((i) => i.id));
  const unknown = [...content.supportingKnowledgeIds, ...content.conflictingKnowledgeIds].filter((id) => !known.has(id));
  if (unknown.length) throw new ProposalRuleError('INVALID_INPUT', `Unknown shared finding: ${unknown.join(', ')}.`);
}

/** Load → apply exactly one transition → save. */
async function transact(
  projectId: string,
  apply: (store: DecisionStore, now: string) => { store: DecisionStore } & ProposalMutationResult,
): Promise<ProposalMutationResult> {
  await demoLatency();
  try {
    const before = load(projectId);
    const { store, ...result } = apply(before, new Date().toISOString());
    if (store !== before) save(store);
    return { ...result, committedState: store.committed };
  } catch (error) {
    throw toApiError(error);
  }
}

export const demoProposalAdapter: ProposalService = {
  mode: 'demo',

  async listProposals(projectId) {
    await demoLatency();
    if (isDemoScenario('proposals-error')) throw new ApiError('unavailable');
    return load(projectId).proposals;
  },

  async getCommittedState(projectId) {
    await demoLatency();
    if (isDemoScenario('proposals-error')) throw new ApiError('unavailable');
    return load(projectId).committed;
  },

  createProposal(projectId, founderId, content, { submit }) {
    return transact(projectId, (store, now) => {
      assertKnownKnowledge(projectId, content);
      return createProposal(store, { actor: founderId, content, submit, now });
    });
  },

  submitDraft(projectId, founderId, proposalId, version) {
    return transact(projectId, (store, now) => submitDraft(store, { actor: founderId, proposalId, version, now }));
  },

  reviseProposal(projectId, founderId, proposalId, version, content, changeNote) {
    return transact(projectId, (store, now) => {
      assertKnownKnowledge(projectId, content);
      return reviseProposal(store, { actor: founderId, proposalId, version, content, changeNote, now });
    });
  },

  approve(projectId, founderId, proposalId, version) {
    return transact(projectId, (store, now) => approveProposal(store, { actor: founderId, proposalId, version, now }));
  },

  requestChanges(projectId, founderId, proposalId, version, reason) {
    return transact(projectId, (store, now) =>
      requestChanges(store, { actor: founderId, proposalId, version, reason, now }),
    );
  },
};
