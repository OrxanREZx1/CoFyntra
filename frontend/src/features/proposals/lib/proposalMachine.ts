/**
 * Decision proposal state machine — pure, deterministic, no I/O.
 *
 * Used by the DEMO adapter to mirror the rules the real backend must enforce
 * authoritatively. The core rule: a proposal becomes the official direction
 * only when BOTH founders approve the EXACT SAME version. React components
 * never apply these transitions themselves.
 *
 * Type-only imports so it can run under Node's test runner unchanged.
 */
import type {
  CommittedStartupState,
  DecisionProposal,
  FounderApproval,
  FounderId,
  KnowledgeConflict,
  ProposalConflict,
  ProposalContent,
} from '@/types';

export const FOUNDER_IDS: readonly FounderId[] = ['founder-a', 'founder-b'];

export const LIMITS = { title: 120, direction: 1000, nextAction: 600, reason: 500, changeNote: 300 } as const;

export type ProposalRuleCode = 'NOT_FOUND' | 'STALE_VERSION' | 'NOT_ALLOWED' | 'INVALID_INPUT';

export class ProposalRuleError extends Error {
  readonly code: ProposalRuleCode;
  constructor(code: ProposalRuleCode, message: string) {
    super(message);
    this.name = 'ProposalRuleError';
    this.code = code;
  }
}

/** Everything the demo persists for one project. */
export interface DecisionStore {
  projectId: string;
  proposals: DecisionProposal[];
  committed: CommittedStartupState;
}

export function emptyStore(projectId: string): DecisionStore {
  return { projectId, proposals: [], committed: { projectId, revision: 0, decision: null } };
}

// ─── Helpers ────────────────────────────────────────────────

function fail(code: ProposalRuleCode, message: string): never {
  throw new ProposalRuleError(code, message);
}

function assertFounder(actor: FounderId) {
  if (!FOUNDER_IDS.includes(actor)) fail('NOT_ALLOWED', 'Only the two project founders can act on proposals.');
}

function findProposal(store: DecisionStore, proposalId: string): DecisionProposal {
  return store.proposals.find((p) => p.id === proposalId) ?? fail('NOT_FOUND', 'This proposal no longer exists.');
}

function assertCurrentVersion(proposal: DecisionProposal, version: number) {
  if (proposal.version !== version) {
    fail(
      'STALE_VERSION',
      `This proposal was updated to version ${proposal.version}. Review the latest version before deciding.`,
    );
  }
}

function replaceProposal(store: DecisionStore, proposal: DecisionProposal): DecisionStore {
  return { ...store, proposals: store.proposals.map((p) => (p.id === proposal.id ? proposal : p)) };
}

function uniqueIds(ids: readonly string[]): string[] {
  return [...new Set(ids.map((id) => id.trim()).filter(Boolean))];
}

function requiredText(value: string, label: string, max: number): string {
  const text = value.trim();
  if (!text) fail('INVALID_INPUT', `${label} is required.`);
  if (text.length > max) fail('INVALID_INPUT', `${label} must be ${max} characters or fewer.`);
  return text;
}

export function normalizeContent(input: ProposalContent): ProposalContent {
  const supporting = uniqueIds(input.supportingKnowledgeIds);
  const conflicting = uniqueIds(input.conflictingKnowledgeIds);
  if (supporting.some((id) => conflicting.includes(id))) {
    fail('INVALID_INPUT', 'A finding cannot be both supporting and conflicting.');
  }
  return {
    title: requiredText(input.title, 'Title', LIMITS.title),
    direction: requiredText(input.direction, 'Proposed direction', LIMITS.direction),
    nextAction: requiredText(input.nextAction, 'Next action', LIMITS.nextAction),
    supportingKnowledgeIds: supporting,
    conflictingKnowledgeIds: conflicting,
  };
}

function sameContent(a: ProposalContent, b: ProposalContent): boolean {
  const ids = (x: string[]) => [...x].sort().join('|');
  return (
    a.title === b.title &&
    a.direction === b.direction &&
    a.nextAction === b.nextAction &&
    ids(a.supportingKnowledgeIds) === ids(b.supportingKnowledgeIds) &&
    ids(a.conflictingKnowledgeIds) === ids(b.conflictingKnowledgeIds)
  );
}

function contentOf(p: ProposalContent): ProposalContent {
  return {
    title: p.title,
    direction: p.direction,
    nextAction: p.nextAction,
    supportingKnowledgeIds: [...p.supportingKnowledgeIds],
    conflictingKnowledgeIds: [...p.conflictingKnowledgeIds],
  };
}

// ─── Queries ────────────────────────────────────────────────

/** The founder's decision on the proposal's CURRENT version, if any. */
export function decisionFor(proposal: DecisionProposal, founderId: FounderId): FounderApproval | undefined {
  return proposal.approvals.find((a) => a.founderId === founderId && a.version === proposal.version);
}

/** True only when both founders approved the current version. */
export function isApprovedByBoth(proposal: DecisionProposal): boolean {
  return FOUNDER_IDS.every((f) => decisionFor(proposal, f)?.decision === 'APPROVED');
}

export function nextProposalId(store: DecisionStore): string {
  const max = store.proposals.reduce((m, p) => Math.max(m, Number(p.id.replace(/\D/g, '')) || 0), 0);
  return `P-${String(max + 1).padStart(3, '0')}`;
}

/**
 * Pairs each conflicting finding with the supporting finding it disagrees
 * with, using ONLY supplied conflict relationships.
 */
export function getProposalConflicts(
  proposal: Pick<ProposalContent, 'supportingKnowledgeIds' | 'conflictingKnowledgeIds'>,
  knowledgeConflicts: readonly KnowledgeConflict[],
): ProposalConflict[] {
  return proposal.conflictingKnowledgeIds.map((conflictingItemId) => {
    const match = knowledgeConflicts.find(
      (c) =>
        c.itemIds.includes(conflictingItemId) &&
        c.itemIds.some((id) => id !== conflictingItemId && proposal.supportingKnowledgeIds.includes(id)),
    );
    return {
      conflictingItemId,
      supportingItemId: match ? (match.itemIds.find((id) => id !== conflictingItemId) ?? null) : null,
      knowledgeConflictId: match?.id ?? null,
    };
  });
}

// ─── Transitions ────────────────────────────────────────────

export interface TransitionResult<Outcome extends string> {
  store: DecisionStore;
  proposal: DecisionProposal;
  outcome: Outcome;
}

export function createProposal(
  store: DecisionStore,
  args: { actor: FounderId; content: ProposalContent; submit: boolean; now: string },
): TransitionResult<'CREATED'> {
  assertFounder(args.actor);
  const proposal: DecisionProposal = {
    ...normalizeContent(args.content),
    id: nextProposalId(store),
    projectId: store.projectId,
    createdByFounderId: args.actor,
    version: 1,
    // Creating never approves — not even for the proposer.
    status: args.submit ? 'PENDING_APPROVAL' : 'DRAFT',
    approvals: [],
    changeNote: null,
    history: [],
    committedRevision: null,
    createdAt: args.now,
    updatedAt: args.now,
  };
  return { store: { ...store, proposals: [...store.proposals, proposal] }, proposal, outcome: 'CREATED' };
}

/** Moves a draft to PENDING_APPROVAL. Proposer only. */
export function submitDraft(
  store: DecisionStore,
  args: { actor: FounderId; proposalId: string; version: number; now: string },
): TransitionResult<'SUBMITTED'> {
  assertFounder(args.actor);
  const p = findProposal(store, args.proposalId);
  if (p.createdByFounderId !== args.actor) fail('NOT_ALLOWED', 'Only the proposer can submit this draft.');
  if (p.status !== 'DRAFT') fail('NOT_ALLOWED', 'This proposal has already been submitted.');
  assertCurrentVersion(p, args.version);
  const proposal: DecisionProposal = { ...p, status: 'PENDING_APPROVAL', updatedAt: args.now };
  return { store: replaceProposal(store, proposal), proposal, outcome: 'SUBMITTED' };
}

/**
 * Edits a proposal. Proposer only.
 * - DRAFT: edited in place (nobody has reviewed it yet).
 * - PENDING_APPROVAL / CHANGES_REQUESTED: publishes version N+1 and CLEARS
 *   every approval — both founders must approve the new version again.
 * - COMMITTED: immutable; propose a new decision instead.
 */
export function reviseProposal(
  store: DecisionStore,
  args: {
    actor: FounderId;
    proposalId: string;
    version: number;
    content: ProposalContent;
    changeNote?: string | null;
    now: string;
  },
): TransitionResult<'DRAFT_UPDATED' | 'REVISED'> {
  assertFounder(args.actor);
  const p = findProposal(store, args.proposalId);
  if (p.createdByFounderId !== args.actor) fail('NOT_ALLOWED', 'Only the proposer can revise this proposal.');
  if (p.status === 'COMMITTED') fail('NOT_ALLOWED', 'A committed decision cannot be changed. Create a new proposal instead.');
  assertCurrentVersion(p, args.version);
  const content = normalizeContent(args.content);

  if (p.status === 'DRAFT') {
    const proposal: DecisionProposal = { ...p, ...content, updatedAt: args.now };
    return { store: replaceProposal(store, proposal), proposal, outcome: 'DRAFT_UPDATED' };
  }

  if (sameContent(content, p)) fail('INVALID_INPUT', 'Nothing changed. Edit the proposal before publishing a new version.');
  const note = args.changeNote?.trim() || null;
  if (note && note.length > LIMITS.changeNote) {
    fail('INVALID_INPUT', `The change note must be ${LIMITS.changeNote} characters or fewer.`);
  }

  const proposal: DecisionProposal = {
    ...p,
    ...content,
    version: p.version + 1,
    status: 'PENDING_APPROVAL',
    approvals: [], // previous approvals never carry over to a new version
    changeNote: note,
    history: [
      ...p.history,
      { ...contentOf(p), version: p.version, createdAt: p.updatedAt, changeNote: p.changeNote, approvals: p.approvals },
    ],
    updatedAt: args.now,
  };
  return { store: replaceProposal(store, proposal), proposal, outcome: 'REVISED' };
}

/**
 * Records the actor's approval of `version`. When both founders have approved
 * the same version, performs exactly one commit transition.
 */
export function approveProposal(
  store: DecisionStore,
  args: { actor: FounderId; proposalId: string; version: number; now: string },
): TransitionResult<'APPROVED' | 'ALREADY_APPROVED' | 'COMMITTED' | 'ALREADY_COMMITTED'> & {
  committed: CommittedStartupState;
} {
  assertFounder(args.actor);
  const p = findProposal(store, args.proposalId);
  assertCurrentVersion(p, args.version);

  // Idempotent: re-approving a committed version never commits twice.
  if (p.status === 'COMMITTED') return { store, proposal: p, committed: store.committed, outcome: 'ALREADY_COMMITTED' };
  if (p.status === 'DRAFT') fail('NOT_ALLOWED', 'This proposal is still a draft and cannot be approved yet.');
  if (p.status === 'CHANGES_REQUESTED') {
    fail('NOT_ALLOWED', 'Changes requested. Approval is paused until the proposer publishes a revised version.');
  }
  // Idempotent: no duplicate approval entries.
  if (decisionFor(p, args.actor)?.decision === 'APPROVED') {
    return { store, proposal: p, committed: store.committed, outcome: 'ALREADY_APPROVED' };
  }

  const approval: FounderApproval = {
    founderId: args.actor,
    proposalId: p.id,
    version: p.version,
    decision: 'APPROVED',
    reason: null,
    decidedAt: args.now,
  };
  let proposal: DecisionProposal = {
    ...p,
    approvals: [...p.approvals.filter((a) => a.founderId !== args.actor), approval],
    updatedAt: args.now,
  };

  if (!isApprovedByBoth(proposal)) {
    return { store: replaceProposal(store, proposal), proposal, committed: store.committed, outcome: 'APPROVED' };
  }

  // Single commit transition: official state advances by exactly one revision.
  const committed: CommittedStartupState = {
    projectId: store.projectId,
    revision: store.committed.revision + 1,
    decision: {
      ...contentOf(proposal),
      proposalId: proposal.id,
      proposalVersion: proposal.version,
      approvedBy: proposal.approvals
        .filter((a) => a.version === proposal.version && a.decision === 'APPROVED')
        .map((a) => ({ founderId: a.founderId, approvedAt: a.decidedAt })),
      committedAt: args.now,
    },
  };
  proposal = { ...proposal, status: 'COMMITTED', committedRevision: committed.revision };
  return { store: { ...replaceProposal(store, proposal), committed }, proposal, committed, outcome: 'COMMITTED' };
}

/** Blocks commitment until the proposer publishes a revised version. */
export function requestChanges(
  store: DecisionStore,
  args: { actor: FounderId; proposalId: string; version: number; reason: string; now: string },
): TransitionResult<'CHANGES_REQUESTED' | 'ALREADY_REQUESTED'> {
  assertFounder(args.actor);
  const p = findProposal(store, args.proposalId);
  assertCurrentVersion(p, args.version);
  if (p.status === 'COMMITTED') fail('NOT_ALLOWED', 'This decision is already committed. Create a new proposal instead.');
  if (p.status === 'DRAFT') fail('NOT_ALLOWED', 'This proposal is still a draft.');
  const reason = requiredText(args.reason, 'A reason', LIMITS.reason);

  const existing = decisionFor(p, args.actor);
  if (existing?.decision === 'CHANGES_REQUESTED') return { store, proposal: p, outcome: 'ALREADY_REQUESTED' };

  const record: FounderApproval = {
    founderId: args.actor,
    proposalId: p.id,
    version: p.version,
    decision: 'CHANGES_REQUESTED',
    reason,
    decidedAt: args.now,
  };
  const proposal: DecisionProposal = {
    ...p,
    // Replaces this founder's earlier approval of the same version, if any.
    approvals: [...p.approvals.filter((a) => a.founderId !== args.actor), record],
    status: 'CHANGES_REQUESTED',
    updatedAt: args.now,
  };
  return { store: replaceProposal(store, proposal), proposal, outcome: 'CHANGES_REQUESTED' };
}
