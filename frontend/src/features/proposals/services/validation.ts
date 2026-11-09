import { ApiError } from '@/lib/api/errors';
import {
  PROPOSAL_STATUSES,
  type CommittedStartupState,
  type DecisionProposal,
  type FounderApproval,
  type FounderId,
  type ProposalStatus,
  type ProposalVersion,
} from '@/types';

/**
 * Strict parsing for L3 payloads. Rebuilds objects from known fields only,
 * and rejects anything outside the selected project.
 */

const FOUNDERS: readonly FounderId[] = ['founder-a', 'founder-b'];
const invalid = (): never => {
  throw new ApiError('invalid_response');
};

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null;
const str = (v: unknown): string => (typeof v === 'string' ? v : invalid());
const nullableStr = (v: unknown): string | null => (typeof v === 'string' && v.trim() ? v : null);
const date = (v: unknown): string => (typeof v === 'string' && !Number.isNaN(Date.parse(v)) ? v : invalid());
const posInt = (v: unknown): number => (typeof v === 'number' && Number.isInteger(v) && v >= 1 ? v : invalid());
const founder = (v: unknown): FounderId => (FOUNDERS.includes(v as FounderId) ? (v as FounderId) : invalid());
const ids = (v: unknown): string[] => (Array.isArray(v) && v.every((x) => typeof x === 'string') ? [...v] : invalid());

function parseApproval(raw: unknown, proposalId: string): FounderApproval {
  if (!isRecord(raw)) return invalid();
  if (raw.proposalId !== proposalId) return invalid();
  if (raw.decision !== 'APPROVED' && raw.decision !== 'CHANGES_REQUESTED') return invalid();
  return {
    founderId: founder(raw.founderId),
    proposalId,
    version: posInt(raw.version),
    decision: raw.decision,
    reason: nullableStr(raw.reason),
    decidedAt: date(raw.decidedAt),
  };
}

function parseContent(raw: Record<string, unknown>) {
  return {
    title: str(raw.title),
    direction: str(raw.direction),
    nextAction: str(raw.nextAction),
    supportingKnowledgeIds: ids(raw.supportingKnowledgeIds),
    conflictingKnowledgeIds: ids(raw.conflictingKnowledgeIds),
  };
}

export function parseProposal(raw: unknown, projectId: string): DecisionProposal {
  if (!isRecord(raw) || raw.projectId !== projectId) return invalid();
  const id = str(raw.id);
  if (!PROPOSAL_STATUSES.includes(raw.status as ProposalStatus)) return invalid();
  const version = posInt(raw.version);
  const approvals = (Array.isArray(raw.approvals) ? raw.approvals : invalid()).map((a) => parseApproval(a, id));
  const history: ProposalVersion[] = (Array.isArray(raw.history) ? raw.history : []).map((h) => {
    if (!isRecord(h)) return invalid();
    return {
      ...parseContent(h),
      version: posInt(h.version),
      createdAt: date(h.createdAt),
      changeNote: nullableStr(h.changeNote),
      approvals: (Array.isArray(h.approvals) ? h.approvals : []).map((a) => parseApproval(a, id)),
    };
  });
  return {
    ...parseContent(raw),
    id,
    projectId,
    createdByFounderId: founder(raw.createdByFounderId),
    version,
    status: raw.status as ProposalStatus,
    // Only decisions on the current version are meaningful.
    approvals: approvals.filter((a) => a.version === version),
    changeNote: nullableStr(raw.changeNote),
    history,
    committedRevision: typeof raw.committedRevision === 'number' ? raw.committedRevision : null,
    createdAt: date(raw.createdAt),
    updatedAt: date(raw.updatedAt),
  };
}

export function parseProposalList(raw: unknown, projectId: string): DecisionProposal[] {
  return (Array.isArray(raw) ? raw : invalid()).map((p) => parseProposal(p, projectId));
}

export function parseCommittedState(raw: unknown, projectId: string): CommittedStartupState {
  if (!isRecord(raw) || raw.projectId !== projectId) return invalid();
  const revision = typeof raw.revision === 'number' && Number.isInteger(raw.revision) && raw.revision >= 0 ? raw.revision : invalid();
  if (raw.decision === null || raw.decision === undefined) {
    return revision === 0 ? { projectId, revision, decision: null } : invalid();
  }
  const d = isRecord(raw.decision) ? raw.decision : invalid();
  const approvedBy = (Array.isArray(d.approvedBy) ? d.approvedBy : invalid()).map((a) => {
    if (!isRecord(a)) return invalid();
    return { founderId: founder(a.founderId), approvedAt: date(a.approvedAt) };
  });
  // An official decision always carries both founders' approvals.
  if (!FOUNDERS.every((f) => approvedBy.some((a) => a.founderId === f))) return invalid();
  return {
    projectId,
    revision,
    decision: {
      ...parseContent(d),
      proposalId: str(d.proposalId),
      proposalVersion: posInt(d.proposalVersion),
      approvedBy,
      committedAt: date(d.committedAt),
    },
  };
}
