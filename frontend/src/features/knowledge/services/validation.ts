import { ApiError } from '@/lib/api/errors';
import {
  EPISTEMIC_STATUSES,
  KNOWLEDGE_CATEGORIES,
  type EpistemicStatus,
  type FounderId,
  type KnowledgeCategory,
  type KnowledgeConflict,
  type KnowledgeItem,
  type SharedKnowledgeSnapshot,
} from '@/types';

const FOUNDER_IDS: readonly FounderId[] = ['founder-a', 'founder-b'];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function optionalString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() !== '' ? value : null;
}

/**
 * Builds a KnowledgeItem from a whitelist of fields. Anything else in the
 * payload (e.g. an accidentally included transcript) is dropped, so it never
 * reaches the shared cache. Returns null for malformed or out-of-project items.
 */
function parseItem(raw: unknown, projectId: string): KnowledgeItem | null {
  if (!isRecord(raw)) return null;
  const { id, category, finding, contributorId, createdAt, epistemicStatus } = raw;
  if (
    typeof id !== 'string' ||
    raw.projectId !== projectId ||
    !KNOWLEDGE_CATEGORIES.includes(category as KnowledgeCategory) ||
    typeof finding !== 'string' ||
    finding.trim() === '' ||
    !FOUNDER_IDS.includes(contributorId as FounderId) ||
    typeof createdAt !== 'string' ||
    Number.isNaN(Date.parse(createdAt)) ||
    !EPISTEMIC_STATUSES.includes(epistemicStatus as EpistemicStatus)
  ) {
    return null;
  }
  return {
    id,
    projectId,
    category: category as KnowledgeCategory,
    finding,
    contributorId: contributorId as FounderId,
    contributorName: optionalString(raw.contributorName) ?? 'Unknown contributor',
    sourceRef: optionalString(raw.sourceRef),
    supportingExcerpt: optionalString(raw.supportingExcerpt),
    createdAt,
    epistemicStatus: epistemicStatus as EpistemicStatus,
  };
}

function parseConflict(raw: unknown, projectId: string, knownIds: Set<string>): KnowledgeConflict | null {
  if (!isRecord(raw) || typeof raw.id !== 'string' || raw.projectId !== projectId) return null;
  const ids = raw.itemIds;
  if (!Array.isArray(ids) || ids.length !== 2 || !ids.every((x) => typeof x === 'string')) return null;
  const [a, b] = ids as [string, string];
  // Only keep relationships between items that exist in this project.
  if (a === b || !knownIds.has(a) || !knownIds.has(b)) return null;
  return { id: raw.id, projectId, itemIds: [a, b], explanation: optionalString(raw.explanation) };
}

export function parseKnowledgeSnapshot(raw: unknown, projectId: string): SharedKnowledgeSnapshot {
  if (!isRecord(raw) || !Array.isArray(raw.items)) throw new ApiError('invalid_response');

  const items = raw.items
    .map((item) => parseItem(item, projectId))
    .filter((item): item is KnowledgeItem => item !== null);
  const knownIds = new Set(items.map((i) => i.id));
  const conflicts = (Array.isArray(raw.conflicts) ? raw.conflicts : [])
    .map((c) => parseConflict(c, projectId, knownIds))
    .filter((c): c is KnowledgeConflict => c !== null);

  return { projectId, items, conflicts };
}
