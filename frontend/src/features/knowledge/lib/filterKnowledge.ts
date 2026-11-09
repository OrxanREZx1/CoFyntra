// Pure filtering logic for shared knowledge (type-only imports, no React).
import type { EpistemicStatus, FounderId, KnowledgeCategory, KnowledgeItem } from '@/types';

export interface KnowledgeFilters {
  /** Free-text search over the finding/claim text. */
  query: string;
  category: KnowledgeCategory | 'all';
  contributorId: FounderId | 'all';
  status: EpistemicStatus | 'all';
}

export const DEFAULT_KNOWLEDGE_FILTERS: KnowledgeFilters = {
  query: '',
  category: 'all',
  contributorId: 'all',
  status: 'all',
};

export function isFiltering(filters: KnowledgeFilters): boolean {
  return (
    filters.query.trim() !== '' ||
    filters.category !== 'all' ||
    filters.contributorId !== 'all' ||
    filters.status !== 'all'
  );
}

export function matchesFilters(item: KnowledgeItem, filters: KnowledgeFilters): boolean {
  if (filters.category !== 'all' && item.category !== filters.category) return false;
  if (filters.contributorId !== 'all' && item.contributorId !== filters.contributorId) return false;
  if (filters.status !== 'all' && item.epistemicStatus !== filters.status) return false;
  const query = filters.query.trim().toLowerCase();
  if (query && !item.finding.toLowerCase().includes(query)) return false;
  return true;
}

/** Filters items and returns them newest first. Does not mutate the input. */
export function filterKnowledge(items: readonly KnowledgeItem[], filters: KnowledgeFilters): KnowledgeItem[] {
  return items
    .filter((item) => matchesFilters(item, filters))
    .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
}
