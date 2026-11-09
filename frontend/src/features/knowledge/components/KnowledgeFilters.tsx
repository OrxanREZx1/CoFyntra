'use client';

import { useId } from 'react';
import { RotateCcw, Search, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { EPISTEMIC_STATUS_META } from '@/components/ui/EpistemicBadge';
import { FOUNDERS } from '@/mocks';
import { EPISTEMIC_STATUSES, KNOWLEDGE_CATEGORIES, type FounderId } from '@/types';
import { DEFAULT_KNOWLEDGE_FILTERS, isFiltering, type KnowledgeFilters as Filters } from '../lib/filterKnowledge';

const FOUNDER_IDS: FounderId[] = ['founder-a', 'founder-b'];

interface KnowledgeFiltersProps {
  filters: Filters;
  onChange: (filters: Filters) => void;
  currentFounderId: FounderId;
  shownCount: number;
  totalCount: number;
  /** 'page' lays the selects out in one row on wider screens. */
  variant?: 'panel' | 'page';
}

const selectClass = cn(
  'w-full min-w-0 h-8 px-2 rounded-lg text-xs appearance-none cursor-pointer',
  'bg-[var(--color-bg-elevated)] border border-[var(--color-border)] text-[var(--color-text-secondary)]',
  'hover:border-[var(--color-border-active)] focus-visible:border-[var(--color-primary)] outline-none',
);

export function KnowledgeFilters({
  filters,
  onChange,
  currentFounderId,
  shownCount,
  totalCount,
  variant = 'panel',
}: KnowledgeFiltersProps) {
  const id = useId();
  const set = <K extends keyof Filters>(key: K, value: Filters[K]) => onChange({ ...filters, [key]: value });
  const active = isFiltering(filters);

  return (
    <div role="search" aria-label="Filter shared knowledge" className="space-y-2">
      <div className="relative">
        <label htmlFor={`${id}-q`} className="sr-only">
          Search findings
        </label>
        <Search
          className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--color-text-muted)] pointer-events-none"
          aria-hidden="true"
        />
        <input
          id={`${id}-q`}
          type="search"
          value={filters.query}
          onChange={(e) => set('query', e.target.value)}
          placeholder="Search findings…"
          autoComplete="off"
          className={cn(
            'w-full h-8 pl-8 pr-8 rounded-lg text-base sm:text-xs outline-none',
            'bg-[var(--color-bg-elevated)] border border-[var(--color-border)] text-[var(--color-text-primary)]',
            'placeholder:text-[var(--color-text-disabled)] focus-visible:border-[var(--color-primary)]',
            '[&::-webkit-search-cancel-button]:hidden',
          )}
        />
        {filters.query && (
          <button
            type="button"
            onClick={() => set('query', '')}
            aria-label="Clear search"
            className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]"
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>

      <div className={cn('grid grid-cols-3 gap-1.5', variant === 'page' && 'sm:max-w-xl')}>
        <div>
          <label htmlFor={`${id}-cat`} className="sr-only">
            Category
          </label>
          <select
            id={`${id}-cat`}
            value={filters.category}
            onChange={(e) => set('category', e.target.value as Filters['category'])}
            className={selectClass}
          >
            <option value="all">All categories</option>
            {KNOWLEDGE_CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor={`${id}-who`} className="sr-only">
            Contributor
          </label>
          <select
            id={`${id}-who`}
            value={filters.contributorId}
            onChange={(e) => set('contributorId', e.target.value as Filters['contributorId'])}
            className={selectClass}
          >
            <option value="all">All founders</option>
            {FOUNDER_IDS.map((f) => (
              <option key={f} value={f}>
                {FOUNDERS[f]?.name ?? f}
                {f === currentFounderId ? ' (you)' : ''}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor={`${id}-status`} className="sr-only">
            Evidence status
          </label>
          <select
            id={`${id}-status`}
            value={filters.status}
            onChange={(e) => set('status', e.target.value as Filters['status'])}
            className={selectClass}
          >
            <option value="all">All statuses</option>
            {EPISTEMIC_STATUSES.map((s) => (
              <option key={s} value={s}>
                {EPISTEMIC_STATUS_META[s].label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex items-center justify-between h-5">
        <p className="text-[10px] text-[var(--color-text-muted)]" aria-live="polite">
          {active ? `Showing ${shownCount} of ${totalCount} findings` : `${totalCount} shared findings`}
        </p>
        {active && (
          <button
            type="button"
            onClick={() => onChange(DEFAULT_KNOWLEDGE_FILTERS)}
            className="flex items-center gap-1 text-[10px] font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
          >
            <RotateCcw className="w-3 h-3" aria-hidden="true" /> Reset filters
          </button>
        )}
      </div>
    </div>
  );
}
