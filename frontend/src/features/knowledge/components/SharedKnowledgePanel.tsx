'use client';

import { useEffect, useMemo } from 'react';
import { AlertTriangle, BookOpen, RotateCw, SearchX, Share2, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Skeleton } from '@/components/ui/Skeleton';
import type { FounderId } from '@/types';
import { useSharedKnowledge } from '../hooks/useSharedKnowledge';
import type { KnowledgeFocusRequest } from '../hooks/useKnowledgeNavigation';
import { DEFAULT_KNOWLEDGE_FILTERS, filterKnowledge, isFiltering, type KnowledgeFilters as Filters } from '../lib/filterKnowledge';
import { ConflictCard } from './ConflictCard';
import { KnowledgeCard, knowledgeCardDomId } from './KnowledgeCard';
import { KnowledgeFilters } from './KnowledgeFilters';

interface SharedKnowledgePanelProps {
  projectId: string;
  founderId: FounderId;
  filters: Filters;
  onFiltersChange: (filters: Filters) => void;
  focusRequest: KnowledgeFocusRequest | null;
  onFocusHandled: () => void;
  onOpenReference: (id: string) => void;
  onCatchMeUp?: () => void;
  variant?: 'panel' | 'page';
}

function StateMessage({
  icon: Icon,
  title,
  body,
  action,
  tone = 'neutral',
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  body: string;
  action?: { label: string; onClick: () => void };
  tone?: 'neutral' | 'error';
}) {
  return (
    <div className="flex flex-col items-center text-center px-4 py-10" role={tone === 'error' ? 'alert' : undefined}>
      <div
        className={cn(
          'w-10 h-10 rounded-xl flex items-center justify-center mb-3',
          tone === 'error' ? 'bg-[var(--color-error-muted)]' : 'bg-[var(--color-bg-raised)]',
        )}
      >
        <Icon className={cn('w-5 h-5', tone === 'error' ? 'text-[var(--color-error)]' : 'text-[var(--color-text-muted)]')} />
      </div>
      <p className="text-sm font-medium text-[var(--color-text-primary)] mb-1">{title}</p>
      <p className="text-xs text-[var(--color-text-muted)] max-w-xs">{body}</p>
      {action && (
        <button
          type="button"
          onClick={action.onClick}
          className="mt-4 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-[var(--color-text-primary)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] hover:border-[var(--color-border-active)] transition-colors"
        >
          <RotateCw className="w-3.5 h-3.5" /> {action.label}
        </button>
      )}
    </div>
  );
}

export function SharedKnowledgePanel({
  projectId,
  founderId,
  filters,
  onFiltersChange,
  focusRequest,
  onFocusHandled,
  onOpenReference,
  onCatchMeUp,
  variant = 'panel',
}: SharedKnowledgePanelProps) {
  const knowledge = useSharedKnowledge(projectId);
  const ready = !knowledge.isLoading && !knowledge.error;
  const visible = useMemo(() => filterKnowledge(knowledge.items, filters), [knowledge.items, filters]);
  const filtering = isFiltering(filters);

  // Scroll to and briefly highlight a referenced card.
  useEffect(() => {
    if (!focusRequest || !ready) return;
    const el = document.getElementById(knowledgeCardDomId(focusRequest.id));
    if (el) {
      const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      el.scrollIntoView({ block: 'center', behavior: reduceMotion ? 'auto' : 'smooth' });
      el.focus({ preventScroll: true });
      const ring = getComputedStyle(document.documentElement).getPropertyValue('--color-focus').trim();
      el.animate?.(
        [{ boxShadow: `0 0 0 2px ${ring}` }, { boxShadow: '0 0 0 2px transparent' }],
        { duration: reduceMotion ? 0 : 2000, easing: 'ease-out' },
      );
      el.setAttribute('data-highlighted', 'true');
      window.setTimeout(() => el.removeAttribute('data-highlighted'), 2000);
    }
    onFocusHandled();
  }, [focusRequest, ready, onFocusHandled]);

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* What this is + Catch Me Up */}
      <div className="px-4 pt-3 pb-2 flex-shrink-0 space-y-2.5 border-b border-[var(--color-border)]">
        <div className="flex items-start gap-2">
          <Share2 className="w-3.5 h-3.5 mt-0.5 text-[var(--color-secondary)] flex-shrink-0" aria-hidden="true" />
          <p className="flex-1 text-[11px] leading-snug text-[var(--color-text-secondary)]">
            <span className="font-medium text-[var(--color-text-primary)]">What you both learned</span> — findings
            shared from each founder&apos;s work. Private chats are never shared.
            {knowledge.mode === 'demo' && (
              <span className="text-[var(--color-text-muted)]"> Demo findings are illustrative, not verified research.</span>
            )}
          </p>
          {onCatchMeUp && (
            <button
              type="button"
              onClick={onCatchMeUp}
              className="flex-shrink-0 flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium text-[var(--color-on-primary)] bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] transition-colors"
            >
              <Sparkles className="w-3 h-3" aria-hidden="true" /> Catch me up
            </button>
          )}
        </div>
        {ready && knowledge.items.length > 0 && (
          <KnowledgeFilters
            filters={filters}
            onChange={onFiltersChange}
            currentFounderId={founderId}
            shownCount={visible.length}
            totalCount={knowledge.items.length}
            variant={variant}
          />
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain px-4 py-3">
        {knowledge.error ? (
          <StateMessage
            icon={AlertTriangle}
            tone="error"
            title="Couldn't load shared knowledge"
            body={knowledge.error.userMessage}
            action={{ label: 'Try again', onClick: () => void knowledge.reload() }}
          />
        ) : knowledge.isLoading ? (
          <div className="space-y-2.5" aria-busy="true" aria-label="Loading shared knowledge">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-36 w-full rounded-xl" />
            ))}
          </div>
        ) : knowledge.items.length === 0 ? (
          <StateMessage
            icon={BookOpen}
            title="No shared findings yet"
            body="When either founder shares a discovery from their work, it will appear here for both of you."
          />
        ) : (
          <div className="space-y-4">
            {!filtering && knowledge.conflicts.length > 0 && (
              <section aria-labelledby={`conflicts-${variant}`} className="space-y-2">
                <h3
                  id={`conflicts-${variant}`}
                  className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]"
                >
                  Needs discussion · {knowledge.conflicts.length}
                </h3>
                {knowledge.conflicts.map((conflict) => (
                  <ConflictCard
                    key={conflict.id}
                    conflict={conflict}
                    itemsById={knowledge.itemsById}
                    currentFounderId={founderId}
                    onOpenReference={onOpenReference}
                    layout={variant === 'page' ? 'split' : 'stacked'}
                  />
                ))}
              </section>
            )}

            <section aria-labelledby={`findings-${variant}`} className="space-y-2">
              <h3
                id={`findings-${variant}`}
                className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]"
              >
                Findings
              </h3>
              {visible.length === 0 ? (
                <div className="flex flex-col items-center text-center py-8">
                  <SearchX className="w-5 h-5 mb-2 text-[var(--color-text-muted)]" aria-hidden="true" />
                  <p className="text-xs text-[var(--color-text-secondary)] mb-2">No findings match these filters.</p>
                  <button
                    type="button"
                    onClick={() => onFiltersChange(DEFAULT_KNOWLEDGE_FILTERS)}
                    className="text-xs font-medium text-[var(--color-primary)] hover:text-[var(--color-primary-hover)]"
                  >
                    Reset filters
                  </button>
                </div>
              ) : (
                <div className={cn('grid gap-2.5', variant === 'page' && 'lg:grid-cols-2')}>
                  {visible.map((item) => (
                    <KnowledgeCard
                      key={item.id}
                      item={item}
                      currentFounderId={founderId}
                      conflictPartnerIds={knowledge.conflictPartners.get(item.id)}
                      itemsById={knowledge.itemsById}
                      onOpenReference={onOpenReference}
                    />
                  ))}
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
