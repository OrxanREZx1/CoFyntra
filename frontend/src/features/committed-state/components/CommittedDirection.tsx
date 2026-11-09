'use client';

import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Hourglass, RotateCw, ShieldCheck, Target } from 'lucide-react';
import { cn } from '@/lib/utils';
import { FounderAvatar } from '@/components/ui/FounderAvatar';
import { RelativeTime } from '@/components/ui/RelativeTime';
import { Skeleton } from '@/components/ui/Skeleton';
import { useProposals } from '@/features/proposals/hooks/useProposals';
import { useCommittedState } from '../hooks/useCommittedState';

interface CommittedDirectionProps {
  projectId: string;
  /** Compact mode clamps long text (right panel). */
  compact?: boolean;
}

/**
 * The project's OFFICIAL direction. Renders only service-confirmed committed
 * state — a pending proposal never appears here, however many approvals it has.
 */
export function CommittedDirection({ projectId, compact = false }: CommittedDirectionProps) {
  const { state, isLoading, error, reload } = useCommittedState(projectId);
  const { proposals } = useProposals(projectId);
  const [expanded, setExpanded] = useState(!compact);
  const cardRef = useRef<HTMLElement>(null);
  const lastRevision = useRef<number | null>(null);

  const openProposals = proposals.filter((p) => p.status === 'PENDING_APPROVAL' || p.status === 'CHANGES_REQUESTED').length;

  // Briefly highlight when a NEW commit arrives (not on first load).
  useEffect(() => {
    if (!state) return;
    const prev = lastRevision.current;
    lastRevision.current = state.revision;
    if (prev !== null && state.revision > prev) {
      const ring = getComputedStyle(document.documentElement).getPropertyValue('--color-secondary').trim();
      cardRef.current?.animate?.(
        [{ boxShadow: `0 0 0 2px ${ring}` }, { boxShadow: '0 0 0 2px transparent' }],
        { duration: 2200, easing: 'ease-out' },
      );
    }
  }, [state]);

  const shell = 'mx-4 my-3 p-3 rounded-xl border';

  if (error) {
    return (
      <div className={cn(shell, 'border-[var(--color-border)] bg-[var(--color-bg-elevated)] flex items-center gap-2')} role="alert">
        <p className="flex-1 text-xs text-[var(--color-text-muted)]">Couldn&apos;t load the committed direction.</p>
        <button type="button" onClick={() => void reload()} aria-label="Retry" className="p-1 rounded text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]">
          <RotateCw className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  if (isLoading || !state) {
    return (
      <div className={cn(shell, 'border-[var(--color-border)] bg-[var(--color-bg-elevated)] space-y-2')} aria-busy="true">
        <Skeleton className="h-3 w-1/3" />
        <Skeleton className="h-4 w-full" />
      </div>
    );
  }

  const decision = state.decision;
  const pendingNote =
    openProposals > 0 &&
    (decision
      ? `${openProposals} new ${openProposals === 1 ? 'proposal is' : 'proposals are'} in review — this stays the official direction until both founders approve a new version.`
      : `${openProposals} ${openProposals === 1 ? 'proposal is' : 'proposals are'} in review.`);

  if (!decision) {
    return (
      <section aria-label="Committed direction" className={cn(shell, 'border-[var(--color-border)] bg-[var(--color-bg-elevated)]')}>
        <div className="flex items-center gap-2 mb-1.5">
          <Target className="w-3.5 h-3.5 text-[var(--color-text-muted)]" aria-hidden="true" />
          <h2 className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">Committed direction</h2>
        </div>
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">No direction committed yet.</p>
        <p className="mt-1 text-[10px] text-[var(--color-text-muted)] leading-relaxed">
          A direction becomes official only when both founders approve the same proposal version.
        </p>
        {pendingNote && (
          <p className="mt-1.5 flex items-center gap-1 text-[10px] text-[var(--color-primary)]">
            <Hourglass className="w-3 h-3" aria-hidden="true" /> {pendingNote}
          </p>
        )}
      </section>
    );
  }

  return (
    <section
      ref={cardRef}
      aria-label="Committed direction"
      className={cn(shell, 'border-[var(--color-secondary)]/40 bg-[var(--color-secondary-muted)]')}
    >
      <div className="flex items-center gap-1.5 mb-1.5">
        <ShieldCheck className="w-3.5 h-3.5 text-[var(--color-secondary)]" aria-hidden="true" />
        <h2 className="text-[10px] font-semibold text-[var(--color-secondary)] uppercase tracking-wider">
          Official direction
        </h2>
        <span className="ml-auto font-mono text-[10px] text-[var(--color-secondary)]">Revision {state.revision}</span>
      </div>

      <p className="text-xs font-semibold text-[var(--color-text-primary)] wrap-anywhere">{decision.title}</p>
      <p className={cn('mt-0.5 text-sm text-[var(--color-text-primary)] leading-relaxed wrap-anywhere', !expanded && 'line-clamp-2')}>
        {decision.direction}
      </p>
      <p className={cn('mt-1.5 flex items-start gap-1 text-xs text-[var(--color-text-secondary)] wrap-anywhere', !expanded && 'line-clamp-2')}>
        <ArrowRight className="w-3 h-3 mt-0.5 flex-shrink-0 text-[var(--color-secondary)]" aria-hidden="true" />
        <span>
          <span className="font-medium text-[var(--color-secondary)]">Next action: </span>
          {decision.nextAction}
        </span>
      </p>
      {compact && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          className="mt-0.5 text-[10px] font-medium text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1">
        <div className="flex -space-x-1">
          {decision.approvedBy.map((a) => (
            <FounderAvatar key={a.founderId} founderId={a.founderId} size="sm" />
          ))}
        </div>
        <span className="text-[10px] text-[var(--color-text-secondary)]">Confirmed by both founders</span>
        <span className="text-[10px] text-[var(--color-text-muted)]">
          · {decision.proposalId} v{decision.proposalVersion} · <RelativeTime timestamp={decision.committedAt} />
        </span>
      </div>

      {pendingNote && (
        <p className="mt-2 pt-2 border-t border-[var(--color-secondary)]/20 flex items-start gap-1 text-[10px] text-[var(--color-text-secondary)]">
          <Hourglass className="w-3 h-3 mt-px flex-shrink-0 text-[var(--color-primary)]" aria-hidden="true" /> {pendingNote}
        </p>
      )}
    </section>
  );
}
