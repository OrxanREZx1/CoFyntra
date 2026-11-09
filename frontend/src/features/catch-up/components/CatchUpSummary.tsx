'use client';

import { Clock, RefreshCw } from 'lucide-react';
import { cn, formatFullTimestamp } from '@/lib/utils';
import { FounderAvatar } from '@/components/ui/FounderAvatar';
import { RelativeTime } from '@/components/ui/RelativeTime';
import type { CatchUpSummary as Summary } from '@/types';

interface CatchUpSummaryProps {
  summary: Summary;
  isDemo: boolean;
  isRefreshing: boolean;
  onRefresh: () => void;
}

/** Heading, co-founder, coverage and the one-paragraph summary. */
export function CatchUpSummary({ summary, isDemo, isRefreshing, onRefresh }: CatchUpSummaryProps) {
  const { coverage } = summary;

  return (
    <header className="space-y-2">
      <div className="flex items-center gap-2">
        <FounderAvatar founderId={summary.coFounderId} size="md" />
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Catch me up</h2>
          <p className="text-[11px] text-[var(--color-text-muted)] truncate">
            What {summary.coFounderName} has shared with you
          </p>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          disabled={isRefreshing}
          aria-label="Refresh catch-up"
          className="p-1.5 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)] disabled:opacity-50"
        >
          <RefreshCw className={cn('w-3.5 h-3.5', isRefreshing && 'motion-safe:animate-spin')} />
        </button>
      </div>

      <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{summary.summary}</p>

      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10px] text-[var(--color-text-muted)]">
        {coverage?.to && (
          <span className="flex items-center gap-1">
            <Clock className="w-2.5 h-2.5" aria-hidden="true" />
            Covers {coverage.itemCount} shared {coverage.itemCount === 1 ? 'finding' : 'findings'}
            {coverage.from && coverage.from !== coverage.to
              ? `, ${formatFullTimestamp(coverage.from)} – ${formatFullTimestamp(coverage.to)}`
              : `, ${formatFullTimestamp(coverage.to)}`}
          </span>
        )}
        {summary.generatedAt && (
          <span>
            Generated <RelativeTime timestamp={summary.generatedAt} />
          </span>
        )}
        {isDemo && <span>· Demo summary assembled from shared findings — not AI-generated</span>}
      </div>
    </header>
  );
}
