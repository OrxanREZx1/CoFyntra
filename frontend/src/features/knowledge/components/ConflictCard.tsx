'use client';

import { Scale } from 'lucide-react';
import { cn } from '@/lib/utils';
import { EpistemicBadge } from '@/components/ui/EpistemicBadge';
import { FounderAvatar } from '@/components/ui/FounderAvatar';
import type { FounderId, KnowledgeConflict, KnowledgeItem } from '@/types';
import { KnowledgeReference } from './KnowledgeReference';

interface ConflictCardProps {
  conflict: KnowledgeConflict;
  itemsById: ReadonlyMap<string, KnowledgeItem>;
  currentFounderId: FounderId;
  onOpenReference: (id: string) => void;
  /** 'split' puts both findings side by side on wider screens. */
  layout?: 'stacked' | 'split';
}

/** One attributed finding: founder, claim, status and a reference chip. */
export function FindingSummary({
  id,
  item,
  currentFounderId,
  onOpenReference,
}: {
  id: string;
  item: KnowledgeItem | undefined;
  currentFounderId: FounderId;
  onOpenReference: (id: string) => void;
}) {
  if (!item) {
    return (
      <div className="p-2.5 rounded-lg border border-dashed border-[var(--color-border)]">
        <KnowledgeReference id={id} item={undefined} onOpen={onOpenReference} />
      </div>
    );
  }
  return (
    <div className="p-2.5 rounded-lg bg-[var(--color-bg-base)] border border-[var(--color-border)] min-w-0">
      <div className="flex items-center gap-1.5 mb-1.5">
        <FounderAvatar founderId={item.contributorId} size="sm" />
        <span className="text-[11px] font-medium text-[var(--color-text-secondary)] truncate">
          {item.contributorName}
          {item.contributorId === currentFounderId && <span className="text-[var(--color-text-muted)]"> (you)</span>}
        </span>
      </div>
      <p className="text-xs text-[var(--color-text-primary)] leading-relaxed wrap-anywhere">“{item.finding}”</p>
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <EpistemicBadge status={item.epistemicStatus} compact />
        <KnowledgeReference id={id} item={item} onOpen={onOpenReference} />
      </div>
    </div>
  );
}

/**
 * Presents two findings that may disagree, side by side and neutrally.
 * It never declares a winner and never merges or hides either finding.
 */
export function ConflictCard({
  conflict,
  itemsById,
  currentFounderId,
  onOpenReference,
  layout = 'stacked',
}: ConflictCardProps) {
  const [a, b] = conflict.itemIds;

  return (
    <section
      aria-label={`Potential conflict between ${a} and ${b}`}
      className="p-3 rounded-xl border border-[var(--color-warning)]/25 bg-[var(--color-warning-muted)]/40"
    >
      <div className="flex items-center gap-1.5 mb-1">
        <Scale className="w-3.5 h-3.5 text-[var(--color-warning)]" aria-hidden="true" />
        <h4 className="text-xs font-semibold text-[var(--color-warning)]">Potential conflict</h4>
      </div>
      <p className="text-[11px] text-[var(--color-text-secondary)] leading-relaxed mb-2.5">
        {conflict.explanation ?? 'These findings may point in different directions.'} Worth discussing together —
        neither is marked as correct.
      </p>
      <div className={cn('grid gap-2', layout === 'split' && 'md:grid-cols-2')}>
        <FindingSummary id={a} item={itemsById.get(a)} currentFounderId={currentFounderId} onOpenReference={onOpenReference} />
        <FindingSummary id={b} item={itemsById.get(b)} currentFounderId={currentFounderId} onOpenReference={onOpenReference} />
      </div>
    </section>
  );
}
