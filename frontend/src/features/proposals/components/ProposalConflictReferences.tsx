'use client';

import { AlertTriangle } from 'lucide-react';
import type { FounderId, KnowledgeConflict, KnowledgeItem, ProposalContent } from '@/types';
import { ConflictCard, FindingSummary } from '@/features/knowledge/components/ConflictCard';
import { getProposalConflicts } from '../lib/proposalMachine';

interface ProposalConflictReferencesProps {
  proposal: Pick<ProposalContent, 'supportingKnowledgeIds' | 'conflictingKnowledgeIds'>;
  itemsById: ReadonlyMap<string, KnowledgeItem>;
  knowledgeConflicts: readonly KnowledgeConflict[];
  founderId: FounderId;
  onOpenReference: (id: string) => void;
  layout?: 'stacked' | 'split';
}

/**
 * Surfaces conflicting shared findings referenced by a proposal. Uses only
 * supplied conflict data; it never rejects the proposal or picks a side.
 */
export function ProposalConflictReferences({
  proposal,
  itemsById,
  knowledgeConflicts,
  founderId,
  onOpenReference,
  layout = 'stacked',
}: ProposalConflictReferencesProps) {
  const conflicts = getProposalConflicts(proposal, knowledgeConflicts);
  if (conflicts.length === 0) return null;

  return (
    <section aria-label="Potential conflicts with this proposal" className="space-y-2">
      <div className="flex items-start gap-1.5 px-2.5 py-2 rounded-lg bg-[var(--color-warning-muted)] border border-[var(--color-warning)]/25">
        <AlertTriangle className="w-3.5 h-3.5 mt-px text-[var(--color-warning)] flex-shrink-0" aria-hidden="true" />
        <p className="text-[11px] leading-relaxed text-[var(--color-text-secondary)]">
          <span className="font-semibold text-[var(--color-warning)]">Potential conflict · </span>
          {conflicts.length === 1 ? '1 shared finding points' : `${conflicts.length} shared findings point`} another way.
          Worth discussing before approving — this doesn&apos;t mean either founder is wrong.
        </p>
      </div>
      {conflicts.map((c) => {
        const pair = c.knowledgeConflictId ? knowledgeConflicts.find((k) => k.id === c.knowledgeConflictId) : undefined;
        return pair ? (
          <ConflictCard
            key={c.conflictingItemId}
            conflict={pair}
            itemsById={itemsById}
            currentFounderId={founderId}
            onOpenReference={onOpenReference}
            layout={layout}
          />
        ) : (
          <FindingSummary
            key={c.conflictingItemId}
            id={c.conflictingItemId}
            item={itemsById.get(c.conflictingItemId)}
            currentFounderId={founderId}
            onOpenReference={onOpenReference}
          />
        );
      })}
    </section>
  );
}
