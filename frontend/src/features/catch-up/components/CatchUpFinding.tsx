'use client';

import type { CatchUpFinding as Finding, KnowledgeItem } from '@/types';
import { KnowledgeReference } from '@/features/knowledge/components/KnowledgeReference';

interface CatchUpFindingProps {
  finding: Finding;
  itemsById: ReadonlyMap<string, KnowledgeItem>;
  onOpenReference: (id: string) => void;
}

/** One catch-up statement plus the shared knowledge items it is based on. */
export function CatchUpFinding({ finding, itemsById, onOpenReference }: CatchUpFindingProps) {
  return (
    <li className="p-2.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border)]">
      <p className="text-xs text-[var(--color-text-primary)] leading-relaxed wrap-anywhere">{finding.text}</p>
      <div className="mt-1.5 flex flex-wrap items-center gap-1">
        <span className="text-[10px] text-[var(--color-text-muted)]">Based on</span>
        {finding.knowledgeItemIds.map((id) => (
          <KnowledgeReference key={id} id={id} item={itemsById.get(id)} onOpen={onOpenReference} />
        ))}
      </div>
    </li>
  );
}
