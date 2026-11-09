'use client';

import { memo, useState } from 'react';
import { AlertTriangle, Lock } from 'lucide-react';
import { cn } from '@/lib/utils';
import { EpistemicBadge } from '@/components/ui/EpistemicBadge';
import { FounderAvatar } from '@/components/ui/FounderAvatar';
import { RelativeTime } from '@/components/ui/RelativeTime';
import type { FounderId, KnowledgeItem } from '@/types';
import { CATEGORY_META } from './categoryMeta';
import { KnowledgeReference } from './KnowledgeReference';

const EXCERPT_CLAMP_CHARS = 160;

export const knowledgeCardDomId = (itemId: string) => `knowledge-item-${itemId}`;

interface KnowledgeCardProps {
  item: KnowledgeItem;
  currentFounderId: FounderId;
  /** IDs of items this one potentially conflicts with. */
  conflictPartnerIds?: string[];
  itemsById: ReadonlyMap<string, KnowledgeItem>;
  onOpenReference: (id: string) => void;
}

function KnowledgeCardImpl({
  item,
  currentFounderId,
  conflictPartnerIds = [],
  itemsById,
  onOpenReference,
}: KnowledgeCardProps) {
  const [expanded, setExpanded] = useState(false);
  const category = CATEGORY_META[item.category];
  const CategoryIcon = category.icon;
  const isOwn = item.contributorId === currentFounderId;
  const excerpt = item.supportingExcerpt;
  const longExcerpt = (excerpt?.length ?? 0) > EXCERPT_CLAMP_CHARS;
  const hasConflict = conflictPartnerIds.length > 0;

  return (
    <article
      id={knowledgeCardDomId(item.id)}
      tabIndex={-1}
      aria-labelledby={`${knowledgeCardDomId(item.id)}-finding`}
      className={cn(
        'p-3 rounded-xl border outline-none transition-colors scroll-mt-4',
        'bg-[var(--color-bg-elevated)] border-[var(--color-border)]',
        'hover:border-[var(--color-border-active)] focus-visible:border-[var(--color-primary)]',
        hasConflict && 'border-[var(--color-warning)]/30',
      )}
    >
      {/* Category · ID · status */}
      <div className="flex flex-wrap items-center gap-1.5 mb-2">
        <span
          className={cn(
            'inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium border',
            category.className,
          )}
        >
          <CategoryIcon className="w-2.5 h-2.5" aria-hidden="true" />
          {item.category}
        </span>
        <span className="font-mono text-[10px] text-[var(--color-text-muted)]">{item.id}</span>
        <EpistemicBadge status={item.epistemicStatus} compact className="ml-auto" />
      </div>

      {/* Finding */}
      <p
        id={`${knowledgeCardDomId(item.id)}-finding`}
        className="text-sm text-[var(--color-text-primary)] leading-relaxed wrap-anywhere"
      >
        {item.finding}
      </p>

      {/* Supporting excerpt */}
      {excerpt ? (
        <figure className="mt-2">
          <blockquote
            className={cn(
              'pl-2.5 border-l-2 border-[var(--color-border-active)] text-xs italic leading-relaxed',
              'text-[var(--color-text-secondary)] wrap-anywhere',
              longExcerpt && !expanded && 'line-clamp-3',
            )}
          >
            “{excerpt}”
          </blockquote>
          {longExcerpt && (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              aria-expanded={expanded}
              className="mt-0.5 pl-2.5 text-[10px] font-medium text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]"
            >
              {expanded ? 'Show less' : 'Show more'}
            </button>
          )}
        </figure>
      ) : (
        <p className="mt-2 text-[11px] text-[var(--color-text-disabled)]">No supporting excerpt was shared.</p>
      )}

      {/* Conflict indicator */}
      {hasConflict && (
        <div className="mt-2.5 flex flex-wrap items-center gap-1.5 px-2 py-1.5 rounded-lg bg-[var(--color-warning-muted)] border border-[var(--color-warning)]/20">
          <AlertTriangle className="w-3 h-3 text-[var(--color-warning)] flex-shrink-0" aria-hidden="true" />
          <span className="text-[10px] font-medium text-[var(--color-warning)]">Potential conflict with</span>
          {conflictPartnerIds.map((id) => (
            <KnowledgeReference key={id} id={id} item={itemsById.get(id)} onOpen={onOpenReference} />
          ))}
        </div>
      )}

      {/* Attribution · time */}
      <div className="mt-2.5 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <FounderAvatar founderId={item.contributorId} size="sm" />
          <span className="text-[11px] text-[var(--color-text-secondary)] truncate">
            {item.contributorName}
            {isOwn && <span className="text-[var(--color-text-muted)]"> (you)</span>}
          </span>
        </div>
        <span className="text-[10px] text-[var(--color-text-muted)] flex-shrink-0">
          <RelativeTime timestamp={item.createdAt} />
        </span>
      </div>

      {/* Provenance — an identifier only; the source chat is never fetched or linked. */}
      <p className="mt-1.5 flex items-center gap-1 text-[10px] text-[var(--color-text-muted)] min-w-0">
        <Lock className="w-2.5 h-2.5 flex-shrink-0" aria-hidden="true" />
        {item.sourceRef ? (
          <span className="truncate">
            Source <span className="font-mono">{item.sourceRef}</span> · from a private chat (not viewable)
          </span>
        ) : (
          <span>Source reference not provided</span>
        )}
      </p>
    </article>
  );
}

export const KnowledgeCard = memo(KnowledgeCardImpl);
