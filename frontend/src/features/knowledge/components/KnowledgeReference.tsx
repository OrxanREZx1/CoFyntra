'use client';

import { Hash } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { KnowledgeItem } from '@/types';

interface KnowledgeReferenceProps {
  id: string;
  /** The resolved item, or undefined if it isn't in the shared knowledge. */
  item: KnowledgeItem | undefined;
  /** Opens the shared knowledge card. Never opens a private chat. */
  onOpen: (id: string) => void;
  className?: string;
}

/** A chip linking to a shared knowledge item by ID. */
export function KnowledgeReference({ id, item, onOpen, className }: KnowledgeReferenceProps) {
  const base =
    'inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md font-mono text-[10px] font-medium border whitespace-nowrap';

  if (!item) {
    return (
      <span
        className={cn(base, 'border-dashed border-[var(--color-border)] text-[var(--color-text-disabled)]', className)}
        title="This shared finding is no longer available"
      >
        <Hash className="w-2.5 h-2.5" aria-hidden="true" />
        {id} · unavailable
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={() => onOpen(id)}
      aria-label={`Show shared finding ${id}: ${item.finding}`}
      title={item.finding}
      className={cn(
        base,
        'border-[var(--color-border)] bg-[var(--color-bg-raised)] text-[var(--color-text-secondary)]',
        'hover:text-[var(--color-text-primary)] hover:border-[var(--color-border-active)] transition-colors',
        className,
      )}
    >
      <Hash className="w-2.5 h-2.5" aria-hidden="true" />
      {id}
    </button>
  );
}
