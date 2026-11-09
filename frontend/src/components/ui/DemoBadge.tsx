'use client';

import { cn } from '@/lib/utils';

interface DemoBadgeProps {
  className?: string;
  compact?: boolean;
}

export function DemoBadge({ className, compact = false }: DemoBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-medium',
        'bg-[var(--color-warning-muted)] text-[var(--color-warning)] border border-[var(--color-warning)]/20',
        className,
      )}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-warning)] demo-pulse flex-shrink-0" />
      {!compact && 'Demo Mode'}
    </span>
  );
}

