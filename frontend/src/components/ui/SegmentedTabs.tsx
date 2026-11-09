'use client';

import { useRef, type KeyboardEvent } from 'react';
import type { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface SegmentedTab<T extends string> {
  id: T;
  label: string;
  icon: LucideIcon;
}

interface SegmentedTabsProps<T extends string> {
  tabs: SegmentedTab<T>[];
  value: T;
  onChange: (id: T) => void;
  ariaLabel: string;
  className?: string;
}

/** Accessible segmented tab bar (ARIA tabs pattern, arrow-key navigation). */
export function SegmentedTabs<T extends string>({ tabs, value, onChange, ariaLabel, className }: SegmentedTabsProps<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const handleKeyDown = (e: KeyboardEvent, index: number) => {
    const delta = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    const target = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (index + delta + tabs.length) % tabs.length;
    if (delta === 0 && e.key !== 'Home' && e.key !== 'End') return;
    e.preventDefault();
    onChange(tabs[target].id);
    refs.current[target]?.focus();
  };

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        'flex gap-0.5 p-0.5 rounded-lg bg-[var(--color-bg-raised)] border border-[var(--color-border)]',
        className,
      )}
    >
      {tabs.map(({ id, label, icon: Icon }, i) => {
        const selected = id === value;
        return (
          <button
            key={id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            aria-selected={selected}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(id)}
            onKeyDown={(e) => handleKeyDown(e, i)}
            className={cn(
              'flex-1 min-w-0 flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-md text-xs font-medium transition-colors',
              selected
                ? 'bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] shadow-sm'
                : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]',
            )}
          >
            <Icon className="w-3.5 h-3.5 flex-shrink-0" aria-hidden="true" />
            <span className="truncate">{label}</span>
          </button>
        );
      })}
    </div>
  );
}
