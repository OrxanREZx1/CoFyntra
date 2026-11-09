'use client';

import { cn } from '@/lib/utils';
import type { FounderId } from '@/types';
import { FOUNDERS } from '@/mocks';

interface FounderAvatarProps {
  founderId: FounderId;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeMap = {
  sm: 'w-6 h-6 text-[10px]',
  md: 'w-8 h-8 text-xs',
  lg: 'w-10 h-10 text-sm',
};

export function FounderAvatar({ founderId, size = 'md', className }: FounderAvatarProps) {
  const founder = FOUNDERS[founderId];
  if (!founder) return null;

  return (
    <div
      className={cn(
        'rounded-full flex items-center justify-center font-semibold flex-shrink-0 ring-1 ring-[var(--color-border)]',
        sizeMap[size],
        className,
      )}
      style={{ backgroundColor: `color-mix(in srgb, ${founder.color} 16%, transparent)`, color: founder.color }}
      title={founder.name}
      aria-label={founder.name}
    >
      {founder.avatar}
    </div>
  );
}

