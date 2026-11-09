import { ClipboardCheck, Flame, Puzzle, TrendingUp, Users, type LucideIcon } from 'lucide-react';
import type { KnowledgeCategory } from '@/types';

export const CATEGORY_META: Record<KnowledgeCategory, { icon: LucideIcon; className: string }> = {
  Problem: {
    icon: Flame,
    className: 'bg-[var(--color-error-muted)] text-[var(--color-error)] border-[var(--color-error)]/20',
  },
  'Target Customer': {
    icon: Users,
    className: 'bg-[var(--color-primary-muted)] text-[var(--color-primary)] border-[var(--color-primary)]/20',
  },
  Market: {
    icon: TrendingUp,
    className: 'bg-[var(--color-secondary-muted)] text-[var(--color-secondary)] border-[var(--color-secondary)]/20',
  },
  Assumption: {
    icon: Puzzle,
    className: 'bg-[var(--color-warning-muted)] text-[var(--color-warning)] border-[var(--color-warning)]/20',
  },
  Evidence: {
    icon: ClipboardCheck,
    className: 'bg-[var(--color-success-muted)] text-[var(--color-success)] border-[var(--color-success)]/20',
  },
};
