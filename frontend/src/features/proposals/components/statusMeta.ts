import { CirclePause, FilePen, Hourglass, ShieldCheck, type LucideIcon } from 'lucide-react';
import type { ProposalStatus } from '@/types';

/** Every status has its own icon + label, so it never relies on color alone. */
export const PROPOSAL_STATUS_META: Record<ProposalStatus, { label: string; icon: LucideIcon; className: string }> = {
  DRAFT: {
    label: 'Draft',
    icon: FilePen,
    className: 'bg-[var(--color-bg-hover)] text-[var(--color-text-secondary)] border-[var(--color-border)]',
  },
  PENDING_APPROVAL: {
    label: 'Awaiting approval',
    icon: Hourglass,
    className: 'bg-[var(--color-primary-muted)] text-[var(--color-primary)] border-[var(--color-primary)]/25',
  },
  CHANGES_REQUESTED: {
    label: 'Changes requested',
    icon: CirclePause,
    className: 'bg-[var(--color-error-muted)] text-[var(--color-error)] border-[var(--color-error)]/25',
  },
  COMMITTED: {
    label: 'Committed',
    icon: ShieldCheck,
    className: 'bg-[var(--color-secondary-muted)] text-[var(--color-secondary)] border-[var(--color-secondary)]/25',
  },
};
