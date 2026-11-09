'use client';

import { CheckCircle2, Clock, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { FounderAvatar } from '@/components/ui/FounderAvatar';
import { RelativeTime } from '@/components/ui/RelativeTime';
import { FOUNDERS } from '@/mocks';
import type { DecisionProposal, FounderId } from '@/types';
import { decisionFor, FOUNDER_IDS } from '../lib/proposalMachine';

/** Each founder's decision on the CURRENT version — two independent records. */
export function ApprovalStatus({ proposal, currentFounderId }: { proposal: DecisionProposal; currentFounderId: FounderId }) {
  const approvedCount = FOUNDER_IDS.filter((f) => decisionFor(proposal, f)?.decision === 'APPROVED').length;

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
          Approvals · version {proposal.version}
        </p>
        <p className="text-[10px] text-[var(--color-text-muted)]">{approvedCount} of 2 approved</p>
      </div>
      <div
        className="flex gap-1 mb-2"
        role="img"
        aria-label={`${approvedCount} of 2 founders approved version ${proposal.version}`}
      >
        {[0, 1].map((i) => (
          <span
            key={i}
            className={cn(
              'h-1 flex-1 rounded-full transition-colors',
              i < approvedCount ? 'bg-[var(--color-success)]' : 'bg-[var(--color-bg-hover)]',
            )}
          />
        ))}
      </div>
      <ul className="space-y-1.5">
        {FOUNDER_IDS.map((founderId) => {
          const d = decisionFor(proposal, founderId);
          return (
            <li key={founderId} className="flex items-center gap-2 min-w-0">
              <FounderAvatar founderId={founderId} size="sm" />
              <span className="text-xs text-[var(--color-text-secondary)] truncate flex-1">
                {FOUNDERS[founderId]?.name}
                {founderId === currentFounderId && <span className="text-[var(--color-text-muted)]"> (you)</span>}
              </span>
              {d?.decision === 'APPROVED' ? (
                <span className="flex items-center gap-1 text-[11px] text-[var(--color-success)] whitespace-nowrap">
                  <CheckCircle2 className="w-3.5 h-3.5" aria-hidden="true" />
                  Approved v{d.version}
                  <span className="hidden sm:inline text-[var(--color-text-muted)]">
                    · <RelativeTime timestamp={d.decidedAt} />
                  </span>
                </span>
              ) : d?.decision === 'CHANGES_REQUESTED' ? (
                <span className="flex items-center gap-1 text-[11px] text-[var(--color-error)] whitespace-nowrap">
                  <XCircle className="w-3.5 h-3.5" aria-hidden="true" />
                  Changes requested
                </span>
              ) : (
                <span className="flex items-center gap-1 text-[11px] text-[var(--color-text-muted)] whitespace-nowrap">
                  <Clock className="w-3.5 h-3.5" aria-hidden="true" />
                  Awaiting approval
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
