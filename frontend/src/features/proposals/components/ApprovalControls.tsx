'use client';

import { Check, CheckCircle2, CirclePause, Pencil, Send, ShieldCheck, UserRound } from 'lucide-react';
import { cn } from '@/lib/utils';
import { FOUNDERS } from '@/mocks';
import type { DecisionProposal, FounderId } from '@/types';
import type { ProposalAction } from '../hooks/useProposals';
import { decisionFor } from '../lib/proposalMachine';

interface ApprovalControlsProps {
  proposal: DecisionProposal;
  founderId: FounderId;
  busyAction?: ProposalAction;
  isDemo: boolean;
  onApprove: () => void;
  onRequestChanges: () => void;
  onRevise: () => void;
  onSubmitDraft: () => void;
}

const btn = 'inline-flex items-center justify-center gap-1.5 h-8 px-3 rounded-lg text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed';
const primary = cn(btn, 'bg-[var(--color-primary)] text-[var(--color-on-primary)] hover:bg-[var(--color-primary-hover)]');
const secondary = cn(
  btn,
  'border border-[var(--color-border)] bg-[var(--color-bg-raised)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-border-active)]',
);

function Spinner() {
  return <span className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" aria-hidden="true" />;
}

function Note({ icon: Icon, tone, children }: { icon: React.ComponentType<{ className?: string }>; tone: 'success' | 'info' | 'paused'; children: React.ReactNode }) {
  return (
    <p
      className={cn(
        'flex items-start gap-1.5 text-xs leading-relaxed',
        tone === 'success' && 'text-[var(--color-success)]',
        tone === 'info' && 'text-[var(--color-text-secondary)]',
        tone === 'paused' && 'text-[var(--color-error)]',
      )}
    >
      <Icon className="w-3.5 h-3.5 mt-px flex-shrink-0" />
      <span>{children}</span>
    </p>
  );
}

/** What the current founder can do with this proposal right now. */
export function ApprovalControls({
  proposal,
  founderId,
  busyAction,
  isDemo,
  onApprove,
  onRequestChanges,
  onRevise,
  onSubmitDraft,
}: ApprovalControlsProps) {
  const isProposer = proposal.createdByFounderId === founderId;
  const mine = decisionFor(proposal, founderId);
  const busy = Boolean(busyAction);
  const proposerName = FOUNDERS[proposal.createdByFounderId]?.name ?? 'Your co-founder';

  let content: React.ReactNode;
  switch (proposal.status) {
    case 'COMMITTED':
      content = (
        <Note icon={ShieldCheck} tone="success">
          Decision confirmed by both founders on version {proposal.version}.
        </Note>
      );
      break;

    case 'DRAFT':
      content = isProposer ? (
        <div className="space-y-2">
          <Note icon={Pencil} tone="info">
            Draft — your co-founder can&apos;t act on it until you submit it.
          </Note>
          <div className="flex flex-wrap gap-2">
            <button type="button" className={primary} onClick={onSubmitDraft} disabled={busy}>
              {busyAction === 'submit' ? <Spinner /> : <Send className="w-3.5 h-3.5" />} Submit for approval
            </button>
            <button type="button" className={secondary} onClick={onRevise} disabled={busy}>
              <Pencil className="w-3.5 h-3.5" /> Edit draft
            </button>
          </div>
        </div>
      ) : (
        <Note icon={Pencil} tone="info">
          Draft — {proposerName} hasn&apos;t submitted it yet.
        </Note>
      );
      break;

    case 'CHANGES_REQUESTED':
      content = (
        <div className="space-y-2">
          <Note icon={CirclePause} tone="paused">
            Changes requested. Approval is paused
            {isProposer ? ' — publish a revised version to continue.' : ` until ${proposerName} publishes a revised version.`}
          </Note>
          {isProposer && (
            <button type="button" className={primary} onClick={onRevise} disabled={busy}>
              {busyAction === 'revise' ? <Spinner /> : <Pencil className="w-3.5 h-3.5" />} Revise proposal
            </button>
          )}
        </div>
      );
      break;

    case 'PENDING_APPROVAL':
      content =
        mine?.decision === 'APPROVED' ? (
          <div className="space-y-2">
            <Note icon={CheckCircle2} tone="success">
              You approved version {proposal.version}. Waiting for your co-founder to approve.
            </Note>
            <div className="flex flex-wrap gap-2">
              <button type="button" className={secondary} onClick={onRequestChanges} disabled={busy}>
                {busyAction === 'request-changes' ? <Spinner /> : <CirclePause className="w-3.5 h-3.5" />} Request changes
              </button>
              {isProposer && (
                <button type="button" className={secondary} onClick={onRevise} disabled={busy}>
                  <Pencil className="w-3.5 h-3.5" /> Revise
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-xs text-[var(--color-text-secondary)]">
              Both founders must approve this version before it becomes official.
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                className={cn(primary, 'flex-1 sm:flex-none')}
                onClick={onApprove}
                disabled={busy}
                aria-label={`Approve version ${proposal.version} of ${proposal.title}`}
              >
                {busyAction === 'approve' ? <Spinner /> : <Check className="w-3.5 h-3.5" />} Approve v{proposal.version}
              </button>
              <button type="button" className={cn(secondary, 'flex-1 sm:flex-none')} onClick={onRequestChanges} disabled={busy}>
                {busyAction === 'request-changes' ? <Spinner /> : <CirclePause className="w-3.5 h-3.5" />} Request changes
              </button>
              {isProposer && (
                <button type="button" className={secondary} onClick={onRevise} disabled={busy}>
                  <Pencil className="w-3.5 h-3.5" /> Revise
                </button>
              )}
            </div>
          </div>
        );
      break;
  }

  return (
    <div className="space-y-2">
      {content}
      {isDemo && proposal.status !== 'COMMITTED' && (
        <p className="flex items-center gap-1 text-[10px] text-[var(--color-text-disabled)]">
          <UserRound className="w-2.5 h-2.5" aria-hidden="true" />
          Acting as {FOUNDERS[founderId]?.name} — demo identity, not a real sign-in
        </p>
      )}
    </div>
  );
}
