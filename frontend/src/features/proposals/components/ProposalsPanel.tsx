'use client';

import { useCallback, useMemo, useState } from 'react';
import { AlertTriangle, Plus, RotateCw, Scale, Users } from 'lucide-react';
import { Skeleton } from '@/components/ui/Skeleton';
import type { DecisionProposal, FounderId, ProposalContent } from '@/types';
import { useSharedKnowledge } from '@/features/knowledge/hooks/useSharedKnowledge';
import { useProposalActions, useProposals } from '../hooks/useProposals';
import { CreateProposalDialog, type ProposalDialogMode } from './CreateProposalDialog';
import { ProposalCard } from './ProposalCard';
import { groupProposals, ProposalList } from './ProposalList';
import { RequestChangesDialog } from './RequestChangesDialog';

interface ProposalsPanelProps {
  projectId: string;
  founderId: FounderId;
  /** Opens a shared knowledge card. */
  onOpenReference: (id: string) => void;
  variant?: 'panel' | 'page';
}

export function ProposalsPanel({ projectId, founderId, onOpenReference, variant = 'panel' }: ProposalsPanelProps) {
  const { proposals, isLoading, error, reload, mode } = useProposals(projectId);
  const knowledge = useSharedKnowledge(projectId);
  const actions = useProposalActions(projectId, founderId);

  // Dialogs hold a snapshot of the proposal as the founder saw it, so an
  // edit made against an outdated version is rejected rather than applied.
  const [dialog, setDialog] = useState<ProposalDialogMode | null>(null);
  const [changesFor, setChangesFor] = useState<DecisionProposal | null>(null);

  const groups = useMemo(() => groupProposals(proposals, founderId), [proposals, founderId]);

  const { approve, submit } = actions;
  const handleApprove = useCallback((p: DecisionProposal) => void approve(p), [approve]);
  const handleSubmitDraft = useCallback((p: DecisionProposal) => void submit(p), [submit]);
  const handleRevise = useCallback(
    (p: DecisionProposal) => setDialog(p.status === 'DRAFT' ? { kind: 'edit-draft', proposal: p } : { kind: 'revise', proposal: p }),
    [],
  );

  const handleDialogSubmit = async (content: ProposalContent, opts: { submit: boolean; changeNote: string | null }) => {
    if (!dialog) return false;
    if (dialog.kind === 'create') return (await actions.create(content, opts.submit))?.ok ?? false;
    const revised = await actions.revise(dialog.proposal, content, dialog.kind === 'revise' ? opts.changeNote : null);
    if (!revised?.ok) return false;
    if (dialog.kind === 'edit-draft' && opts.submit) return (await actions.submit(revised.result.proposal))?.ok ?? false;
    return true;
  };

  const dialogBusy = dialog ? Boolean(actions.busy[dialog.kind === 'create' ? 'new' : dialog.proposal.id]) : false;

  let body: React.ReactNode;
  if (error) {
    body = (
      <div className="flex flex-col items-center text-center px-4 py-10" role="alert">
        <AlertTriangle className="w-6 h-6 mb-2 text-[var(--color-error)]" aria-hidden="true" />
        <p className="text-sm font-medium text-[var(--color-text-primary)]">Couldn&apos;t load proposals</p>
        <p className="text-xs text-[var(--color-text-muted)] mt-1">{error.userMessage}</p>
        <button
          type="button"
          onClick={() => void reload()}
          className="mt-4 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-[var(--color-border)] bg-[var(--color-bg-elevated)] hover:border-[var(--color-border-active)]"
        >
          <RotateCw className="w-3.5 h-3.5" /> Try again
        </button>
      </div>
    );
  } else if (isLoading) {
    body = (
      <div className="space-y-3" aria-busy="true" aria-label="Loading proposals">
        <Skeleton className="h-56 w-full rounded-xl" />
        <Skeleton className="h-32 w-full rounded-xl" />
      </div>
    );
  } else if (groups.length === 0) {
    body = (
      <div className="flex flex-col items-center text-center px-4 py-10">
        <Scale className="w-6 h-6 mb-2 text-[var(--color-text-muted)]" aria-hidden="true" />
        <p className="text-sm font-medium text-[var(--color-text-primary)]">No proposals yet</p>
        <p className="text-xs text-[var(--color-text-muted)] mt-1 max-w-xs">
          When you&apos;re ready to commit to a direction, propose it here. Your co-founder reviews the same version.
        </p>
      </div>
    );
  } else {
    body = (
      <ProposalList
        groups={groups}
        renderProposal={(p) => (
          <ProposalCard
            key={p.id}
            proposal={p}
            founderId={founderId}
            itemsById={knowledge.itemsById}
            knowledgeConflicts={knowledge.conflicts}
            busyAction={actions.busy[p.id]}
            isDemo={mode === 'demo'}
            layout={variant === 'page' ? 'split' : 'stacked'}
            onApprove={handleApprove}
            onRequestChanges={setChangesFor}
            onRevise={handleRevise}
            onSubmitDraft={handleSubmitDraft}
            onOpenReference={onOpenReference}
          />
        )}
      />
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-start gap-2 px-4 pt-3 pb-2.5 border-b border-[var(--color-border)] flex-shrink-0">
        <Users className="w-3.5 h-3.5 mt-0.5 text-[var(--color-secondary)] flex-shrink-0" aria-hidden="true" />
        <p className="flex-1 text-[11px] leading-snug text-[var(--color-text-secondary)]">
          <span className="font-medium text-[var(--color-text-primary)]">Think independently, decide together.</span> A
          proposal becomes official only when both founders approve the same version.
          {mode === 'demo' && (
            <span className="text-[var(--color-text-muted)]"> Demo: saved in this browser only, not synced between devices.</span>
          )}
        </p>
        <button
          type="button"
          onClick={() => setDialog({ kind: 'create' })}
          className="flex-shrink-0 flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium text-[var(--color-on-primary)] bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] transition-colors"
        >
          <Plus className="w-3 h-3" aria-hidden="true" /> New proposal
        </button>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain px-4 py-3">{body}</div>

      {dialog && (
        <CreateProposalDialog
          key={dialog.kind === 'create' ? 'create' : `${dialog.kind}-${dialog.proposal.id}-${dialog.proposal.version}`}
          mode={dialog}
          items={knowledge.items}
          knowledgeConflicts={knowledge.conflicts}
          isSubmitting={dialogBusy}
          onSubmit={handleDialogSubmit}
          onClose={() => setDialog(null)}
        />
      )}
      {changesFor && (
        <RequestChangesDialog
          proposal={changesFor}
          isSubmitting={Boolean(actions.busy[changesFor.id])}
          onSubmit={async (reason) => (await actions.requestChanges(changesFor, reason))?.ok ?? false}
          onClose={() => setChangesFor(null)}
        />
      )}
    </div>
  );
}
