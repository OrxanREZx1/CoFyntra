'use client';

import { memo } from 'react';
import { ArrowRight, History, Info, MessageSquareWarning, Target } from 'lucide-react';
import { cn } from '@/lib/utils';
import { FounderAvatar } from '@/components/ui/FounderAvatar';
import { RelativeTime } from '@/components/ui/RelativeTime';
import { FOUNDERS } from '@/mocks';
import type { DecisionProposal, FounderId, KnowledgeConflict, KnowledgeItem } from '@/types';
import { FindingSummary } from '@/features/knowledge/components/ConflictCard';
import type { ProposalAction } from '../hooks/useProposals';
import { ApprovalControls } from './ApprovalControls';
import { ApprovalStatus } from './ApprovalStatus';
import { ProposalConflictReferences } from './ProposalConflictReferences';
import { PROPOSAL_STATUS_META } from './statusMeta';

export const proposalDomId = (id: string) => `proposal-${id}`;

interface ProposalCardProps {
  proposal: DecisionProposal;
  founderId: FounderId;
  itemsById: ReadonlyMap<string, KnowledgeItem>;
  knowledgeConflicts: readonly KnowledgeConflict[];
  busyAction?: ProposalAction;
  isDemo: boolean;
  layout: 'stacked' | 'split';
  onApprove: (p: DecisionProposal) => void;
  onRequestChanges: (p: DecisionProposal) => void;
  onRevise: (p: DecisionProposal) => void;
  onSubmitDraft: (p: DecisionProposal) => void;
  onOpenReference: (id: string) => void;
}

function Field({ label, icon: Icon, children }: { label: string; icon: React.ComponentType<{ className?: string }>; children: React.ReactNode }) {
  return (
    <div>
      <p className="flex items-center gap-1 mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
        <Icon className="w-3 h-3" aria-hidden="true" />
        {label}
      </p>
      <p className="text-sm text-[var(--color-text-primary)] leading-relaxed wrap-anywhere">{children}</p>
    </div>
  );
}

function ProposalCardImpl(props: ProposalCardProps) {
  const { proposal: p, founderId, itemsById, knowledgeConflicts, busyAction, isDemo, layout, onOpenReference } = props;
  const status = PROPOSAL_STATUS_META[p.status];
  const StatusIcon = status.icon;
  const proposer = FOUNDERS[p.createdByFounderId];
  const committed = p.status === 'COMMITTED';
  const changeRequests = p.approvals.filter((a) => a.decision === 'CHANGES_REQUESTED');

  return (
    <article
      id={proposalDomId(p.id)}
      aria-labelledby={`${proposalDomId(p.id)}-title`}
      className={cn(
        'rounded-xl border bg-[var(--color-bg-elevated)] overflow-hidden',
        // Proposals are visibly "proposed": dashed until committed.
        committed ? 'border-[var(--color-border)]' : 'border-dashed border-[var(--color-border-active)]/60',
        busyAction && 'opacity-80',
      )}
      aria-busy={Boolean(busyAction)}
    >
      {/* Status strip */}
      <div className="flex flex-wrap items-center gap-1.5 px-3 py-2 border-b border-[var(--color-border)]">
        <span className={cn('inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold border', status.className)}>
          <StatusIcon className="w-3 h-3" aria-hidden="true" />
          {status.label}
        </span>
        {!committed && (
          <span className="text-[10px] font-medium text-[var(--color-text-muted)]">Proposed · not yet official</span>
        )}
        <span className="ml-auto font-mono text-[10px] text-[var(--color-text-muted)]">
          {p.id} · v{p.version}
        </span>
      </div>

      <div className="p-3 space-y-3">
        {/* Title + proposer */}
        <div>
          <h3 id={`${proposalDomId(p.id)}-title`} className="text-sm font-semibold text-[var(--color-text-primary)] leading-snug wrap-anywhere">
            {p.title}
          </h3>
          <div className="mt-1 flex items-center gap-1.5 min-w-0">
            <FounderAvatar founderId={p.createdByFounderId} size="sm" />
            <span className="text-[11px] text-[var(--color-text-muted)] truncate">
              Proposed by {proposer?.name}
              {p.createdByFounderId === founderId && ' (you)'} · <RelativeTime timestamp={p.createdAt} />
              {p.version > 1 && (
                <>
                  {' '}
                  · v{p.version} <RelativeTime timestamp={p.updatedAt} />
                </>
              )}
            </span>
          </div>
        </div>

        {/* Revision notice */}
        {p.version > 1 && !committed && (
          <div className="flex items-start gap-1.5 px-2.5 py-2 rounded-lg bg-[var(--color-primary-muted)] border border-[var(--color-primary)]/20">
            <Info className="w-3.5 h-3.5 mt-px text-[var(--color-primary)] flex-shrink-0" aria-hidden="true" />
            <div className="text-[11px] leading-relaxed text-[var(--color-text-secondary)]">
              <p className="font-medium text-[var(--color-text-primary)]">
                This proposal was updated to version {p.version}. Previous approvals no longer apply.
              </p>
              {p.changeNote && <p className="mt-0.5 wrap-anywhere">What changed: {p.changeNote}</p>}
            </div>
          </div>
        )}

        <Field label={committed ? 'Approved direction' : 'Proposed direction'} icon={Target}>
          {p.direction}
        </Field>
        <Field label={committed ? 'Approved next action' : 'Proposed next action'} icon={ArrowRight}>
          {p.nextAction}
        </Field>

        {/* Change requests */}
        {changeRequests.map((cr) => (
          <div
            key={cr.founderId}
            className="flex items-start gap-1.5 px-2.5 py-2 rounded-lg bg-[var(--color-error-muted)] border border-[var(--color-error)]/20"
          >
            <MessageSquareWarning className="w-3.5 h-3.5 mt-px text-[var(--color-error)] flex-shrink-0" aria-hidden="true" />
            <p className="text-[11px] leading-relaxed text-[var(--color-text-secondary)] wrap-anywhere">
              <span className="font-medium text-[var(--color-text-primary)]">
                {FOUNDERS[cr.founderId]?.name}
                {cr.founderId === founderId && ' (you)'} requested changes to v{cr.version}:
              </span>{' '}
              “{cr.reason}”
            </p>
          </div>
        ))}

        {/* Evidence (kept collapsible once committed) */}
        {(() => {
          const evidence = (
            <div className="space-y-3">
              <section aria-label="Supporting findings" className="space-y-1.5">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
                  Supporting findings · {p.supportingKnowledgeIds.length}
                </p>
                {p.supportingKnowledgeIds.length === 0 ? (
                  <p className="text-[11px] text-[var(--color-text-disabled)]">No supporting findings referenced.</p>
                ) : (
                  p.supportingKnowledgeIds.map((id) => (
                    <FindingSummary key={id} id={id} item={itemsById.get(id)} currentFounderId={founderId} onOpenReference={onOpenReference} />
                  ))
                )}
              </section>
              <ProposalConflictReferences
                proposal={p}
                itemsById={itemsById}
                knowledgeConflicts={knowledgeConflicts}
                founderId={founderId}
                onOpenReference={onOpenReference}
                layout={layout}
              />
            </div>
          );
          return committed ? (
            <details className="group">
              <summary className="cursor-pointer text-[11px] font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]">
                Evidence considered
              </summary>
              <div className="mt-2">{evidence}</div>
            </details>
          ) : (
            evidence
          );
        })()}

        <div className="pt-1 border-t border-[var(--color-border)] space-y-3">
          <div className="pt-2">
            <ApprovalStatus proposal={p} currentFounderId={founderId} />
          </div>
          <ApprovalControls
            proposal={p}
            founderId={founderId}
            busyAction={busyAction}
            isDemo={isDemo}
            onApprove={() => props.onApprove(p)}
            onRequestChanges={() => props.onRequestChanges(p)}
            onRevise={() => props.onRevise(p)}
            onSubmitDraft={() => props.onSubmitDraft(p)}
          />
        </div>

        {/* Version history */}
        {p.history.length > 0 && (
          <details className="border-t border-[var(--color-border)] pt-2">
            <summary className="flex items-center gap-1 cursor-pointer text-[11px] text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]">
              <History className="w-3 h-3" aria-hidden="true" /> Version history ({p.history.length} earlier)
            </summary>
            <ol className="mt-2 space-y-2">
              {[...p.history].reverse().map((h) => (
                <li key={h.version} className="pl-2.5 border-l-2 border-[var(--color-border)] text-[11px] leading-relaxed">
                  <p className="text-[var(--color-text-secondary)]">
                    <span className="font-mono">v{h.version}</span> · {h.direction}
                  </p>
                  {h.approvals.length > 0 && (
                    <p className="text-[var(--color-text-muted)]">
                      {h.approvals
                        .map((a) => `${FOUNDERS[a.founderId]?.name}: ${a.decision === 'APPROVED' ? 'approved' : 'requested changes'}`)
                        .join(' · ')}{' '}
                      — cleared when v{h.version + 1} was published
                    </p>
                  )}
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
    </article>
  );
}

export const ProposalCard = memo(ProposalCardImpl);
