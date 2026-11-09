'use client';

import { AlertTriangle, CheckSquare, Coffee, Lightbulb, Lock, RotateCw, Scale, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Skeleton } from '@/components/ui/Skeleton';
import type { CatchUpFinding as Finding, CatchUpFindingKind, FounderId } from '@/types';
import { ConflictCard } from '@/features/knowledge/components/ConflictCard';
import { useCatchUp } from '../hooks/useCatchUp';
import { CatchUpFinding } from './CatchUpFinding';
import { CatchUpSummary } from './CatchUpSummary';

interface CatchUpPanelProps {
  projectId: string;
  founderId: FounderId;
  /** Opens a shared knowledge card — never a private chat. */
  onOpenReference: (id: string) => void;
  variant?: 'panel' | 'page';
}

const SECTIONS: { kind: CatchUpFindingKind; title: string; icon: React.ComponentType<{ className?: string }>; iconClass: string }[] = [
  { kind: 'discovery', title: 'New discoveries', icon: Sparkles, iconClass: 'text-[var(--color-secondary)]' },
  { kind: 'assumption', title: 'Assumptions to examine', icon: Lightbulb, iconClass: 'text-[var(--color-primary)]' },
  { kind: 'conflict', title: 'Potential conflicts', icon: Scale, iconClass: 'text-[var(--color-warning)]' },
];

function SectionHeading({ icon: Icon, title, count, iconClass }: { icon: React.ComponentType<{ className?: string }>; title: string; count: number; iconClass: string }) {
  return (
    <h3 className="flex items-center gap-1.5 mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
      <Icon className={cn('w-3.5 h-3.5', iconClass)} aria-hidden="true" />
      {title} · {count}
    </h3>
  );
}

export function CatchUpPanel({ projectId, founderId, onOpenReference, variant = 'panel' }: CatchUpPanelProps) {
  const catchUp = useCatchUp(projectId, founderId);
  const { summary } = catchUp;

  let body: React.ReactNode;
  if (catchUp.error) {
    body = (
      <div className="flex flex-col items-center text-center px-4 py-10" role="alert">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-3 bg-[var(--color-error-muted)]">
          <AlertTriangle className="w-5 h-5 text-[var(--color-error)]" />
        </div>
        <p className="text-sm font-medium text-[var(--color-text-primary)] mb-1">Couldn&apos;t prepare your catch-up</p>
        <p className="text-xs text-[var(--color-text-muted)] max-w-xs">{catchUp.error.userMessage}</p>
        <button
          type="button"
          onClick={() => void catchUp.refresh()}
          className="mt-4 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-[var(--color-text-primary)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] hover:border-[var(--color-border-active)] transition-colors"
        >
          <RotateCw className="w-3.5 h-3.5" /> Try again
        </button>
      </div>
    );
  } else if (catchUp.isLoading || !summary) {
    body = (
      <div className="space-y-3" aria-busy="true" aria-label="Preparing catch-up">
        <Skeleton className="h-10 w-2/3" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-20 w-full rounded-lg" />
        <Skeleton className="h-20 w-full rounded-lg" />
      </div>
    );
  } else {
    const byKind = (kind: CatchUpFindingKind) => summary.findings.filter((f) => f.kind === kind);
    const isEmpty = summary.findings.length === 0;

    body = (
      <div className="space-y-5">
        <CatchUpSummary
          summary={summary}
          isDemo={catchUp.mode === 'demo'}
          isRefreshing={catchUp.isRefreshing}
          onRefresh={() => void catchUp.refresh()}
        />

        {isEmpty ? (
          <div className="flex flex-col items-center text-center py-8">
            <Coffee className="w-6 h-6 mb-2 text-[var(--color-text-muted)]" aria-hidden="true" />
            <p className="text-sm font-medium text-[var(--color-text-primary)]">You&apos;re all caught up</p>
            <p className="text-xs text-[var(--color-text-muted)] max-w-xs mt-1">
              {summary.coFounderName} hasn&apos;t shared any new findings yet. New discoveries will show up here.
            </p>
          </div>
        ) : (
          SECTIONS.map(({ kind, title, icon, iconClass }) => {
            const findings = byKind(kind);
            if (findings.length === 0) return null;
            return (
              <section key={kind} aria-label={title}>
                <SectionHeading icon={icon} title={title} count={findings.length} iconClass={iconClass} />
                {kind === 'conflict' ? (
                  <div className="space-y-2">
                    {findings.map((f) => (
                      <ConflictForFinding
                        key={f.id}
                        finding={f}
                        catchUp={catchUp}
                        founderId={founderId}
                        onOpenReference={onOpenReference}
                        variant={variant}
                      />
                    ))}
                  </div>
                ) : (
                  <ul className="space-y-2">
                    {findings.map((f) => (
                      <CatchUpFinding key={f.id} finding={f} itemsById={catchUp.itemsById} onOpenReference={onOpenReference} />
                    ))}
                  </ul>
                )}
              </section>
            );
          })
        )}

        {summary.pendingProposals && summary.pendingProposals.length > 0 && (
          <section aria-label="Awaiting your review">
            <SectionHeading
              icon={CheckSquare}
              title="Awaiting your review"
              count={summary.pendingProposals.length}
              iconClass="text-[var(--color-text-secondary)]"
            />
            <ul className="space-y-1.5">
              {summary.pendingProposals.map((p) => (
                <li
                  key={p.id}
                  className="flex items-start gap-2 p-2.5 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border)]"
                >
                  <span className="font-mono text-[10px] text-[var(--color-text-disabled)] mt-0.5">{p.id}</span>
                  <span className="text-xs text-[var(--color-text-primary)] wrap-anywhere">{p.title}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain px-4 py-4">{body}</div>
      <p className="flex-shrink-0 flex items-center justify-center gap-1.5 px-4 py-2.5 border-t border-[var(--color-border)] text-[10px] text-[var(--color-text-disabled)] text-center">
        <Lock className="w-2.5 h-2.5 flex-shrink-0" aria-hidden="true" />
        Built only from shared findings — never from your co-founder&apos;s private chat.
      </p>
    </div>
  );
}

/** Shows a conflict finding as a full ConflictCard when the relationship is known. */
function ConflictForFinding({
  finding,
  catchUp,
  founderId,
  onOpenReference,
  variant,
}: {
  finding: Finding;
  catchUp: ReturnType<typeof useCatchUp>;
  founderId: FounderId;
  onOpenReference: (id: string) => void;
  variant: 'panel' | 'page';
}) {
  const conflict = catchUp.conflicts.find((c) => finding.knowledgeItemIds.every((id) => c.itemIds.includes(id)));
  if (!conflict) {
    return (
      <ul>
        <CatchUpFinding finding={finding} itemsById={catchUp.itemsById} onOpenReference={onOpenReference} />
      </ul>
    );
  }
  return (
    <ConflictCard
      conflict={conflict}
      itemsById={catchUp.itemsById}
      currentFounderId={founderId}
      onOpenReference={onOpenReference}
      layout={variant === 'page' ? 'split' : 'stacked'}
    />
  );
}
