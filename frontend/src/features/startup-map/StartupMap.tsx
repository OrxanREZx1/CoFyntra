'use client';

import { ArrowRight, Map, RotateCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useCommittedState } from '@/features/committed-state/hooks/useCommittedState';
import { useSharedKnowledge } from '@/features/knowledge/hooks/useSharedKnowledge';
import { deriveStartupMapData } from './startupMapData';

function MapSection({
  title,
  children,
  className,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn('rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-base)] px-3 py-2.5', className)}>
      <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">{title}</h3>
      <div className="mt-1.5 text-[13px] leading-snug text-[var(--color-text-secondary)]">{children}</div>
    </section>
  );
}

function FindingList({ items, emptyLabel }: { items: string[]; emptyLabel: string }) {
  if (items.length === 0) return <p className="text-[var(--color-text-muted)]">{emptyLabel}</p>;
  return (
    <ul className="space-y-1">
      {items.map((item, index) => (
        <li key={`${item}-${index}`} className="line-clamp-2 wrap-anywhere">
          {item}
        </li>
      ))}
    </ul>
  );
}

export function StartupMap({ projectId }: { projectId: string }) {
  const knowledge = useSharedKnowledge(projectId);
  const committed = useCommittedState(projectId);
  const data = deriveStartupMapData(knowledge.items, committed.state);

  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      <div className="mx-auto w-full max-w-4xl p-4 sm:p-6">
        <div className="mb-5">
          <div className="flex items-center gap-2">
            <Map className="h-5 w-5 text-[var(--color-secondary)]" aria-hidden="true" />
            <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">Startup Map</h1>
          </div>
          <p className="mt-1 pl-7 text-sm text-[var(--color-text-muted)]">
            Our startup&apos;s current state and key decisions
          </p>
        </div>
        {knowledge.error && (
          <button
            type="button"
            onClick={() => void knowledge.reload()}
            className="mb-3 inline-flex items-center gap-1 text-xs font-medium text-[var(--color-secondary)] hover:underline"
          >
            <RotateCw className="h-2.5 w-2.5" aria-hidden="true" /> Retry
          </button>
        )}

        <div className="grid gap-3 md:grid-cols-2">
          <MapSection title="Problem" className="md:col-span-2">
            {knowledge.error ? (
              <p role="alert">Couldn&apos;t load shared findings.</p>
            ) : knowledge.isLoading ? (
              <p aria-busy="true">Loading…</p>
            ) : (
              <p className="line-clamp-2 wrap-anywhere">{data.problem ?? 'No shared problem captured yet.'}</p>
            )}
          </MapSection>

          <MapSection title="Target User" className="md:col-span-2">
            <p>
              <span className="font-medium text-[var(--color-text-primary)]">Approved direction: </span>
              {committed.error
                ? 'Couldn’t load approved direction.'
                : committed.isLoading
                  ? 'Loading…'
                  : data.approvedDirection ?? 'Not decided yet'}
            </p>
            <div className="mt-1.5 border-t border-[var(--color-border-subtle)] pt-1.5">
              <p className="mb-1 font-medium text-[var(--color-text-primary)]">Shared findings</p>
              {knowledge.error ? (
                <p role="alert">Couldn&apos;t load candidate segments.</p>
              ) : knowledge.isLoading ? (
                <p aria-busy="true">Loading…</p>
              ) : data.candidateSegments.length > 0 ? (
                <ul className="max-h-24 space-y-1 overflow-y-auto">
                  {data.candidateSegments.map((candidate, index) => (
                    <li key={`${candidate.label}-${index}`} className="flex items-start gap-1.5">
                      <span className="mt-1.5 h-1 w-1 flex-shrink-0 rounded-full bg-[var(--color-secondary)]" aria-hidden="true" />
                      <span className="min-w-0 flex-1 line-clamp-2 wrap-anywhere">{candidate.label}</span>
                      <span className="flex-shrink-0 rounded-full bg-[var(--color-primary-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-primary)]">
                        {candidate.status}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-[var(--color-text-muted)]">No candidate segments yet.</p>
              )}
            </div>
          </MapSection>

          <MapSection title="Market">
            {knowledge.error ? (
              <p role="alert">Couldn&apos;t load market findings.</p>
            ) : knowledge.isLoading ? (
              <p aria-busy="true">Loading…</p>
            ) : (
              <FindingList items={data.marketFindings} emptyLabel="No market or region selected yet." />
            )}
          </MapSection>

          <MapSection title="Assumptions">
            {knowledge.error ? (
              <p role="alert">Couldn&apos;t load assumptions.</p>
            ) : knowledge.isLoading ? (
              <p aria-busy="true">Loading…</p>
            ) : (
              <FindingList items={data.assumptions} emptyLabel="No assumptions captured yet." />
            )}
          </MapSection>

          <MapSection title="Evidence">
            {knowledge.error ? (
              <p role="alert">Couldn&apos;t load evidence.</p>
            ) : knowledge.isLoading ? (
              <p aria-busy="true">Loading…</p>
            ) : (
              <FindingList items={data.evidence} emptyLabel="No supporting evidence captured yet." />
            )}
          </MapSection>

          <section className="rounded-lg border border-[var(--color-secondary)]/35 bg-[var(--color-secondary-muted)] px-3 py-3 md:col-span-2">
            <div className="flex items-center gap-1">
              <ArrowRight className="h-3 w-3 text-[var(--color-secondary)]" aria-hidden="true" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--color-secondary)]">Next Action</h3>
              {data.nextAction && (
                <span className="ml-auto rounded-full bg-[var(--color-bg-base)]/70 px-1 py-0.5 text-[9px] text-[var(--color-secondary)]">
                  {data.nextAction.status}
                </span>
              )}
            </div>
            <p className="mt-1.5 text-sm font-medium leading-snug text-[var(--color-text-primary)] wrap-anywhere">
              {committed.error
                ? 'Couldn’t load the next action.'
                : committed.isLoading
                  ? 'Loading…'
                  : data.nextAction?.text ?? 'No approved next action yet.'}
            </p>
            {committed.error && (
              <button
                type="button"
                onClick={() => void committed.reload()}
                className="mt-1 inline-flex items-center gap-1 text-[10px] font-medium text-[var(--color-secondary)] hover:underline"
              >
                <RotateCw className="h-2.5 w-2.5" aria-hidden="true" /> Retry
              </button>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
