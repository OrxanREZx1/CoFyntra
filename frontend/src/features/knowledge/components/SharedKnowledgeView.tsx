'use client';

import { useCallback, useState } from 'react';
import { BookOpen, CheckSquare, Sparkles } from 'lucide-react';
import { DemoBadge } from '@/components/ui/DemoBadge';
import { SegmentedTabs, type SegmentedTab } from '@/components/ui/SegmentedTabs';
import type { FounderId } from '@/types';
import { CatchUpPanel } from '@/features/catch-up/components/CatchUpPanel';
import { CommittedDirection } from '@/features/committed-state/components/CommittedDirection';
import { ProposalsPanel } from '@/features/proposals/components/ProposalsPanel';
import { BACKEND_MODE } from '@/lib/config';
import { useKnowledgeNavigation } from '../hooks/useKnowledgeNavigation';
import { SharedKnowledgePanel } from './SharedKnowledgePanel';

export type SharedViewTab = 'knowledge' | 'catch-up' | 'decisions';

const TABS: SegmentedTab<SharedViewTab>[] = [
  { id: 'knowledge', label: 'Knowledge', icon: BookOpen },
  { id: 'catch-up', label: 'Catch Me Up', icon: Sparkles },
  { id: 'decisions', label: 'Decisions', icon: CheckSquare },
];

interface SharedKnowledgeViewProps {
  projectId: string;
  founderId: FounderId;
  initialTab?: SharedViewTab;
}

/**
 * Full-width shared workspace: knowledge, catch-up and decisions. Used as the expanded view on
 * desktop and as the dedicated view on mobile (instead of squeezing the
 * desktop side panel into a narrow column).
 */
export function SharedKnowledgeView({ projectId, founderId, initialTab = 'knowledge' }: SharedKnowledgeViewProps) {
  const [tab, setTab] = useState<SharedViewTab>(initialTab);
  const showKnowledge = useCallback(() => setTab('knowledge'), []);
  const nav = useKnowledgeNavigation(projectId, showKnowledge);

  return (
    <section className="flex flex-col h-full min-h-0" aria-label="Shared knowledge">
      <header className="flex items-center justify-between gap-3 px-4 h-14 flex-shrink-0 border-b border-[var(--color-border)]">
        <div className="min-w-0">
          <h1 className="text-sm font-semibold text-[var(--color-text-primary)]">Shared Workspace</h1>
          <p className="text-[10px] text-[var(--color-text-muted)] truncate">
            Connects what you both learn — not your private conversations
          </p>
        </div>
        {BACKEND_MODE === 'demo' && <DemoBadge className="flex-shrink-0 whitespace-nowrap" />}
      </header>

      <div className="w-full max-w-4xl mx-auto px-4 pt-3 flex-shrink-0">
        <SegmentedTabs tabs={TABS} value={tab} onChange={setTab} ariaLabel="Shared knowledge views" />
      </div>

      <div role="tabpanel" className="flex-1 min-h-0 w-full max-w-4xl mx-auto mt-1">
        {tab === 'knowledge' ? (
          <SharedKnowledgePanel
            projectId={projectId}
            founderId={founderId}
            filters={nav.filters}
            onFiltersChange={nav.setFilters}
            focusRequest={nav.focusRequest}
            onFocusHandled={nav.clearFocusRequest}
            onOpenReference={nav.openReference}
            onCatchMeUp={() => setTab('catch-up')}
            variant="page"
          />
        ) : tab === 'catch-up' ? (
          <CatchUpPanel projectId={projectId} founderId={founderId} onOpenReference={nav.openReference} variant="page" />
        ) : (
          <div className="flex flex-col h-full min-h-0">
            <CommittedDirection projectId={projectId} />
            <div className="flex-1 min-h-0">
              <ProposalsPanel projectId={projectId} founderId={founderId} onOpenReference={nav.openReference} variant="page" />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
