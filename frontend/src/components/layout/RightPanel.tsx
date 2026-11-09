'use client';

import { useCallback, useState } from 'react';
import { BookOpen, CheckSquare, Sparkles } from 'lucide-react';
import { SegmentedTabs, type SegmentedTab } from '@/components/ui/SegmentedTabs';
import { SharedKnowledgePanel } from '@/features/knowledge/components/SharedKnowledgePanel';
import { useKnowledgeNavigation } from '@/features/knowledge/hooks/useKnowledgeNavigation';
import { CatchUpPanel } from '@/features/catch-up/components/CatchUpPanel';
import { ProposalsPanel } from '@/features/proposals/components/ProposalsPanel';
import { CommittedDirection } from '@/features/committed-state/components/CommittedDirection';
import type { RightPanelTab, FounderId } from '@/types';

interface RightPanelProps {
  projectId: string;
  activeFounder: FounderId;
  initialTab?: RightPanelTab;
}

const TABS: SegmentedTab<RightPanelTab>[] = [
  { id: 'knowledge', label: 'Knowledge', icon: BookOpen },
  { id: 'catch-up', label: 'Catch Me Up', icon: Sparkles },
  { id: 'decisions', label: 'Decisions', icon: CheckSquare },
];

export function RightPanel({ projectId, activeFounder, initialTab = 'knowledge' }: RightPanelProps) {
  const [activeTab, setActiveTab] = useState<RightPanelTab>(initialTab);
  const showKnowledge = useCallback(() => setActiveTab('knowledge'), []);
  const nav = useKnowledgeNavigation(projectId, showKnowledge);

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Official committed direction — always visible, distinct from proposals */}
      <CommittedDirection projectId={projectId} compact />

      <div className="px-4 flex-shrink-0">
        <SegmentedTabs tabs={TABS} value={activeTab} onChange={setActiveTab} ariaLabel="Intelligence panel" />
      </div>

      <div role="tabpanel" className="flex-1 min-h-0 overflow-hidden mt-2">
        {activeTab === 'knowledge' && (
          <SharedKnowledgePanel
            projectId={projectId}
            founderId={activeFounder}
            filters={nav.filters}
            onFiltersChange={nav.setFilters}
            focusRequest={nav.focusRequest}
            onFocusHandled={nav.clearFocusRequest}
            onOpenReference={nav.openReference}
            onCatchMeUp={() => setActiveTab('catch-up')}
          />
        )}
        {activeTab === 'catch-up' && (
          <CatchUpPanel projectId={projectId} founderId={activeFounder} onOpenReference={nav.openReference} />
        )}
        {activeTab === 'decisions' && (
          <ProposalsPanel projectId={projectId} founderId={activeFounder} onOpenReference={nav.openReference} />
        )}
      </div>
    </div>
  );
}
