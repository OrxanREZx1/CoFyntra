'use client';

import { useState } from 'react';
import { Menu, PanelRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Sidebar } from '@/components/layout/Sidebar';
import { RightPanel } from '@/components/layout/RightPanel';
import { ChatPanel } from '@/features/chat/components/ChatPanel';
import { DemoBadge } from '@/components/ui/DemoBadge';
import { SharedKnowledgeView, type SharedViewTab } from '@/features/knowledge/components/SharedKnowledgeView';
import { StartupMap } from '@/features/startup-map/StartupMap';
import { MOCK_PROJECT } from '@/mocks';
import type { DemoSession } from '@/features/auth/demoSession';
import type { DemoProfile } from '@/features/onboarding/lib/onboarding';
import type { NavSection, FounderId } from '@/types';

interface WorkspaceShellProps {
  session: DemoSession;
  profile: DemoProfile;
  onExitDemo: () => void;
}

/**
 * Client-only: rendered after the demo route gate resolves, so reading the
 * viewport in initial state can't cause a hydration mismatch.
 */
export function WorkspaceShell({ session, profile, onExitDemo }: WorkspaceShellProps) {
  const [activeSection, setActiveSection] = useState<NavSection>('ai-workspace');
  const [activeFounder, setActiveFounder] = useState<FounderId>('founder-a');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Phones start with the center view; the side panel starts open from tablet width up.
  const [rightPanelOpen, setRightPanelOpen] = useState(() => window.matchMedia('(min-width: 640px)').matches);

  // Sidebar sections that open the full shared knowledge view in the main area.
  const sharedView: SharedViewTab | null =
    activeSection === 'shared-knowledge'
      ? 'knowledge'
      : activeSection === 'team-sync'
        ? 'catch-up'
        : activeSection === 'decisions'
          ? 'decisions'
          : null;

  return (
    <div className="flex h-full w-full overflow-hidden bg-[var(--color-bg-base)]">
      {/* Sidebar */}
      <Sidebar
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        activeFounder={activeFounder}
        onFounderChange={(f) => {
          setActiveFounder(f);
        }}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        session={session}
        profile={profile}
        onExitDemo={onExitDemo}
      />

      {/* Main area */}
      <div className="flex flex-col flex-1 min-w-0 min-h-0">
        {/* Mobile top bar */}
        <div className="flex items-center justify-between px-4 h-12 border-b border-[var(--color-border)] lg:hidden flex-shrink-0 bg-[var(--color-bg-elevated)]">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-1.5 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)]"
            aria-label="Open menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-2">
            <span className="font-display text-base font-semibold text-[var(--color-text-primary)]">
              Confyntra
            </span>
            <DemoBadge compact />
          </div>

          <button
            onClick={() => setRightPanelOpen(!rightPanelOpen)}
            className={cn(
              'p-1.5 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)]',
              sharedView && 'invisible', // the full shared view is already open
            )}
            aria-label="Toggle knowledge panel"
          >
            <PanelRight className={cn('w-5 h-5', rightPanelOpen && 'text-[var(--color-primary)]')} />
          </button>
        </div>

        {/* Content area */}
        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Center: chat panel. Stays mounted (only hidden) while the shared view
              is open, so drafts and in-flight messages are preserved. */}
          <div
            className={cn(
              'flex-col flex-1 min-w-0',
              sharedView || activeSection !== 'ai-workspace'
                ? 'hidden'
                : // On mobile when right panel is open, hide chat
                  rightPanelOpen
                  ? 'hidden sm:flex'
                  : 'flex',
            )}
          >
            {/* key: remount per founder so no private chat UI state is shared between founders */}
            <ChatPanel key={activeFounder} projectId={MOCK_PROJECT.id} founderId={activeFounder} />
          </div>

          {activeSection === 'startup-map' && !sharedView && (
            <div className={cn(
              'flex-col flex-1 min-w-0',
              rightPanelOpen ? 'hidden sm:flex' : 'flex',
            )}>
              <StartupMap projectId={MOCK_PROJECT.id} />
            </div>
          )}

          {/* Center: full shared knowledge / catch-up view (expanded on desktop, dedicated on mobile) */}
          {sharedView && (
            <div className="flex flex-col flex-1 min-w-0">
              <SharedKnowledgeView
                key={sharedView}
                projectId={MOCK_PROJECT.id}
                founderId={activeFounder}
                initialTab={sharedView}
              />
            </div>
          )}

          {/* Divider */}
          <div className={cn(
            'hidden lg:block w-px bg-[var(--color-border)] flex-shrink-0',
            rightPanelOpen && !sharedView ? 'opacity-100' : 'opacity-0',
          )} />

          {/* Right: intelligence panel (not shown alongside the full shared view) */}
          <div
            className={cn(
              'flex-col bg-[var(--color-bg-base)] transition-all duration-300 overflow-hidden',
              sharedView
                ? 'hidden'
                : rightPanelOpen
                  ? 'flex w-full sm:w-80 lg:w-96 flex-shrink-0 border-l border-[var(--color-border)]'
                  : 'flex w-0 border-0',
            )}
          >
            {rightPanelOpen && !sharedView && (
              <RightPanel projectId={MOCK_PROJECT.id} activeFounder={activeFounder} />
            )}
          </div>

          {/* Desktop right panel toggle */}
          <button
            onClick={() => setRightPanelOpen(!rightPanelOpen)}
            className={cn(
              'items-center justify-center w-5 flex-shrink-0 group transition-all',
              sharedView ? 'hidden' : 'hidden lg:flex',
              'border-l border-[var(--color-border)] bg-[var(--color-bg-base)]',
              'hover:bg-[var(--color-bg-elevated)]',
            )}
            aria-label="Toggle intelligence panel"
            title={rightPanelOpen ? 'Collapse intelligence panel' : 'Expand intelligence panel'}
          >
            <PanelRight
              className={cn(
                'w-3 h-3 text-[var(--color-text-disabled)] group-hover:text-[var(--color-text-muted)] transition-transform',
                rightPanelOpen ? 'rotate-180' : 'rotate-0',
              )}
            />
          </button>
        </div>
      </div>
    </div>
  );
}
