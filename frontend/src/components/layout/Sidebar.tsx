'use client';

import { ChevronRight, FlaskConical, LogOut, Map, MessageSquare, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { DemoBadge } from '@/components/ui/DemoBadge';
import { FounderAvatar } from '@/components/ui/FounderAvatar';
import { Logo } from '@/components/ui/Logo';
import { ThemeToggle } from '@/features/theme/ThemeToggle';
import type { DemoSession } from '@/features/auth/demoSession';
import type { DemoProfile } from '@/features/onboarding/lib/onboarding';
import { MOCK_PROJECT, FOUNDERS } from '@/mocks';
import type { NavSection, FounderId } from '@/types';

interface SidebarProps {
  activeSection: NavSection;
  onSectionChange: (section: NavSection) => void;
  activeFounder: FounderId;
  onFounderChange: (founder: FounderId) => void;
  isOpen: boolean;
  onClose: () => void;
  session: DemoSession;
  profile: DemoProfile;
  onExitDemo: () => void;
}

const NAV_ITEMS: { id: NavSection; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'startup-map', label: 'Startup Map', icon: Map },
  { id: 'ai-workspace', label: 'AI Workspace', icon: MessageSquare },
];

function initials(name: string) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]!.toUpperCase())
    .join('');
}

export function Sidebar({
  activeSection,
  onSectionChange,
  activeFounder,
  onFounderChange,
  isOpen,
  onClose,
  session,
  profile,
  onExitDemo,
}: SidebarProps) {
  const projectName = profile.projectName ?? 'Your startup';
  const userName = session.displayName ?? 'Guest';

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-20 bg-[var(--color-overlay)] lg:hidden" onClick={onClose} aria-hidden="true" />
      )}

      <aside
        className={cn(
          'fixed top-0 left-0 z-30 h-full w-60 flex flex-col',
          'bg-[var(--color-surface-secondary)] border-r border-[var(--color-border)]',
          'transition-transform duration-300 ease-in-out',
          'lg:relative lg:translate-x-0 lg:z-auto',
          isOpen ? 'translate-x-0' : '-translate-x-full',
        )}
        aria-label="Workspace navigation"
      >
        {/* Logo */}
        <div className="flex items-center justify-between px-4 h-14 border-b border-[var(--color-border)] flex-shrink-0">
          <Logo href={null} size="sm" />
          <button
            onClick={onClose}
            className="lg:hidden p-1 rounded-md text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)]"
            aria-label="Close sidebar"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Project (personalized from onboarding) */}
        <div className="px-4 py-3 border-b border-[var(--color-border)] flex-shrink-0">
          <p className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-text-muted)] mb-1">Project</p>
          <p className="text-sm font-semibold text-[var(--color-text-primary)] truncate" title={projectName}>
            {projectName}
          </p>
          <p className="text-[11px] text-[var(--color-text-secondary)] line-clamp-2 leading-snug" title={profile.idea}>
            {profile.idea}
          </p>
          <p
            className="mt-2 inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[10px] text-[var(--color-text-secondary)] bg-[var(--color-bg-base)]/70 border border-[var(--color-border)]"
            title={`The workspace is pre-filled with sample data from a fictional project (${MOCK_PROJECT.name}) so you can explore. Your onboarding answers only label this demo; they aren't sent anywhere.`}
          >
            <FlaskConical className="w-3 h-3" aria-hidden="true" />
            Sample data: {MOCK_PROJECT.name}
          </p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
          <p className="px-2 pb-1.5 text-[10px] font-medium text-[var(--color-text-muted)] uppercase tracking-wider">Workspace</p>
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => {
                onSectionChange(id);
                onClose();
              }}
              aria-current={activeSection === id ? 'page' : undefined}
              className={cn(
                'w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-all',
                activeSection === id
                  ? 'bg-[var(--color-bg-base)] text-[var(--color-text-primary)] font-medium shadow-[var(--shadow-card)]'
                  : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-base)]/60 hover:text-[var(--color-text-primary)]',
              )}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span className="flex-1 text-left">{label}</span>
              {activeSection === id && <ChevronRight className="w-3.5 h-3.5 opacity-50" />}
            </button>
          ))}
        </nav>

        {/* Demo founder switcher */}
        <div className="px-2 py-2 border-t border-[var(--color-border)] flex-shrink-0">
          <p className="px-2 pb-1.5 text-[10px] font-medium text-[var(--color-text-muted)] uppercase tracking-wider">
            View as (demo founders)
          </p>
          <div className="space-y-0.5">
            {(['founder-a', 'founder-b'] as FounderId[]).map((id) => (
              <button
                key={id}
                onClick={() => {
                  onFounderChange(id);
                  onClose();
                }}
                aria-pressed={activeFounder === id}
                className={cn(
                  'w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg text-xs transition-all',
                  activeFounder === id
                    ? 'bg-[var(--color-bg-base)] text-[var(--color-text-primary)]'
                    : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-base)]/60 hover:text-[var(--color-text-primary)]',
                )}
              >
                <FounderAvatar founderId={id} size="sm" />
                <span className="flex-1 text-left truncate">{FOUNDERS[id].name}</span>
                {activeFounder === id && <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-secondary)] flex-shrink-0" />}
              </button>
            ))}
          </div>
        </div>

        {/* Account (demo) */}
        <div className="px-3 py-3 border-t border-[var(--color-border)] flex-shrink-0 space-y-2">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-[10px] font-semibold bg-[var(--color-surface-accent)] text-[var(--color-text-primary)]"
              aria-hidden="true"
            >
              {initials(userName) || '?'}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-medium text-[var(--color-text-primary)] truncate">{userName}</p>
              <p className="text-[10px] text-[var(--color-text-muted)]" title="Demo session stored in this browser only — not a real sign-in">
                Demo session
              </p>
            </div>
            <ThemeToggle />
          </div>
          <div className="flex items-center justify-between gap-2">
            <DemoBadge />
            <button
              type="button"
              onClick={onExitDemo}
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-base)]/60"
            >
              <LogOut className="w-3 h-3" aria-hidden="true" /> Exit demo
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
