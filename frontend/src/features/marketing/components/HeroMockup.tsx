import { ArrowDown, Bot, Check, Lightbulb, Lock, Quote, Scale, ShieldCheck, Users } from 'lucide-react';
import { cn } from '@/lib/utils';
import { mockCard } from '../styles';

/** Small founder avatar for the illustration (decoupled from app mock data). */
function Avatar({ who, className }: { who: 'a' | 'b'; className?: string }) {
  const color = who === 'a' ? 'var(--color-founder-a)' : 'var(--color-founder-b)';
  return (
    <span
      className={cn('w-6 h-6 rounded-full flex items-center justify-center text-[9px] font-semibold flex-shrink-0 ring-1 ring-[var(--color-border)]', className)}
      style={{ color, backgroundColor: `color-mix(in srgb, ${color} 16%, transparent)` }}
    >
      {who === 'a' ? 'A' : 'B'}
    </span>
  );
}

function PrivateChat({ who, name, user, ai, shared }: { who: 'a' | 'b'; name: string; user: string; ai: string; shared: string }) {
  return (
    <div className={cn(mockCard, 'p-3.5 text-left')}>
      <div className="flex items-center gap-2 pb-2.5 border-b border-[var(--color-border-subtle)]">
        <Avatar who={who} />
        <div className="min-w-0">
          <p className="text-xs font-semibold text-[var(--color-text-primary)]">{name}&apos;s AI chat</p>
          <p className="flex items-center gap-1 text-[10px] text-[var(--color-secondary)]">
            <Lock className="w-2.5 h-2.5" aria-hidden="true" /> Private
          </p>
        </div>
      </div>
      <div className="pt-2.5 space-y-2">
        <p className="ml-6 px-2.5 py-1.5 rounded-xl rounded-br-sm text-[11px] leading-snug bg-[var(--color-primary)] text-[var(--color-on-primary)]">
          {user}
        </p>
        <div className="flex items-end gap-1.5 mr-4">
          <span className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'var(--gradient-brand)' }}>
            <Bot className="w-3 h-3 text-[var(--color-on-brand)]" aria-hidden="true" />
          </span>
          <p className="px-2.5 py-1.5 rounded-xl rounded-bl-sm text-[11px] leading-snug bg-[var(--color-bg-raised)] text-[var(--color-text-secondary)]">
            {ai}
          </p>
        </div>
      </div>
      <p className="mt-2.5 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-[var(--color-secondary-muted)] text-[var(--color-secondary)]">
        <ArrowDown className="w-2.5 h-2.5" aria-hidden="true" /> {shared}
      </p>
    </div>
  );
}

function Finding({ who, status, statusIcon: Icon, text }: { who: 'a' | 'b'; status: string; statusIcon: typeof Quote; text: string }) {
  return (
    <div className="p-2.5 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-base)]">
      <div className="flex items-center gap-1.5 mb-1">
        <Avatar who={who} className="w-5 h-5 text-[8px]" />
        <span className="inline-flex items-center gap-1 text-[10px] font-medium text-[var(--color-text-muted)]">
          <Icon className="w-2.5 h-2.5" aria-hidden="true" /> {status}
        </span>
      </div>
      <p className="text-[11px] leading-snug text-[var(--color-text-primary)]">{text}</p>
    </div>
  );
}

/**
 * Illustrative product mockup (static, decorative). Shows two private AI chats
 * feeding a shared knowledge layer, ending in a mutually approved decision.
 */
export function HeroMockup() {
  return (
    <div
      role="img"
      aria-label="Illustration: two founders each chat privately with their own AI. Key findings flow into shared knowledge, where a potential conflict is flagged, and both founders approve one shared direction."
      className="relative rounded-[28px] border border-[var(--color-border)] bg-[var(--color-surface-secondary)] p-3 sm:p-6"
    >
      <div aria-hidden="true" className="grid gap-3 sm:gap-4 lg:grid-cols-[1fr_1.15fr_1fr] items-start">
        <div className="grid min-[480px]:grid-cols-2 lg:contents gap-3">
          <div className="lg:order-1 lg:mt-6 animate-fade-up" style={{ animationDelay: '120ms' }}>
            <PrivateChat
              who="a"
              name="Alex"
              user="I think students are our best first customers."
              ai="What evidence would confirm that?"
              shared="Shared as a hypothesis"
            />
          </div>
          <div className="lg:order-3 lg:mt-6 animate-fade-up" style={{ animationDelay: '220ms' }}>
            <PrivateChat
              who="b"
              name="Jordan"
              user="Professionals I interviewed can't pick a next step."
              ai="That's reported evidence — worth sharing."
              shared="Shared as evidence"
            />
          </div>
        </div>

        <div className={cn(mockCard, 'lg:order-2 p-3.5 animate-fade-up')} style={{ animationDelay: '320ms' }}>
          <div className="flex items-center justify-between mb-2.5">
            <p className="flex items-center gap-1.5 text-xs font-semibold text-[var(--color-text-primary)]">
              <Users className="w-3.5 h-3.5 text-[var(--color-secondary)]" /> Shared knowledge
            </p>
            <span className="text-[10px] text-[var(--color-text-muted)]">No transcripts shared</span>
          </div>
          <div className="space-y-2">
            <Finding who="a" status="Hypothesis" statusIcon={Lightbulb} text="Students are likely our first customers." />
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-[10px] font-medium bg-[var(--color-warning-muted)] text-[var(--color-warning)]">
              <Scale className="w-3 h-3" /> Potential conflict · worth discussing
            </div>
            <Finding who="b" status="Reported evidence" statusIcon={Quote} text="Professionals struggle to choose their next action." />
          </div>

          <div className="mt-3 p-3 rounded-xl border border-[var(--color-secondary)]/40 bg-[var(--color-secondary-muted)]">
            <p className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-secondary)]">
              <ShieldCheck className="w-3 h-3" /> Approved direction
            </p>
            <p className="mt-1 text-[11px] leading-snug text-[var(--color-text-primary)]">
              Interview five students and five professionals before choosing a segment.
            </p>
            <div className="mt-2 flex items-center gap-1.5">
              <Avatar who="a" className="w-5 h-5 text-[8px]" />
              <Avatar who="b" className="w-5 h-5 text-[8px] -ml-2.5" />
              <span className="inline-flex items-center gap-0.5 text-[10px] text-[var(--color-text-secondary)]">
                <Check className="w-3 h-3 text-[var(--color-success)]" /> Approved by both founders
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
