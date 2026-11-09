import Link from 'next/link';
import {
  Archive,
  ArrowRight,
  Bot,
  Check,
  CheckSquare,
  GitFork,
  Hash,
  Lightbulb,
  Lock,
  MessageSquare,
  Quote,
  Scale,
  Sparkles,
  Split,
  Users,
  type LucideIcon,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Logo } from '@/components/ui/Logo';
import { buttonPrimary, container, mockCard, navLink } from '../styles';

function SectionHeader({ id, eyebrow, title, intro }: { id: string; eyebrow: string; title: string; intro?: string }) {
  return (
    <div className="max-w-2xl">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--color-accent)]">{eyebrow}</p>
      <h2 id={id} className="mt-3 font-display text-3xl sm:text-4xl leading-tight tracking-tight text-[var(--color-text-primary)]">
        {title}
      </h2>
      {intro && <p className="mt-4 text-base leading-relaxed text-[var(--color-text-secondary)]">{intro}</p>}
    </div>
  );
}

// ─── Problem ────────────────────────────────────────────────

const PROBLEMS: { icon: LucideIcon; title: string; body: string }[] = [
  { icon: Archive, title: 'Discoveries get buried', body: 'Important findings stay locked inside private AI chats.' },
  { icon: GitFork, title: 'Assumptions drift apart', body: 'Co-founders unknowingly build on conflicting beliefs.' },
  { icon: Split, title: 'Decisions fragment', body: 'What you agreed on becomes unclear or misunderstood.' },
];

export function ProblemSection() {
  return (
    <section aria-labelledby="problem-title" className="py-20 sm:py-28 bg-[var(--color-surface-secondary)]">
      <div className={container}>
        <SectionHeader id="problem-title" eyebrow="The problem" title="Two founders. Two AI conversations. One disconnected strategy." />
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {PROBLEMS.map(({ icon: Icon, title, body }, i) => (
            <div key={title} className="p-6 rounded-2xl bg-[var(--color-bg-base)] border border-[var(--color-border)]">
              <div className="flex items-center justify-between">
                <span className="w-10 h-10 rounded-xl flex items-center justify-center bg-[var(--color-accent-muted)]">
                  <Icon className="w-5 h-5 text-[var(--color-accent)]" aria-hidden="true" />
                </span>
                <span className="font-display text-2xl text-[var(--color-text-disabled)]" aria-hidden="true">
                  0{i + 1}
                </span>
              </div>
              <h3 className="mt-5 text-base font-semibold text-[var(--color-text-primary)]">{title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-[var(--color-text-secondary)]">{body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── How it works ───────────────────────────────────────────

const STEPS: { icon: LucideIcon; title: string; body: string }[] = [
  { icon: Lock, title: 'Think independently', body: 'Each founder works with their own private AI.' },
  { icon: Sparkles, title: 'Connect discoveries', body: 'Important startup findings enter shared knowledge.' },
  { icon: Scale, title: 'Stay aligned', body: 'Catch up and review conflicting assumptions.' },
  { icon: CheckSquare, title: 'Decide together', body: 'Approve one shared direction and next action.' },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" aria-labelledby="how-title" className="py-20 sm:py-28 scroll-mt-16">
      <div className={container}>
        <SectionHeader
          id="how-title"
          eyebrow="How it works"
          title="From private thinking to a shared decision"
          intro="Confyntra connects what you learn, not what you say."
        />
        <ol className="mt-14 grid gap-8 sm:grid-cols-2 lg:grid-cols-4 lg:gap-6 relative">
          {/* connecting line (desktop) */}
          <span aria-hidden="true" className="hidden lg:block absolute top-6 left-6 right-6 h-px bg-[var(--color-border)]" />
          {STEPS.map(({ icon: Icon, title, body }, i) => (
            <li key={title} className="relative">
              <span className="relative z-10 w-12 h-12 rounded-2xl flex items-center justify-center bg-[var(--color-bg-elevated)] border border-[var(--color-border)] shadow-[var(--shadow-card)]">
                <Icon className="w-5 h-5 text-[var(--color-secondary)]" aria-hidden="true" />
              </span>
              <p className="mt-5 text-xs font-semibold text-[var(--color-text-muted)]">Step {i + 1}</p>
              <h3 className="mt-1 text-lg font-semibold text-[var(--color-text-primary)]">{title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-[var(--color-text-secondary)]">{body}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

// ─── Features ───────────────────────────────────────────────

function Pill({ children, tone = 'neutral' }: { children: React.ReactNode; tone?: 'neutral' | 'secondary' | 'warning' | 'success' }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium',
        tone === 'neutral' && 'bg-[var(--color-bg-raised)] text-[var(--color-text-secondary)]',
        tone === 'secondary' && 'bg-[var(--color-secondary-muted)] text-[var(--color-secondary)]',
        tone === 'warning' && 'bg-[var(--color-warning-muted)] text-[var(--color-warning)]',
        tone === 'success' && 'bg-[var(--color-success-muted)] text-[var(--color-success)]',
      )}
    >
      {children}
    </span>
  );
}

const ILLUSTRATIONS: Record<string, React.ReactNode> = {
  private: (
    <div className="space-y-2">
      <Pill tone="secondary">
        <Lock className="w-2.5 h-2.5" /> Only you see this chat
      </Pill>
      <p className="ml-10 px-3 py-2 rounded-xl rounded-br-sm text-xs bg-[var(--color-primary)] text-[var(--color-on-primary)]">
        Who should our first customers be?
      </p>
      <div className="flex items-end gap-1.5 mr-8">
        <span className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'var(--gradient-brand)' }}>
          <Bot className="w-3 h-3 text-[var(--color-on-brand)]" />
        </span>
        <p className="px-3 py-2 rounded-xl rounded-bl-sm text-xs bg-[var(--color-bg-raised)] text-[var(--color-text-secondary)]">
          Let&apos;s compare candidates on pain, reach and ability to pay.
        </p>
      </div>
    </div>
  ),
  knowledge: (
    <div className="space-y-2">
      {[
        ['Target Customer', 'Hypothesis', Lightbulb, 'Students are likely our first customers.'],
        ['Evidence', 'Reported', Quote, 'Professionals struggle to choose a next step.'],
      ].map(([cat, status, Icon, text]) => {
        const I = Icon as LucideIcon;
        return (
          <div key={cat as string} className="p-2.5 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-base)]">
            <div className="flex items-center gap-1.5">
              <Pill>{cat as string}</Pill>
              <Pill tone="secondary">
                <I className="w-2.5 h-2.5" /> {status as string}
              </Pill>
            </div>
            <p className="mt-1.5 text-xs text-[var(--color-text-primary)]">{text as string}</p>
          </div>
        );
      })}
    </div>
  ),
  catchup: (
    <div className="space-y-2">
      <p className="text-xs font-semibold text-[var(--color-text-primary)]">What Jordan shared</p>
      {['New reported evidence on customers', 'An assumption worth examining'].map((t, i) => (
        <div key={t} className="flex items-center justify-between gap-2 p-2 rounded-lg bg-[var(--color-bg-base)] border border-[var(--color-border)]">
          <span className="text-[11px] text-[var(--color-text-secondary)]">{t}</span>
          <Pill>
            <Hash className="w-2.5 h-2.5" />K-10{i + 2}
          </Pill>
        </div>
      ))}
    </div>
  ),
  conflict: (
    <div className="space-y-2">
      <Pill tone="warning">
        <Scale className="w-2.5 h-2.5" /> Potential conflict
      </Pill>
      <div className="grid grid-cols-2 gap-2">
        {['Students first', 'Professionals first'].map((t) => (
          <p key={t} className="p-2 rounded-lg text-[11px] text-center bg-[var(--color-bg-base)] border border-[var(--color-border)] text-[var(--color-text-primary)]">
            {t}
          </p>
        ))}
      </div>
      <p className="text-[10px] text-[var(--color-text-muted)]">Neither is marked as correct.</p>
    </div>
  ),
  approval: (
    <div className="space-y-2">
      <div className="flex gap-1">
        <span className="h-1.5 flex-1 rounded-full bg-[var(--color-success)]" />
        <span className="h-1.5 flex-1 rounded-full bg-[var(--color-success)]" />
      </div>
      {['Alex · approved v2', 'Jordan · approved v2'].map((t) => (
        <p key={t} className="flex items-center gap-1.5 text-[11px] text-[var(--color-text-secondary)]">
          <Check className="w-3 h-3 text-[var(--color-success)]" /> {t}
        </p>
      ))}
      <Pill tone="success">Decision confirmed by both founders</Pill>
    </div>
  ),
};

const FEATURES: { key: keyof typeof ILLUSTRATIONS; icon: LucideIcon; title: string; body: string; span: string }[] = [
  { key: 'private', icon: MessageSquare, title: 'Private AI Workspace', body: 'Think out loud with your own assistant. Your raw conversation is never shown to your co-founder.', span: 'lg:col-span-3' },
  { key: 'knowledge', icon: Users, title: 'Shared Startup Knowledge', body: 'Key findings are shared with attribution and an honest evidence label — hypothesis, reported or verified.', span: 'lg:col-span-3' },
  { key: 'catchup', icon: Sparkles, title: 'Co-Founder Catch-Up', body: 'See what your co-founder learned while you were heads-down, with links to every finding.', span: 'lg:col-span-2' },
  { key: 'conflict', icon: Scale, title: 'Conflict-Aware Decisions', body: 'Disagreements are surfaced side by side before you commit — not buried.', span: 'lg:col-span-2' },
  { key: 'approval', icon: CheckSquare, title: 'Mutual Approval', body: 'A direction becomes official only when both founders approve the same version.', span: 'lg:col-span-2' },
];

export function FeatureHighlights() {
  return (
    <section id="features" aria-labelledby="features-title" className="py-20 sm:py-28 bg-[var(--color-surface-secondary)] scroll-mt-16">
      <div className={container}>
        <SectionHeader id="features-title" eyebrow="Features" title="Everything two founders need to stay on the same page" />
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
          {FEATURES.map(({ key, icon: Icon, title, body, span }) => (
            <article key={key} className={cn('flex flex-col p-5 sm:p-6 rounded-2xl bg-[var(--color-bg-base)] border border-[var(--color-border)]', span)}>
              <div aria-hidden="true" className={cn(mockCard, 'p-4 mb-6 shadow-none bg-[var(--color-bg-elevated)]')}>
                {ILLUSTRATIONS[key]}
              </div>
              <h3 className="flex items-center gap-2 text-base font-semibold text-[var(--color-text-primary)]">
                <Icon className="w-4 h-4 text-[var(--color-secondary)]" aria-hidden="true" />
                {title}
              </h3>
              <p className="mt-1.5 text-sm leading-relaxed text-[var(--color-text-secondary)]">{body}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── Final CTA ──────────────────────────────────────────────

export function FinalCta() {
  return (
    <section aria-labelledby="cta-title" className="py-20 sm:py-28">
      <div className={container}>
        <div className="relative overflow-hidden rounded-[28px] px-6 py-14 sm:px-14 sm:py-20 text-center bg-[var(--color-surface-accent)] border border-[var(--color-border)]">
          <h2 id="cta-title" className="font-display text-3xl sm:text-5xl leading-tight tracking-tight text-[var(--color-text-primary)] max-w-2xl mx-auto">
            Build your startup with a shared understanding.
          </h2>
          <p className="mt-4 text-base text-[var(--color-text-secondary)]">Your conversations stay private. Your startup knowledge stays connected.</p>
          <Link href="/signup" className={cn(buttonPrimary, 'mt-8 h-12 px-7 text-[15px]')}>
            Get Started <ArrowRight className="w-4 h-4" aria-hidden="true" />
          </Link>
        </div>
      </div>
    </section>
  );
}

// ─── Footer ─────────────────────────────────────────────────

export function SiteFooter() {
  return (
    <footer className="border-t border-[var(--color-border)]">
      <div className={`${container} py-12 grid gap-10 md:grid-cols-[2fr_1fr_1fr]`}>
        <div className="max-w-sm">
          <Logo />
          <p className="mt-3 text-sm leading-relaxed text-[var(--color-text-secondary)]">
            An AI collaboration workspace for startup co-founders. Think independently. Build in sync.
          </p>
        </div>
        <nav aria-label="Product">
          <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Product</p>
          <ul className="mt-3 space-y-2">
            <li><a href="#how-it-works" className={navLink}>How It Works</a></li>
            <li><a href="#features" className={navLink}>Features</a></li>
          </ul>
        </nav>
        <nav aria-label="Account">
          <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Account</p>
          <ul className="mt-3 space-y-2">
            <li><Link href="/login" className={navLink}>Log In</Link></li>
            <li><Link href="/signup" className={navLink}>Get Started</Link></li>
          </ul>
        </nav>
      </div>
      <div className={`${container} pb-10`}>
        <p className="text-xs text-[var(--color-text-muted)] border-t border-[var(--color-border-subtle)] pt-6">
          Demo build. Privacy policy and terms of service are not yet available.
        </p>
      </div>
    </footer>
  );
}
