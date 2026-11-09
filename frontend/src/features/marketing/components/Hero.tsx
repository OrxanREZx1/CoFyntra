import Link from 'next/link';
import { ArrowRight, Lock } from 'lucide-react';
import { cn } from '@/lib/utils';
import { buttonPrimary, buttonSecondary, container } from '../styles';
import { HeroMockup } from './HeroMockup';

export function Hero() {
  return (
    <section aria-labelledby="hero-title" className="pt-14 sm:pt-20 pb-16 sm:pb-24">
      <div className={container}>
        <div className="max-w-3xl mx-auto text-center">
          <p className="animate-fade-up inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border border-[var(--color-border)] bg-[var(--color-bg-elevated)] text-[var(--color-text-secondary)]">
            <Lock className="w-3 h-3 text-[var(--color-secondary)]" aria-hidden="true" />
            Your conversations stay private. Your startup knowledge stays connected.
          </p>
          <h1
            id="hero-title"
            className="animate-fade-up mt-6 font-display text-[2.6rem] leading-[1.05] sm:text-6xl lg:text-7xl tracking-tight text-[var(--color-text-primary)]"
            style={{ animationDelay: '60ms' }}
          >
            Think independently.
            <br />
            <span className="text-[var(--color-secondary)]">Build in sync.</span>
          </h1>
          <p
            className="animate-fade-up mt-6 text-base sm:text-lg leading-relaxed text-[var(--color-text-secondary)] max-w-2xl mx-auto"
            style={{ animationDelay: '120ms' }}
          >
            Confyntra connects your team&apos;s AI-powered discoveries, keeps startup knowledge organized, and helps
            co-founders make decisions together — without sharing private conversations.
          </p>
          <div className="animate-fade-up mt-9 flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3" style={{ animationDelay: '180ms' }}>
            <Link href="/signup" className={cn(buttonPrimary, 'h-12 px-6 text-[15px]')}>
              Start Collaborating <ArrowRight className="w-4 h-4" aria-hidden="true" />
            </Link>
            <a href="#how-it-works" className={cn(buttonSecondary, 'h-12 px-6 text-[15px]')}>
              See How It Works
            </a>
          </div>
        </div>

        <div className="mt-14 sm:mt-20">
          <HeroMockup />
        </div>
      </div>
    </section>
  );
}
