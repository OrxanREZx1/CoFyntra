import type { Metadata } from 'next';
import { Logo } from '@/components/ui/Logo';
import { ThemeToggle } from '@/features/theme/ThemeToggle';
import { OnboardingRoute } from '@/features/onboarding/components/OnboardingRoute';

export const metadata: Metadata = { title: 'Welcome' };

export default function OnboardingPage() {
  return (
    <div className="min-h-dvh flex flex-col">
      <header className="flex items-center justify-between px-5 sm:px-8 h-16">
        <Logo />
        <ThemeToggle />
      </header>
      <main className="flex-1 flex justify-center px-5 sm:px-8 pt-6 sm:pt-14 pb-12">
        <OnboardingRoute />
      </main>
    </div>
  );
}
