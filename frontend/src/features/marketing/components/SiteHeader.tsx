import Link from 'next/link';
import { cn } from '@/lib/utils';
import { Logo } from '@/components/ui/Logo';
import { ThemeToggle } from '@/features/theme/ThemeToggle';
import { buttonPrimary, container, navLink } from '../styles';
import { MobileNav } from './MobileNav';

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-base)]/85 backdrop-blur-md">
      <div className={`${container} relative flex items-center justify-between h-16`}>
        <Logo />
        <nav aria-label="Main" className="hidden md:flex items-center gap-7">
          <a href="#how-it-works" className={navLink}>
            How It Works
          </a>
          <a href="#features" className={navLink}>
            Features
          </a>
        </nav>
        <div className="flex items-center gap-1.5">
          <ThemeToggle />
          <Link href="/login" className={cn(navLink, 'hidden md:inline-flex px-3 py-2')}>
            Log In
          </Link>
          <Link href="/signup" className={cn(buttonPrimary, 'hidden md:inline-flex h-9 px-4')}>
            Get Started
          </Link>
          <MobileNav />
        </div>
      </div>
    </header>
  );
}
