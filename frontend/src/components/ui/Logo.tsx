import Link from 'next/link';
import { Layers } from 'lucide-react';
import { cn } from '@/lib/utils';

/** Confyntra mark + wordmark. Server-safe. */
export function Logo({ href = '/', className, size = 'md' }: { href?: string | null; className?: string; size?: 'sm' | 'md' }) {
  const content = (
    <>
      <span
        className={cn('rounded-lg flex items-center justify-center flex-shrink-0', size === 'sm' ? 'w-7 h-7' : 'w-8 h-8')}
        style={{ background: 'var(--gradient-brand)' }}
        aria-hidden="true"
      >
        <Layers className={cn('text-[var(--color-on-brand)]', size === 'sm' ? 'w-3.5 h-3.5' : 'w-4 h-4')} />
      </span>
      <span className={cn('font-display font-semibold tracking-tight text-[var(--color-text-primary)]', size === 'sm' ? 'text-base' : 'text-lg')}>
        Confyntra
      </span>
    </>
  );
  return href ? (
    <Link href={href} className={cn('inline-flex items-center gap-2 rounded-lg', className)} aria-label="Confyntra home">
      {content}
    </Link>
  ) : (
    <span className={cn('inline-flex items-center gap-2', className)}>{content}</span>
  );
}
