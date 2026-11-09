'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useRef } from 'react';
import { useDemoAccess, type DemoAccess } from '@/features/auth/useDemoAccess';
import { BACKEND_MODE } from '@/lib/config';

type Status = DemoAccess['status'];

interface DemoRouteGateProps {
  /** Status → route to send the visitor to. Statuses not listed render children. */
  redirects: Partial<Record<Exclude<Status, 'loading'>, string>>;
  children: (access: DemoAccess) => React.ReactNode;
}

/**
 * ⚠️ UX ROUTING ONLY — NOT A SECURITY BOUNDARY.
 * Sends visitors to the right step of the demo journey based on browser-local
 * demo state. Anyone can bypass this; real access control and onboarding
 * status must be enforced by the backend / auth provider.
 */
export function DemoRouteGate({ redirects, children }: DemoRouteGateProps) {
  const access = useDemoAccess();
  const router = useRouter();
  const target = BACKEND_MODE === 'demo' && access.status !== 'loading' ? redirects[access.status] : undefined;
  const redirected = useRef<string | null>(null);

  useEffect(() => {
    if (target && redirected.current !== target) {
      redirected.current = target;
      router.replace(target);
    }
  }, [target, router]);

  if (BACKEND_MODE === 'api') {
    return (
      <div className="flex-1 flex items-center justify-center min-h-dvh px-5" role="alert">
        <p className="max-w-md text-center text-sm text-[var(--color-text-secondary)]">
          Live sessions are not integrated because the backend authentication contract is missing. Demo browser state
          cannot be used as live authentication.
        </p>
      </div>
    );
  }

  if (access.status === 'loading' || target) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-dvh" aria-busy="true" aria-label="Loading">
        <span className="w-5 h-5 border-2 border-[var(--color-text-muted)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }
  return <>{children(access)}</>;
}
