'use client';

import { useDemoProfile } from '@/features/onboarding/profileStore';
import type { DemoProfile } from '@/features/onboarding/lib/onboarding';
import { useDemoSession, type DemoSession } from './demoSession';

export type DemoAccess =
  | { status: 'loading' }
  | { status: 'signed-out' }
  | { status: 'needs-onboarding'; session: DemoSession }
  | { status: 'ready'; session: DemoSession; profile: DemoProfile };

/**
 * Where the visitor is in the demo journey. UX routing only — the real
 * backend/auth provider must enforce access and onboarding status.
 */
export function useDemoAccess(): DemoAccess {
  const session = useDemoSession();
  const profile = useDemoProfile();
  if (session === undefined || profile === undefined) return { status: 'loading' };
  if (!session) return { status: 'signed-out' };
  if (!profile) return { status: 'needs-onboarding', session };
  return { status: 'ready', session, profile };
}

/** Where to send someone right after entering the demo. */
export function nextDemoRoute(hasProfile: boolean): '/workspace' | '/onboarding' {
  return hasProfile ? '/workspace' : '/onboarding';
}
