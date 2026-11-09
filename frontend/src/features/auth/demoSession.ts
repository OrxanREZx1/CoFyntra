'use client';

/**
 * ⚠️ DEMO-ONLY session. Not authentication.
 *
 * Confyntra has no auth backend yet. This records only that someone entered
 * the demo in THIS browser, plus an optional display name for greetings.
 * It never stores or transmits an email address or password, it proves no
 * identity, and any route gating built on it is a UX convenience — not a
 * security boundary. Real access control and onboarding status must come from
 * the backend / auth provider.
 */
import { useSyncExternalStore } from 'react';
import { createLocalStore } from '@/lib/localStore';

export interface DemoSession {
  kind: 'demo';
  displayName: string | null;
  startedAt: string;
}

export const DEMO_SESSION_KEY = 'confyntra.demo.session.v1';

const store = createLocalStore<DemoSession>(DEMO_SESSION_KEY, (raw) => {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  if (r.kind !== 'demo' || typeof r.startedAt !== 'string') return null;
  return {
    kind: 'demo',
    displayName: typeof r.displayName === 'string' && r.displayName.trim() ? r.displayName.slice(0, 80) : null,
    startedAt: r.startedAt,
  };
});

/** Starts a demo session. Pass only a display name — never credentials. */
export function startDemoSession(displayName: string | null) {
  store.set({ kind: 'demo', displayName: displayName?.trim().slice(0, 80) || null, startedAt: new Date().toISOString() });
}

export function endDemoSession() {
  store.set(null);
}

/** `undefined` until known on the client; `null` when there is no demo session. */
export function useDemoSession(): DemoSession | null | undefined {
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getServerSnapshot);
}

export function readDemoSession(): DemoSession | null {
  return store.get();
}
