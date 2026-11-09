'use client';

/**
 * ⚠️ DEMO-ONLY onboarding profile, kept in this browser's localStorage,
 * separate from the demo session and from all workspace data. Contains only
 * the three onboarding answers — no contact details or credentials. It is not
 * sent to any server and does not drive any AI personalization.
 */
import { useSyncExternalStore } from 'react';
import { createLocalStore } from '@/lib/localStore';
import { isDemoProfile, toProfile, type DemoProfile, type OnboardingDraft } from './lib/onboarding';

export const DEMO_PROFILE_KEY = 'confyntra.demo.profile.v1';

const store = createLocalStore<DemoProfile>(DEMO_PROFILE_KEY, (raw) => (isDemoProfile(raw) ? raw : null));

export function saveDemoProfile(draft: OnboardingDraft) {
  store.set(toProfile(draft, new Date().toISOString()));
}

export function clearDemoProfile() {
  store.set(null);
}

export function readDemoProfile(): DemoProfile | null {
  return store.get();
}

/** `undefined` until known on the client; `null` when onboarding isn't complete. */
export function useDemoProfile(): DemoProfile | null | undefined {
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getServerSnapshot);
}
