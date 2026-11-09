'use client';

import { useSyncExternalStore } from 'react';
import { createLocalStore } from '@/lib/localStore';

export type Theme = 'light' | 'dark';

/** Must match the key read by THEME_INIT_SCRIPT. */
export const THEME_STORAGE_KEY = 'confyntra.theme';

const themeStore = createLocalStore<Theme>(THEME_STORAGE_KEY, (raw) =>
  raw === 'light' || raw === 'dark' ? raw : null,
);

/** Light is the default for first-time visitors (no system-preference sniffing). */
export const DEFAULT_THEME: Theme = 'light';

export function setTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  themeStore.set(theme);
}

/** Current theme; `undefined` during SSR/hydration (render a neutral state). */
export function useTheme(): Theme | undefined {
  const stored = useSyncExternalStore(themeStore.subscribe, themeStore.getSnapshot, themeStore.getServerSnapshot);
  return stored === undefined ? undefined : (stored ?? DEFAULT_THEME);
}
