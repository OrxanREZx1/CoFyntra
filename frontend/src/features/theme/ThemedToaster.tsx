'use client';

import { Toaster } from 'sonner';
import { useTheme } from './theme';

/** Sonner toaster that follows the app theme and uses design tokens. */
export function ThemedToaster() {
  const theme = useTheme() ?? 'light';
  return (
    <Toaster
      position="bottom-right"
      theme={theme}
      toastOptions={{
        style: {
          background: 'var(--color-bg-elevated)',
          border: '1px solid var(--color-border)',
          color: 'var(--color-text-primary)',
          boxShadow: 'var(--shadow-card)',
        },
      }}
    />
  );
}
