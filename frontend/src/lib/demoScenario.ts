/**
 * DEMO/QA ONLY: lets testers force edge-case states in demo mode via the URL,
 * e.g. `/?demo=knowledge-error` or `/?demo=knowledge-empty&demo=catchup-error`.
 * Has no effect on the real backend adapters.
 */
export type DemoScenario = 'knowledge-error' | 'knowledge-empty' | 'catchup-error' | 'proposals-error';

export function isDemoScenario(name: DemoScenario): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).getAll('demo').includes(name);
}

/** Simulated round-trip so loading states are visible in demo mode. */
export function demoLatency(ms = 300): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
