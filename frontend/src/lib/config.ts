/**
 * Which data source the app uses.
 *
 * 'demo' (default): typed in-browser mock adapters — no backend involved.
 * 'api': the real backend via the shared HTTP client. Only enable this once
 *        the backend contract (docs/sync_contract.md) exists and the
 *        placeholder endpoints in each feature's service have been updated.
 *
 * NEXT_PUBLIC_BACKEND_MODE is public configuration, never a secret.
 */
export const BACKEND_MODE: 'demo' | 'api' =
  process.env.NEXT_PUBLIC_BACKEND_MODE === 'api' ? 'api' : 'demo';

/** Revalidate shared live state every 4s while the tab is visible and online. */
const configuredSyncInterval = process.env.NEXT_PUBLIC_SYNC_POLL_INTERVAL_MS;
const parsedSyncInterval = configuredSyncInterval === undefined ? 4_000 : Number(configuredSyncInterval);

if (!Number.isInteger(parsedSyncInterval) || parsedSyncInterval < 3_000 || parsedSyncInterval > 5_000) {
  throw new Error('NEXT_PUBLIC_SYNC_POLL_INTERVAL_MS must be an integer between 3000 and 5000.');
}

export const LIVE_SYNC_INTERVAL_MS = parsedSyncInterval;
