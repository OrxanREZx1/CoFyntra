'use client';

import useSWR from 'swr';
import { normalizeError } from '@/lib/api/errors';
import { BACKEND_MODE, LIVE_SYNC_INTERVAL_MS } from '@/lib/config';
import { proposalService } from '@/features/proposals/services/proposalService';

/**
 * The project's OFFICIAL committed state. Cached separately from proposals:
 * it is only ever written from a service-confirmed commit, never from a
 * proposal or from UI state.
 */
export const committedStateKey = (projectId: string) => ['committed-state', projectId] as const;

export function useCommittedState(projectId: string) {
  const { data, error, mutate } = useSWR(committedStateKey(projectId), () => proposalService.getCommittedState(projectId), {
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
    shouldRetryOnError: false,
    refreshInterval: BACKEND_MODE === 'api' ? LIVE_SYNC_INTERVAL_MS : 0,
    refreshWhenHidden: false,
    refreshWhenOffline: false,
  });
  return {
    state: data,
    isLoading: data === undefined && !error,
    error: error ? normalizeError(error) : undefined,
    reload: () => mutate(),
  };
}
