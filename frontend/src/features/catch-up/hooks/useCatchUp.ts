'use client';

import { useMemo } from 'react';
import useSWR from 'swr';
import { normalizeError } from '@/lib/api/errors';
import { BACKEND_MODE, LIVE_SYNC_INTERVAL_MS } from '@/lib/config';
import type { FounderId } from '@/types';
import { useSharedKnowledge } from '@/features/knowledge/hooks/useSharedKnowledge';
import { validateCatchUpReferences } from '../lib/catchUpLogic';
import { catchUpService } from '../services/catchUpService';

/** Founder-scoped: each founder gets their own catch-up about the other. */
export const catchUpKey = (projectId: string, founderId: FounderId) => ['catch-up', projectId, founderId] as const;

export function useCatchUp(projectId: string, founderId: FounderId) {
  const knowledge = useSharedKnowledge(projectId);
  const { data, error, isValidating, mutate } = useSWR(
    catchUpKey(projectId, founderId),
    () => catchUpService.getCatchUp(projectId, founderId),
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
      shouldRetryOnError: false,
      refreshInterval: BACKEND_MODE === 'api' ? LIVE_SYNC_INTERVAL_MS : 0,
      refreshWhenHidden: false,
      refreshWhenOffline: false,
    },
  );

  // Only show references that resolve to real items in this project.
  const knowledgeReady = !knowledge.isLoading && !knowledge.error;
  const validated = useMemo(() => {
    if (!data || !knowledgeReady) return undefined;
    return validateCatchUpReferences(data, projectId, new Set(knowledge.itemsById.keys()));
  }, [data, knowledgeReady, projectId, knowledge.itemsById]);

  const combinedError = error ?? knowledge.error;

  return {
    mode: catchUpService.mode,
    summary: validated?.summary,
    unresolvedIds: validated?.unresolvedIds ?? [],
    itemsById: knowledge.itemsById,
    conflicts: knowledge.conflicts,
    isLoading: !combinedError && validated === undefined,
    isRefreshing: isValidating && data !== undefined,
    error: combinedError ? normalizeError(combinedError) : undefined,
    refresh: () => {
      if (knowledge.error) void knowledge.reload();
      return mutate();
    },
  };
}
