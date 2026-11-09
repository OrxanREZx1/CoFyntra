'use client';

import { useMemo } from 'react';
import useSWR from 'swr';
import { normalizeError } from '@/lib/api/errors';
import { BACKEND_MODE, LIVE_SYNC_INTERVAL_MS } from '@/lib/config';
import type { KnowledgeConflict, KnowledgeItem } from '@/types';
import { knowledgeService } from '../services/knowledgeService';

/**
 * Shared knowledge is project-scoped (both founders see the same L2 items)
 * and cached under its own key — completely separate from the
 * founder-scoped private chat cache ('chat', ...).
 */
export const knowledgeKey = (projectId: string) => ['shared-knowledge', projectId] as const;

const NO_ITEMS: KnowledgeItem[] = [];
const NO_CONFLICTS: KnowledgeConflict[] = [];

export function useSharedKnowledge(projectId: string) {
  const { data, error, mutate } = useSWR(knowledgeKey(projectId), () => knowledgeService.getSharedKnowledge(projectId), {
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
    shouldRetryOnError: false,
    refreshInterval: BACKEND_MODE === 'api' ? LIVE_SYNC_INTERVAL_MS : 0,
    refreshWhenHidden: false,
    refreshWhenOffline: false,
  });

  const items = data?.items ?? NO_ITEMS;
  const conflicts = data?.conflicts ?? NO_CONFLICTS;

  const itemsById = useMemo(() => new Map(items.map((i) => [i.id, i])), [items]);

  /** itemId -> IDs of the items it potentially conflicts with. */
  const conflictPartners = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const { itemIds: [a, b] } of conflicts) {
      map.set(a, [...(map.get(a) ?? []), b]);
      map.set(b, [...(map.get(b) ?? []), a]);
    }
    return map;
  }, [conflicts]);

  return {
    mode: knowledgeService.mode,
    items,
    conflicts,
    itemsById,
    conflictPartners,
    isLoading: data === undefined && !error,
    error: error ? normalizeError(error) : undefined,
    reload: () => mutate(),
  };
}
