'use client';

import { useCallback, useRef, useState } from 'react';
import { DEFAULT_KNOWLEDGE_FILTERS, matchesFilters, type KnowledgeFilters } from '../lib/filterKnowledge';
import { useSharedKnowledge } from './useSharedKnowledge';

export interface KnowledgeFocusRequest {
  id: string;
  /** Changes on every request so re-opening the same item re-triggers focus. */
  nonce: number;
}

/**
 * Shared state for a view that shows the knowledge list and catch-up:
 * filters, plus "open reference" navigation from anywhere (catch-up,
 * conflict cards) to a shared knowledge card.
 */
export function useKnowledgeNavigation(projectId: string, showKnowledgeTab: () => void) {
  const { itemsById } = useSharedKnowledge(projectId);
  const [filters, setFilters] = useState<KnowledgeFilters>(DEFAULT_KNOWLEDGE_FILTERS);
  const [focusRequest, setFocusRequest] = useState<KnowledgeFocusRequest | null>(null);
  const nonceRef = useRef(0);

  const openReference = useCallback(
    (id: string) => {
      const item = itemsById.get(id);
      if (!item) return;
      // Only reset filters if they would hide the target card.
      setFilters((current) => (matchesFilters(item, current) ? current : DEFAULT_KNOWLEDGE_FILTERS));
      nonceRef.current += 1;
      setFocusRequest({ id, nonce: nonceRef.current });
      showKnowledgeTab();
    },
    [itemsById, showKnowledgeTab],
  );

  const clearFocusRequest = useCallback(() => setFocusRequest(null), []);

  return { filters, setFilters, focusRequest, clearFocusRequest, openReference };
}
