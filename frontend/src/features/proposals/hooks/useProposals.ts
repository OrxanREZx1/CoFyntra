'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import useSWR, { useSWRConfig } from 'swr';
import { toast } from 'sonner';
import { normalizeError, type ApiError } from '@/lib/api/errors';
import { BACKEND_MODE, LIVE_SYNC_INTERVAL_MS } from '@/lib/config';
import type { DecisionProposal, FounderId, ProposalContent } from '@/types';
import { committedStateKey } from '@/features/committed-state/hooks/useCommittedState';
import { DEMO_DECISIONS_STORAGE_PREFIX } from '../services/demoProposalAdapter';
import { proposalService, type ProposalMutationResult } from '../services/proposalService';

/** Project-scoped L3 proposals cache, separate from the committed state cache. */
export const proposalsKey = (projectId: string) => ['proposals', projectId] as const;

const NO_PROPOSALS: DecisionProposal[] = [];

export function useProposals(projectId: string) {
  const { mutate: globalMutate } = useSWRConfig();
  const { data, error, mutate } = useSWR(proposalsKey(projectId), () => proposalService.listProposals(projectId), {
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
    shouldRetryOnError: false,
    refreshInterval: BACKEND_MODE === 'api' ? LIVE_SYNC_INTERVAL_MS : 0,
    refreshWhenHidden: false,
    refreshWhenOffline: false,
  });

  // Demo only: other tabs of THIS browser write the same localStorage, so
  // refresh when they change it. (Not cross-browser sync.)
  useEffect(() => {
    if (proposalService.mode !== 'demo') return;
    const onStorage = (e: StorageEvent) => {
      if (e.key === `${DEMO_DECISIONS_STORAGE_PREFIX}:${projectId}`) {
        void globalMutate(proposalsKey(projectId));
        void globalMutate(committedStateKey(projectId));
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [projectId, globalMutate]);

  return {
    mode: proposalService.mode,
    proposals: data ?? NO_PROPOSALS,
    isLoading: data === undefined && !error,
    error: error ? normalizeError(error) : undefined,
    reload: () => mutate(),
  };
}

export type ProposalAction = 'create' | 'submit' | 'revise' | 'approve' | 'request-changes';

export type ActionResult = { ok: true; result: ProposalMutationResult } | { ok: false; error: ApiError };

function upsert(list: DecisionProposal[], proposal: DecisionProposal): DecisionProposal[] {
  return list.some((p) => p.id === proposal.id)
    ? list.map((p) => (p.id === proposal.id ? proposal : p))
    : [...list, proposal];
}

function announce({ outcome, proposal }: ProposalMutationResult) {
  const v = `version ${proposal.version}`;
  switch (outcome) {
    case 'CREATED':
      return proposal.status === 'DRAFT'
        ? toast.success('Draft saved', { description: 'Submit it when you’re ready for your co-founder to review.' })
        : toast.success('Proposal shared', { description: 'Both founders must approve this version.' });
    case 'SUBMITTED':
      return toast.success('Proposal submitted', { description: 'Both founders must approve this version.' });
    case 'DRAFT_UPDATED':
      return toast.success('Draft updated');
    case 'REVISED':
      return toast.success(`Version ${proposal.version} published`, {
        description: 'This proposal was updated. Previous approvals no longer apply.',
      });
    case 'APPROVED':
      return toast.success(`You approved ${v}`, { description: 'Waiting for your co-founder to approve.' });
    case 'ALREADY_APPROVED':
      return toast.info(`You already approved ${v}`);
    case 'COMMITTED':
      return toast.success('Decision confirmed by both founders', {
        description: 'The committed direction has been updated.',
      });
    case 'ALREADY_COMMITTED':
      return toast.info('This decision is already committed');
    case 'CHANGES_REQUESTED':
      return toast.success('Changes requested', { description: 'Approval is paused until the proposal is revised.' });
    case 'ALREADY_REQUESTED':
      return toast.info('You already requested changes on this version');
  }
}

/**
 * All L3 mutations. Rules:
 * - one in-flight action per proposal (duplicate clicks are ignored);
 * - caches are updated only from the service's confirmed result — a commit is
 *   never shown before the service confirms it;
 * - every mutation is followed by a revalidation from the source of truth;
 * - a stale-version conflict refreshes the proposal and tells the founder.
 */
export function useProposalActions(projectId: string, founderId: FounderId) {
  const { mutate } = useSWRConfig();
  const inFlight = useRef(new Set<string>());
  const [busy, setBusy] = useState<Record<string, ProposalAction>>({});

  const run = useCallback(
    async (target: string, action: ProposalAction, call: () => Promise<ProposalMutationResult>): Promise<ActionResult | null> => {
      if (inFlight.current.has(target)) return null;
      inFlight.current.add(target);
      setBusy((b) => ({ ...b, [target]: action }));
      try {
        const result = await call();
        await mutate(proposalsKey(projectId), (prev: DecisionProposal[] = []) => upsert(prev, result.proposal), {
          revalidate: false,
        });
        if (result.committedState) {
          await mutate(committedStateKey(projectId), result.committedState, { revalidate: false });
        }
        announce(result);
        return { ok: true, result };
      } catch (e) {
        const error = normalizeError(e);
        if (error.kind === 'conflict') {
          toast.warning('This proposal has changed', { description: error.userMessage });
        } else {
          toast.error('Action not completed', { description: error.userMessage });
        }
        return { ok: false, error };
      } finally {
        inFlight.current.delete(target);
        setBusy((b) => {
          const next = { ...b };
          delete next[target];
          return next;
        });
        // Reconcile with the source of truth, including catch-up summaries
        // that list proposals awaiting each founder.
        void mutate(proposalsKey(projectId));
        void mutate(committedStateKey(projectId));
        void mutate((key) => Array.isArray(key) && key[0] === 'catch-up' && key[1] === projectId);
      }
    },
    [mutate, projectId],
  );

  return {
    busy,
    create: (content: ProposalContent, submit: boolean) =>
      run('new', 'create', () => proposalService.createProposal(projectId, founderId, content, { submit })),
    submit: (p: DecisionProposal) =>
      run(p.id, 'submit', () => proposalService.submitDraft(projectId, founderId, p.id, p.version)),
    revise: (p: DecisionProposal, content: ProposalContent, changeNote: string | null) =>
      run(p.id, 'revise', () =>
        proposalService.reviseProposal(projectId, founderId, p.id, p.version, content, changeNote),
      ),
    // The version the founder is LOOKING AT is sent, so a stale view is rejected.
    approve: (p: DecisionProposal) =>
      run(p.id, 'approve', () => proposalService.approve(projectId, founderId, p.id, p.version)),
    requestChanges: (p: DecisionProposal, reason: string) =>
      run(p.id, 'request-changes', () => proposalService.requestChanges(projectId, founderId, p.id, p.version, reason)),
  };
}
