import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import type { ProposalContent } from '@/types';
import {
  approveProposal,
  createProposal,
  decisionFor,
  emptyStore,
  getProposalConflicts,
  ProposalRuleError,
  requestChanges,
  reviseProposal,
  submitDraft,
  type DecisionStore,
} from './proposalMachine.ts';

const P = 'project-1';
const T = (n: number) => `2026-11-09T10:0${n}:00.000Z`;
const content: ProposalContent = {
  title: 'Initial Customer Segment',
  direction: 'Focus on university student founders as our first customer segment.',
  nextAction: 'Interview five university student founders.',
  supportingKnowledgeIds: ['K-101'],
  conflictingKnowledgeIds: ['K-102'],
};

function rejects(fn: () => unknown, code: ProposalRuleError['code']) {
  assert.throws(fn, (e: unknown) => e instanceof ProposalRuleError && e.code === code);
}

/** Fresh store with one PENDING_APPROVAL proposal by founder A. */
function pending(): { store: DecisionStore; id: string } {
  const { store, proposal } = createProposal(emptyStore(P), { actor: 'founder-a', content, submit: true, now: T(0) });
  return { store, id: proposal.id };
}

describe('creating proposals', () => {
  test('creates v1 PENDING_APPROVAL with no approvals (Case A)', () => {
    const { store, id } = pending();
    const p = store.proposals[0];
    assert.equal(p.id, id);
    assert.equal(p.id, 'P-001');
    assert.equal(p.version, 1);
    assert.equal(p.status, 'PENDING_APPROVAL');
    assert.deepEqual(p.approvals, [], 'the proposer is not auto-approved');
    assert.equal(store.committed.revision, 0, 'creating never touches committed state');
    assert.equal(store.committed.decision, null);
  });

  test('draft → submit; only the proposer can submit', () => {
    const { store, proposal } = createProposal(emptyStore(P), { actor: 'founder-a', content, submit: false, now: T(0) });
    assert.equal(proposal.status, 'DRAFT');
    rejects(() => approveProposal(store, { actor: 'founder-b', proposalId: proposal.id, version: 1, now: T(1) }), 'NOT_ALLOWED');
    rejects(() => submitDraft(store, { actor: 'founder-b', proposalId: proposal.id, version: 1, now: T(1) }), 'NOT_ALLOWED');
    const r = submitDraft(store, { actor: 'founder-a', proposalId: proposal.id, version: 1, now: T(1) });
    assert.equal(r.proposal.status, 'PENDING_APPROVAL');
    assert.equal(r.proposal.version, 1);
  });

  test('editing a draft does not bump the version', () => {
    const { store, proposal } = createProposal(emptyStore(P), { actor: 'founder-a', content, submit: false, now: T(0) });
    const r = reviseProposal(store, {
      actor: 'founder-a', proposalId: proposal.id, version: 1, content: { ...content, title: 'Edited' }, now: T(1),
    });
    assert.equal(r.outcome, 'DRAFT_UPDATED');
    assert.equal(r.proposal.version, 1);
    assert.equal(r.proposal.title, 'Edited');
  });

  test('validates input', () => {
    const s = emptyStore(P);
    rejects(() => createProposal(s, { actor: 'founder-a', content: { ...content, direction: '   ' }, submit: true, now: T(0) }), 'INVALID_INPUT');
    rejects(
      () => createProposal(s, { actor: 'founder-a', content: { ...content, conflictingKnowledgeIds: ['K-101'] }, submit: true, now: T(0) }),
      'INVALID_INPUT',
    );
    rejects(() => createProposal(s, { actor: 'founder-c' as never, content, submit: true, now: T(0) }), 'NOT_ALLOWED');
  });

  test('IDs are sequential', () => {
    let s = pending().store;
    s = createProposal(s, { actor: 'founder-b', content, submit: true, now: T(1) }).store;
    assert.deepEqual(s.proposals.map((p) => p.id), ['P-001', 'P-002']);
  });
});

describe('two-founder approval', () => {
  test('one approval does not commit (Case B)', () => {
    const { store, id } = pending();
    const r = approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    assert.equal(r.outcome, 'APPROVED');
    assert.equal(r.proposal.status, 'PENDING_APPROVAL');
    assert.equal(decisionFor(r.proposal, 'founder-a')?.decision, 'APPROVED');
    assert.equal(decisionFor(r.proposal, 'founder-a')?.version, 1);
    assert.equal(decisionFor(r.proposal, 'founder-b'), undefined);
    assert.equal(r.store.committed.revision, 0);
    assert.equal(r.store.committed.decision, null);
  });

  test('both approving the same version commits exactly once (Case C)', () => {
    const { store, id } = pending();
    const a = approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    const b = approveProposal(a.store, { actor: 'founder-b', proposalId: id, version: 1, now: T(2) });
    assert.equal(b.outcome, 'COMMITTED');
    assert.equal(b.proposal.status, 'COMMITTED');
    assert.equal(b.proposal.committedRevision, 1);
    assert.equal(b.store.committed.revision, 1);
    const d = b.store.committed.decision!;
    assert.equal(d.direction, content.direction);
    assert.equal(d.nextAction, content.nextAction);
    assert.equal(d.proposalId, id);
    assert.equal(d.proposalVersion, 1);
    assert.deepEqual(d.approvedBy.map((x) => x.founderId).sort(), ['founder-a', 'founder-b']);
    assert.equal(d.committedAt, T(2));
  });

  test('order does not matter (B then A)', () => {
    const { store, id } = pending();
    const b = approveProposal(store, { actor: 'founder-b', proposalId: id, version: 1, now: T(1) });
    assert.equal(b.outcome, 'APPROVED');
    const a = approveProposal(b.store, { actor: 'founder-a', proposalId: id, version: 1, now: T(2) });
    assert.equal(a.outcome, 'COMMITTED');
  });

  test('duplicate approval creates no duplicate entry (Case F)', () => {
    const { store, id } = pending();
    const a1 = approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    const a2 = approveProposal(a1.store, { actor: 'founder-a', proposalId: id, version: 1, now: T(2) });
    assert.equal(a2.outcome, 'ALREADY_APPROVED');
    assert.equal(a2.proposal.approvals.length, 1);
    assert.equal(a2.proposal.approvals[0].decidedAt, T(1), 'original approval kept');
    assert.equal(a2.store, a1.store, 'store unchanged');
  });

  test('approving after commit never commits twice', () => {
    const { store, id } = pending();
    const a = approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    const b = approveProposal(a.store, { actor: 'founder-b', proposalId: id, version: 1, now: T(2) });
    for (const actor of ['founder-a', 'founder-b'] as const) {
      const again = approveProposal(b.store, { actor, proposalId: id, version: 1, now: T(3) });
      assert.equal(again.outcome, 'ALREADY_COMMITTED');
      assert.equal(again.store.committed.revision, 1);
      assert.equal(again.store, b.store);
    }
  });

  test('approving an old version is rejected and records nothing (Case G)', () => {
    const { store, id } = pending();
    const v2 = reviseProposal(store, {
      actor: 'founder-a', proposalId: id, version: 1, content: { ...content, nextAction: 'Interview ten.' }, now: T(1),
    });
    rejects(() => approveProposal(v2.store, { actor: 'founder-b', proposalId: id, version: 1, now: T(2) }), 'STALE_VERSION');
    rejects(
      () => requestChanges(v2.store, { actor: 'founder-b', proposalId: id, version: 1, reason: 'x', now: T(2) }),
      'STALE_VERSION',
    );
    assert.deepEqual(v2.store.proposals[0].approvals, []);
  });

  test('unknown proposal is rejected', () => {
    rejects(() => approveProposal(emptyStore(P), { actor: 'founder-a', proposalId: 'P-404', version: 1, now: T(0) }), 'NOT_FOUND');
  });
});

describe('request changes', () => {
  test('blocks commitment (Case D)', () => {
    const { store, id } = pending();
    const a = approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    const rc = requestChanges(a.store, { actor: 'founder-b', proposalId: id, version: 1, reason: 'Compare segments first.', now: T(2) });
    assert.equal(rc.proposal.status, 'CHANGES_REQUESTED');
    assert.equal(decisionFor(rc.proposal, 'founder-b')?.reason, 'Compare segments first.');
    rejects(() => approveProposal(rc.store, { actor: 'founder-b', proposalId: id, version: 1, now: T(3) }), 'NOT_ALLOWED');
    assert.equal(rc.store.committed.revision, 0);
  });

  test('requires a reason', () => {
    const { store, id } = pending();
    rejects(() => requestChanges(store, { actor: 'founder-b', proposalId: id, version: 1, reason: '  ', now: T(1) }), 'INVALID_INPUT');
  });

  test('replaces the same founder’s earlier approval; duplicate request is a no-op', () => {
    const { store, id } = pending();
    const a = approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    const rc = requestChanges(a.store, { actor: 'founder-a', proposalId: id, version: 1, reason: 'Second thoughts', now: T(2) });
    assert.equal(rc.proposal.approvals.length, 1);
    assert.equal(decisionFor(rc.proposal, 'founder-a')?.decision, 'CHANGES_REQUESTED');
    const again = requestChanges(rc.store, { actor: 'founder-a', proposalId: id, version: 1, reason: 'again', now: T(3) });
    assert.equal(again.outcome, 'ALREADY_REQUESTED');
    assert.equal(again.proposal.approvals.length, 1);
  });
});

describe('revisions', () => {
  test('v1 → v2 clears all approvals; both must approve again (Case E)', () => {
    const { store, id } = pending();
    const a = approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    const rc = requestChanges(a.store, { actor: 'founder-b', proposalId: id, version: 1, reason: 'Add professionals', now: T(2) });
    const v2 = reviseProposal(rc.store, {
      actor: 'founder-a', proposalId: id, version: 1,
      content: { ...content, nextAction: 'Interview five students and five professionals.' },
      changeNote: 'Added professionals', now: T(3),
    });
    assert.equal(v2.outcome, 'REVISED');
    assert.equal(v2.proposal.version, 2);
    assert.equal(v2.proposal.status, 'PENDING_APPROVAL');
    assert.deepEqual(v2.proposal.approvals, []);
    assert.equal(v2.proposal.changeNote, 'Added professionals');
    assert.equal(v2.proposal.history.length, 1);
    assert.equal(v2.proposal.history[0].version, 1);
    assert.equal(v2.proposal.history[0].approvals.length, 2, 'v1 decisions kept as history only');

    // A's v1 approval does not count toward v2.
    const b2 = approveProposal(v2.store, { actor: 'founder-b', proposalId: id, version: 2, now: T(4) });
    assert.equal(b2.outcome, 'APPROVED');
    assert.equal(b2.store.committed.revision, 0);
    const a2 = approveProposal(b2.store, { actor: 'founder-a', proposalId: id, version: 2, now: T(5) });
    assert.equal(a2.outcome, 'COMMITTED');
    assert.equal(a2.store.committed.decision?.proposalVersion, 2);
    assert.equal(a2.store.committed.decision?.nextAction, 'Interview five students and five professionals.');
  });

  test('only the proposer can revise; stale revise rejected; no-op revise rejected', () => {
    const { store, id } = pending();
    rejects(() => reviseProposal(store, { actor: 'founder-b', proposalId: id, version: 1, content: { ...content, title: 'x' }, now: T(1) }), 'NOT_ALLOWED');
    rejects(() => reviseProposal(store, { actor: 'founder-a', proposalId: id, version: 1, content, now: T(1) }), 'INVALID_INPUT');
    const v2 = reviseProposal(store, { actor: 'founder-a', proposalId: id, version: 1, content: { ...content, title: 'x' }, now: T(1) });
    rejects(() => reviseProposal(v2.store, { actor: 'founder-a', proposalId: id, version: 1, content: { ...content, title: 'y' }, now: T(2) }), 'STALE_VERSION');
  });

  test('committed proposals are immutable', () => {
    const { store, id } = pending();
    const a = approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    const b = approveProposal(a.store, { actor: 'founder-b', proposalId: id, version: 1, now: T(2) });
    rejects(() => reviseProposal(b.store, { actor: 'founder-a', proposalId: id, version: 1, content: { ...content, title: 'x' }, now: T(3) }), 'NOT_ALLOWED');
    rejects(() => requestChanges(b.store, { actor: 'founder-b', proposalId: id, version: 1, reason: 'x', now: T(3) }), 'NOT_ALLOWED');
  });
});

describe('committed state', () => {
  test('a pending proposal never overwrites an older committed direction', () => {
    const { store, id } = pending();
    const a = approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    const committed = approveProposal(a.store, { actor: 'founder-b', proposalId: id, version: 1, now: T(2) }).store;

    const newer = { ...content, title: 'Pivot', direction: 'Target working professionals instead.' };
    const c = createProposal(committed, { actor: 'founder-b', content: newer, submit: true, now: T(3) });
    const half = approveProposal(c.store, { actor: 'founder-b', proposalId: c.proposal.id, version: 1, now: T(4) });
    assert.equal(half.store.committed.revision, 1);
    assert.equal(half.store.committed.decision?.direction, content.direction, 'old direction still official');

    const done = approveProposal(half.store, { actor: 'founder-a', proposalId: c.proposal.id, version: 1, now: T(5) });
    assert.equal(done.store.committed.revision, 2);
    assert.equal(done.store.committed.decision?.direction, newer.direction);
  });

  test('transitions do not mutate the input store', () => {
    const { store, id } = pending();
    const snapshot = JSON.stringify(store);
    approveProposal(store, { actor: 'founder-a', proposalId: id, version: 1, now: T(1) });
    requestChanges(store, { actor: 'founder-b', proposalId: id, version: 1, reason: 'r', now: T(1) });
    assert.equal(JSON.stringify(store), snapshot);
  });
});

describe('proposal conflicts', () => {
  test('pairs conflicting findings only via supplied relationships', () => {
    const conflicts = [{ id: 'CF-1', projectId: P, itemIds: ['K-101', 'K-106'] as [string, string], explanation: null }];
    const r = getProposalConflicts({ supportingKnowledgeIds: ['K-101'], conflictingKnowledgeIds: ['K-106', 'K-102'] }, conflicts);
    assert.deepEqual(r, [
      { conflictingItemId: 'K-106', supportingItemId: 'K-101', knowledgeConflictId: 'CF-1' },
      { conflictingItemId: 'K-102', supportingItemId: null, knowledgeConflictId: null },
    ]);
  });
});
