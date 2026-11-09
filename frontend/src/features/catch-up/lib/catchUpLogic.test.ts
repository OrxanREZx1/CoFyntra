import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { CatchUpSummary } from '@/types';
import { MOCK_KNOWLEDGE_CONFLICTS, MOCK_KNOWLEDGE_ITEMS, MOCK_PROJECT } from '../../../mocks/index.ts';
import { buildDemoCatchUp, validateCatchUpReferences } from './catchUpLogic.ts';

const P = MOCK_PROJECT.id;
const items = MOCK_KNOWLEDGE_ITEMS;
const snapshot = { projectId: P, items, conflicts: MOCK_KNOWLEDGE_CONFLICTS };
for (const [me, co, coName] of [['founder-a', 'founder-b', 'Jordan Park'], ['founder-b', 'founder-a', 'Alex Chen']] as const) {
  test(`catch-up for ${me} only covers ${co}'s items and references real IDs`, () => {
    const s = buildDemoCatchUp({ snapshot, requestingFounderId: me, coFounderId: co, coFounderName: coName });
    const known = new Set(items.map((i) => i.id));
    assert.equal(s.projectId, P);
    assert.ok(s.findings.length > 0);
    for (const f of s.findings) {
      assert.ok(f.knowledgeItemIds.length > 0, 'every finding has a reference');
      for (const id of f.knowledgeItemIds) assert.ok(known.has(id), `${id} exists`);
      if (f.kind !== 'conflict') assert.equal(items.find((i) => i.id === f.knowledgeItemIds[0])?.contributorId, co);
    }
    const coCount = items.filter((i) => i.contributorId === co).length;
    assert.equal(s.coverage?.itemCount, coCount);
    assert.equal(s.findings.filter((f) => f.kind !== 'conflict').length, coCount);
    const conflict = s.findings.find((f) => f.kind === 'conflict');
    assert.ok(conflict);
    assert.deepEqual([...conflict.knowledgeItemIds].sort(), ['K-101', 'K-106']);
    assert.ok(conflict.text.toLowerCase().includes('your finding'), 'refers to own item as "your finding"');
    assert.equal(validateCatchUpReferences(s, P, known).unresolvedIds.length, 0);
  });
}
test('catch-up is deterministic', () => {
  const args = { snapshot, requestingFounderId: 'founder-a', coFounderId: 'founder-b', coFounderName: 'Jordan Park' } as const;
  assert.deepEqual(buildDemoCatchUp(args), buildDemoCatchUp(args));
});
test('catch-up with no co-founder items is empty', () => {
  const s = buildDemoCatchUp({ snapshot: { projectId: P, items: items.filter((i) => i.contributorId === 'founder-a'), conflicts: [] }, requestingFounderId: 'founder-a', coFounderId: 'founder-b', coFounderName: 'Jordan' });
  assert.equal(s.findings.length, 0);
  assert.equal(s.coverage?.itemCount, 0);
});
test('unknown references are removed; findings left empty are dropped', () => {
  const known = new Set(['K-101', 'K-102']);
  const s: CatchUpSummary = { projectId: P, requestingFounderId: 'founder-a', coFounderId: 'founder-b', coFounderName: 'J', summary: 's',
    findings: [
      { id: 'f1', kind: 'discovery', text: 't', knowledgeItemIds: ['K-102', 'K-999'] },
      { id: 'f2', kind: 'discovery', text: 't', knowledgeItemIds: ['K-404'] },
    ], knowledgeItemIds: ['K-102', 'K-999', 'K-404'] };
  const r = validateCatchUpReferences(s, P, known);
  assert.deepEqual(r.summary.findings.map((f) => [f.id, f.knowledgeItemIds]), [['f1', ['K-102']]]);
  assert.deepEqual(r.unresolvedIds.sort(), ['K-404', 'K-999']);
  assert.deepEqual(r.summary.knowledgeItemIds, ['K-102']);
});
test('catch-up from another project is rejected entirely', () => {
  const s: CatchUpSummary = { projectId: 'other', requestingFounderId: 'founder-a', coFounderId: 'founder-b', coFounderName: 'J', summary: 's',
    findings: [{ id: 'f', kind: 'discovery', text: 't', knowledgeItemIds: ['K-101'] }], knowledgeItemIds: ['K-101'] };
  const r = validateCatchUpReferences(s, P, new Set(['K-101']));
  assert.equal(r.summary.findings.length, 0);
});
test('mock data: every item belongs to the project and nothing is pre-verified', () => {
  for (const i of items) assert.equal(i.projectId, P);
  assert.ok(!items.some((i) => i.epistemicStatus === 'verified-evidence'));
  for (const c of MOCK_KNOWLEDGE_CONFLICTS) for (const id of c.itemIds) assert.ok(items.some((i) => i.id === id));
});

