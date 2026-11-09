import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MOCK_KNOWLEDGE_ITEMS, MOCK_COMMITTED_STATE } from '../../mocks/index.ts';
import { deriveStartupMapData } from './startupMapData.ts';

test('derives startup map from shared findings and leaves unapproved direction undecided', () => {
  const map = deriveStartupMapData(MOCK_KNOWLEDGE_ITEMS, MOCK_COMMITTED_STATE);

  assert.match(map.problem ?? '', /customer assumptions/);
  assert.equal(map.approvedDirection, null);
  assert.equal(map.candidateSegments.length, 2);
  assert.equal(map.candidateSegments[0].status, 'Hypothesis');
  assert.equal(map.marketFindings.length, 2);
  assert.equal(map.assumptions.length, 2);
  assert.equal(map.evidence.length, 1);
  assert.equal(map.nextAction, null);
});

test('uses only committed direction and action as approved startup state', () => {
  const committedState = {
    ...MOCK_COMMITTED_STATE,
    revision: 1,
    decision: {
      proposalId: 'P-001',
      proposalVersion: 1,
      title: 'Initial customer segment',
      direction: 'Focus on university student founders.',
      nextAction: 'Interview five university student founders.',
      supportingKnowledgeIds: ['K-101'],
      conflictingKnowledgeIds: ['K-106'],
      approvedBy: [
        { founderId: 'founder-a' as const, approvedAt: '2026-11-08T15:30:00Z' },
        { founderId: 'founder-b' as const, approvedAt: '2026-11-08T15:30:00Z' },
      ],
      committedAt: '2026-11-08T15:30:00Z',
    },
  };

  const map = deriveStartupMapData(MOCK_KNOWLEDGE_ITEMS, committedState);

  assert.equal(map.approvedDirection, 'Focus on university student founders.');
  assert.deepEqual(map.nextAction, {
    text: 'Interview five university student founders.',
    status: 'Approved',
  });
});
