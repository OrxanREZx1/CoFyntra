import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MOCK_KNOWLEDGE_ITEMS } from '../../../mocks/index.ts';
import { DEFAULT_KNOWLEDGE_FILTERS as D, filterKnowledge, isFiltering } from './filterKnowledge.ts';

const items = MOCK_KNOWLEDGE_ITEMS;
const ids = (xs: { id: string }[]) => xs.map((x) => x.id).sort();

test('no filters returns all items, newest first', () => {
  const r = filterKnowledge(items, D);
  assert.equal(r.length, items.length);
  for (let i = 1; i < r.length; i++) assert.ok(Date.parse(r[i - 1].createdAt) >= Date.parse(r[i].createdAt));
  assert.equal(isFiltering(D), false);
});
test('category filter', () => {
  assert.deepEqual(ids(filterKnowledge(items, { ...D, category: 'Target Customer' })), ['K-101', 'K-106']);
});
test('contributor filter', () => {
  const r = filterKnowledge(items, { ...D, contributorId: 'founder-b' });
  assert.ok(r.length > 0 && r.every((i) => i.contributorId === 'founder-b'));
});
test('status filter', () => {
  assert.deepEqual(ids(filterKnowledge(items, { ...D, status: 'unverified-ai-claim' })), ['K-105']);
});
test('search is case-insensitive on claim text only', () => {
  assert.deepEqual(ids(filterKnowledge(items, { ...D, query: '  WORKING professionals ' })), ['K-102', 'K-106', 'K-107']);
  // "budgets" appears only in an excerpt, not a claim
  assert.equal(filterKnowledge(items, { ...D, query: 'budgets' }).length, 0);
});
test('combined filters', () => {
  assert.deepEqual(ids(filterKnowledge(items, { ...D, contributorId: 'founder-a', status: 'hypothesis' })), ['K-101']);
});
test('filtering does not mutate input', () => {
  const copy = [...items];
  filterKnowledge(items, { ...D, query: 'x' });
  assert.deepEqual(items, copy);
});

