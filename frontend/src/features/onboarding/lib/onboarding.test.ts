import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EMPTY_DRAFT, isDemoProfile, toProfile, validateStep, type OnboardingDraft } from './onboarding.ts';

const full: OnboardingDraft = { reason: 'have-cofounder', role: 'student', projectName: '  EcoBox ', idea: '  Reusable packaging for campus cafés. ' };

test('each step requires its answer', () => {
  assert.ok(validateStep(0, EMPTY_DRAFT));
  assert.equal(validateStep(0, { ...EMPTY_DRAFT, reason: 'new-startup' }), null);
  assert.ok(validateStep(1, EMPTY_DRAFT));
  assert.equal(validateStep(1, { ...EMPTY_DRAFT, role: 'other' }), null);
  assert.ok(validateStep(2, EMPTY_DRAFT), 'idea required');
  assert.equal(validateStep(2, { ...EMPTY_DRAFT, idea: 'Tutoring app' }), null, 'short answer is fine; name optional');
  assert.ok(validateStep(2, { ...EMPTY_DRAFT, idea: 'ok idea', projectName: 'x'.repeat(61) }));
  assert.ok(validateStep(2, { ...EMPTY_DRAFT, idea: 'x'.repeat(601) }));
});

test('toProfile trims, nulls empty project name, and validates', () => {
  const p = toProfile(full, '2026-11-09T00:00:00Z');
  assert.deepEqual(p, { reason: 'have-cofounder', role: 'student', projectName: 'EcoBox', idea: 'Reusable packaging for campus cafés.', completedAt: '2026-11-09T00:00:00Z' });
  assert.equal(toProfile({ ...full, projectName: '  ' }, 'now').projectName, null);
  assert.throws(() => toProfile({ ...full, role: null }, 'now'));
});

test('stored profile shape is validated', () => {
  assert.ok(isDemoProfile(toProfile(full, 'now')));
  assert.ok(!isDemoProfile({ ...toProfile(full, 'now'), reason: 'hack' }));
  assert.ok(!isDemoProfile(null));
});
