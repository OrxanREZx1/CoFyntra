import { test } from 'node:test';
import assert from 'node:assert/strict';
import { validateLogin, validateSignup } from './validation.ts';

const ok = { name: 'Alex Chen', email: 'alex@example.com', password: 'secret123', confirmPassword: 'secret123' };

test('valid signup has no errors', () => {
  assert.deepEqual(validateSignup(ok), {});
});

test('signup requires every field', () => {
  const e = validateSignup({ name: ' ', email: '', password: '', confirmPassword: '' });
  assert.deepEqual(Object.keys(e).sort(), ['confirmPassword', 'email', 'name', 'password']);
});

test('signup rejects bad email, weak password and mismatch', () => {
  assert.ok(validateSignup({ ...ok, email: 'alex@' }).email);
  assert.ok(validateSignup({ ...ok, email: 'alex example.com' }).email);
  assert.match(validateSignup({ ...ok, password: 'short1', confirmPassword: 'short1' }).password ?? '', /at least 8/);
  assert.match(validateSignup({ ...ok, password: 'onlyletters', confirmPassword: 'onlyletters' }).password ?? '', /letter and one number/);
  assert.match(validateSignup({ ...ok, confirmPassword: 'secret124' }).confirmPassword ?? '', /match/);
  assert.ok(validateSignup({ ...ok, name: 'A' }).name);
});

test('login validates email and requires a password', () => {
  assert.deepEqual(validateLogin({ email: 'alex@example.com', password: 'x' }), {});
  const e = validateLogin({ email: 'nope', password: '' });
  assert.ok(e.email && e.password);
});
