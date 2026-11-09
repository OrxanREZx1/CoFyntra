import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isAdminDemoLogin } from './demoCredentials.ts';

test('accepts the admin demo credentials', () => {
  assert.equal(isAdminDemoLogin('admin@gmail.com', 'admin'), true);
});

test('normalizes email but requires the exact demo password', () => {
  assert.equal(isAdminDemoLogin(' Admin@Gmail.com ', 'admin'), true);
  assert.equal(isAdminDemoLogin('admin@gmail.com', 'Admin'), false);
  assert.equal(isAdminDemoLogin('other@gmail.com', 'admin'), false);
});
