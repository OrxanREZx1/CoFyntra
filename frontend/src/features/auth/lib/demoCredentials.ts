const ADMIN_DEMO_EMAIL = 'admin@gmail.com';
const ADMIN_DEMO_PASSWORD = 'admin';

export function isAdminDemoLogin(email: string, password: string): boolean {
  return email.trim().toLowerCase() === ADMIN_DEMO_EMAIL && password === ADMIN_DEMO_PASSWORD;
}
