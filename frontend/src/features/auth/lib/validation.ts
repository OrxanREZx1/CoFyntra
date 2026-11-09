// Pure form validation for the auth screens (no React, no storage).

export const LIMITS = { nameMin: 2, nameMax: 80, emailMax: 254, passwordMin: 8, passwordMax: 128 } as const;

export interface SignupValues {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
}
export interface LoginValues {
  email: string;
  password: string;
}
export type FieldErrors<T> = Partial<Record<keyof T, string>>;

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export function validateName(name: string): string | undefined {
  const v = name.trim();
  if (!v) return 'Enter your name.';
  if (v.length < LIMITS.nameMin) return 'Name must be at least 2 characters.';
  if (v.length > LIMITS.nameMax) return `Name must be ${LIMITS.nameMax} characters or fewer.`;
  return undefined;
}

export function validateEmail(email: string): string | undefined {
  const v = email.trim();
  if (!v) return 'Enter your email address.';
  if (v.length > LIMITS.emailMax || !EMAIL.test(v)) return 'Enter a valid email address, like name@example.com.';
  return undefined;
}

export function validateNewPassword(password: string): string | undefined {
  if (!password) return 'Create a password.';
  if (password.length < LIMITS.passwordMin) return `Use at least ${LIMITS.passwordMin} characters.`;
  if (password.length > LIMITS.passwordMax) return `Use ${LIMITS.passwordMax} characters or fewer.`;
  if (!/[A-Za-z]/.test(password) || !/\d/.test(password)) return 'Include at least one letter and one number.';
  return undefined;
}

export function validateConfirmPassword(password: string, confirm: string): string | undefined {
  if (!confirm) return 'Confirm your password.';
  if (confirm !== password) return "Passwords don't match.";
  return undefined;
}

export function validateSignup(v: SignupValues): FieldErrors<SignupValues> {
  const errors: FieldErrors<SignupValues> = {
    name: validateName(v.name),
    email: validateEmail(v.email),
    password: validateNewPassword(v.password),
    confirmPassword: validateConfirmPassword(v.password, v.confirmPassword),
  };
  return prune(errors);
}

export function validateLogin(v: LoginValues): FieldErrors<LoginValues> {
  return prune({ email: validateEmail(v.email), password: v.password ? undefined : 'Enter your password.' });
}

function prune<T>(errors: FieldErrors<T>): FieldErrors<T> {
  return Object.fromEntries(Object.entries(errors).filter(([, e]) => e)) as FieldErrors<T>;
}
