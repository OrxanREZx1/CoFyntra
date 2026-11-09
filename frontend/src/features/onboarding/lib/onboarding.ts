// Onboarding questions, answer model and validation (pure — no React/storage).

export const ONBOARDING_REASONS = [
  { id: 'new-startup', label: "I'm starting a new startup." },
  { id: 'have-cofounder', label: "I'm already building with a co-founder." },
  { id: 'organize-research', label: 'I want to organize our AI-powered research.' },
  { id: 'align-decisions', label: 'I want to align our startup decisions.' },
] as const;

export const FOUNDER_ROLES = [
  { id: 'student', label: 'University student' },
  { id: 'professional', label: 'Working professional' },
  { id: 'researcher', label: 'Researcher' },
  { id: 'founder', label: 'Entrepreneur / founder' },
  { id: 'other', label: 'Other' },
] as const;

export type OnboardingReason = (typeof ONBOARDING_REASONS)[number]['id'];
export type FounderRole = (typeof FOUNDER_ROLES)[number]['id'];

export const ONBOARDING_LIMITS = { projectName: 60, ideaMin: 3, ideaMax: 600 } as const;

/** In-progress answers (the wizard's draft). */
export interface OnboardingDraft {
  reason: OnboardingReason | null;
  role: FounderRole | null;
  projectName: string;
  idea: string;
}

/** Completed, non-sensitive demo profile. */
export interface DemoProfile {
  reason: OnboardingReason;
  role: FounderRole;
  projectName: string | null;
  idea: string;
  completedAt: string;
}

export const EMPTY_DRAFT: OnboardingDraft = { reason: null, role: null, projectName: '', idea: '' };

export const ONBOARDING_STEP_COUNT = 3;

/** Returns an error message for the given step (0-based), or null if valid. */
export function validateStep(step: number, d: OnboardingDraft): string | null {
  switch (step) {
    case 0:
      return d.reason ? null : 'Choose the option that fits best.';
    case 1:
      return d.role ? null : 'Choose the option that describes you best.';
    case 2: {
      if (d.projectName.trim().length > ONBOARDING_LIMITS.projectName) {
        return `Keep the project name under ${ONBOARDING_LIMITS.projectName} characters.`;
      }
      const idea = d.idea.trim();
      if (idea.length < ONBOARDING_LIMITS.ideaMin) return 'Add a sentence about your idea — a short answer is fine.';
      if (idea.length > ONBOARDING_LIMITS.ideaMax) return `Please keep it under ${ONBOARDING_LIMITS.ideaMax} characters.`;
      return null;
    }
    default:
      return null;
  }
}

/** Builds the stored profile; throws if any step is invalid. */
export function toProfile(d: OnboardingDraft, now: string): DemoProfile {
  for (let i = 0; i < ONBOARDING_STEP_COUNT; i++) {
    const error = validateStep(i, d);
    if (error) throw new Error(error);
  }
  return {
    reason: d.reason!,
    role: d.role!,
    projectName: d.projectName.trim() || null,
    idea: d.idea.trim(),
    completedAt: now,
  };
}

export function isDemoProfile(raw: unknown): raw is DemoProfile {
  if (!raw || typeof raw !== 'object') return false;
  const r = raw as Record<string, unknown>;
  return (
    ONBOARDING_REASONS.some((o) => o.id === r.reason) &&
    FOUNDER_ROLES.some((o) => o.id === r.role) &&
    (r.projectName === null || typeof r.projectName === 'string') &&
    typeof r.idea === 'string' &&
    typeof r.completedAt === 'string'
  );
}
