'use client';

import { DemoRouteGate } from '@/features/workspace/DemoRouteGate';
import { OnboardingWizard } from './OnboardingWizard';

/** UX routing only (see DemoRouteGate): no demo session → signup; already onboarded → workspace. */
export function OnboardingRoute() {
  return (
    <DemoRouteGate redirects={{ 'signed-out': '/signup', ready: '/workspace' }}>
      {(access) => (
        <OnboardingWizard displayName={access.status === 'needs-onboarding' ? access.session.displayName : null} />
      )}
    </DemoRouteGate>
  );
}
