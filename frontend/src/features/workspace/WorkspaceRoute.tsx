'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { WorkspaceShell } from '@/components/layout/WorkspaceShell';
import { endDemoSession } from '@/features/auth/demoSession';
import { DemoRouteGate } from './DemoRouteGate';

/**
 * UX routing only (see DemoRouteGate): no demo session → login; onboarding not
 * finished → onboarding. Real access control belongs to the backend.
 */
export function WorkspaceRoute() {
  const router = useRouter();
  const [leaving, setLeaving] = useState(false);

  const exitDemo = () => {
    setLeaving(true); // suppress the signed-out redirect to /login
    router.replace('/');
    endDemoSession();
  };

  return (
    <DemoRouteGate redirects={leaving ? {} : { 'signed-out': '/login', 'needs-onboarding': '/onboarding' }}>
      {(access) =>
        access.status === 'ready' ? (
          <WorkspaceShell session={access.session} profile={access.profile} onExitDemo={exitDemo} />
        ) : null
      }
    </DemoRouteGate>
  );
}
