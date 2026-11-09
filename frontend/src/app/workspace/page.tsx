import type { Metadata } from 'next';
import { WorkspaceRoute } from '@/features/workspace/WorkspaceRoute';

export const metadata: Metadata = { title: 'Workspace' };

export default function WorkspacePage() {
  return (
    <div className="fixed inset-0 flex flex-col overflow-hidden">
      <WorkspaceRoute />
    </div>
  );
}
