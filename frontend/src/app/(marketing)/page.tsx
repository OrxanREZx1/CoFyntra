import { Hero } from '@/features/marketing/components/Hero';
import { FeatureHighlights, FinalCta, HowItWorks, ProblemSection, SiteFooter } from '@/features/marketing/components/Sections';
import { SiteHeader } from '@/features/marketing/components/SiteHeader';

/** Public landing page. Server-rendered; only the theme toggle and mobile menu hydrate. */
export default function LandingPage() {
  return (
    <>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:px-3 focus:py-2 focus:rounded-lg focus:bg-[var(--color-bg-elevated)]"
      >
        Skip to content
      </a>
      <SiteHeader />
      <main id="main">
        <Hero />
        <ProblemSection />
        <HowItWorks />
        <FeatureHighlights />
        <FinalCta />
      </main>
      <SiteFooter />
    </>
  );
}
