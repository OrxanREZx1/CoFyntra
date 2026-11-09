import type { Metadata, Viewport } from 'next';
import { Fraunces, Geist } from 'next/font/google';
import { THEME_CRITICAL_CSS, THEME_INIT_SCRIPT } from '@/features/theme/themeInitScript';
import { ThemedToaster } from '@/features/theme/ThemedToaster';
import './globals.css';

const body = Geist({ subsets: ['latin'], variable: '--font-body', display: 'swap' });
const heading = Fraunces({ subsets: ['latin'], variable: '--font-heading', display: 'swap', axes: ['opsz'] });

export const metadata: Metadata = {
  title: {
    default: 'Confyntra — Think independently. Build in sync.',
    template: '%s · Confyntra',
  },
  description:
    'Confyntra helps startup co-founders connect their AI-powered discoveries, stay aligned and decide together — without sharing private conversations.',
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#FFFFFF' },
    { media: '(prefers-color-scheme: dark)', color: '#0B132B' },
  ],
};

export default function RootLayout({ children }: LayoutProps<'/'>) {
  return (
    // data-theme is set before paint by THEME_INIT_SCRIPT; suppress the
    // expected attribute difference between server HTML and the client.
    <html lang="en" data-theme="light" className={`${body.variable} ${heading.variable}`} suppressHydrationWarning>
      <head>
        <style dangerouslySetInnerHTML={{ __html: THEME_CRITICAL_CSS }} />
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="min-h-full antialiased bg-[var(--color-bg-base)] text-[var(--color-text-primary)]">
        {children}
        <ThemedToaster />
      </body>
    </html>
  );
}
