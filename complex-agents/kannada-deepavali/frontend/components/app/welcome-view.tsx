'use client';

import { Button } from '@/components/ui/button';

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div
      ref={ref}
      className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-4 py-8 pt-20 md:pt-24 lg:pt-20"
    >
      <div className="pointer-events-none absolute inset-0" aria-hidden>
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: "url('/opengraph-image-bg.png')" }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-indigo-950/75 via-amber-950/50 to-orange-900/80" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_20%,rgba(251,191,36,0.35),transparent_55%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_80%_80%,rgba(234,88,12,0.25),transparent_45%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_15%_70%,rgba(168,85,247,0.2),transparent_40%)]" />
      </div>

      <div className="relative z-10 flex w-full items-center justify-center">
        <Button
          size="lg"
          onClick={onStartCall}
          className="w-full cursor-pointer rounded-full border border-amber-200/30 bg-amber-500 px-8 text-sm font-semibold tracking-wide text-amber-950 shadow-[0_0_40px_rgba(251,191,36,0.45)] transition-all hover:bg-amber-400 hover:shadow-[0_0_56px_rgba(251,191,36,0.55)] sm:w-auto sm:min-w-80"
        >
          {startButtonText}
        </Button>
      </div>
    </div>
  );
};