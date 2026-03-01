import { HelpCircle } from 'lucide-react';

/** Renders a small info icon that shows the given tip on hover. Use next to metric labels and chart titles. */
export function MetricTooltip({ tip }: { tip: string }) {
  return (
    <span
      className="inline-flex align-middle text-gray-400 hover:text-gray-600 cursor-help ml-0.5"
      title={tip}
      aria-label={tip}
    >
      <HelpCircle className="w-3.5 h-3.5 shrink-0" />
    </span>
  );
}
