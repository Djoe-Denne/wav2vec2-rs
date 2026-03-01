export function Glossary() {
  return (
    <details className="bg-white rounded shadow p-4 mb-4 group">
      <summary className="cursor-pointer font-semibold text-gray-700 list-none flex items-center gap-2 [&::-webkit-details-marker]:hidden">
        <span className="transition-transform inline-block group-open:rotate-90" aria-hidden>▸</span>
        <span>Glossary</span>
        <span className="text-sm text-gray-500 font-normal">(acronyms and terms)</span>
      </summary>
      <dl className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-2 text-sm">
        <div><dt className="font-medium text-gray-800">RTF</dt><dd className="text-gray-600">Real-time factor: total_ms ÷ duration_ms. &lt;1 means faster than real time.</dd></div>
        <div><dt className="font-medium text-gray-800">ECDF</dt><dd className="text-gray-600">Empirical cumulative distribution function: proportion of utterances at or below each latency value.</dd></div>
        <div><dt className="font-medium text-gray-800">P90</dt><dd className="text-gray-600">90th percentile: 90% of values are at or below this number (tail latency).</dd></div>
        <div><dt className="font-medium text-gray-800">ms</dt><dd className="text-gray-600">Milliseconds.</dd></div>
        <div><dt className="font-medium text-gray-800">GPU</dt><dd className="text-gray-600">Graphics processing unit. Peak GPU alloc = maximum GPU memory allocated during a run.</dd></div>
        <div><dt className="font-medium text-gray-800">CPU</dt><dd className="text-gray-600">Central processing unit. Peak CPU = maximum process memory (RSS) during a stage.</dd></div>
        <div><dt className="font-medium text-gray-800">MB</dt><dd className="text-gray-600">Megabyte(s); 1 MB = 1,048,576 bytes.</dd></div>
        <div><dt className="font-medium text-gray-800">forward</dt><dd className="text-gray-600">Neural network inference (encoder forward pass).</dd></div>
        <div><dt className="font-medium text-gray-800">post</dt><dd className="text-gray-600">Tensor conversions and overhead after inference.</dd></div>
        <div><dt className="font-medium text-gray-800">dp</dt><dd className="text-gray-600">Dynamic programming: alignment (e.g. CTC or Viterbi).</dd></div>
        <div><dt className="font-medium text-gray-800">group</dt><dd className="text-gray-600">Grouping: merging tokens into words or segments.</dd></div>
        <div><dt className="font-medium text-gray-800">conf</dt><dd className="text-gray-600">Confidence computation per word or segment.</dd></div>
        <div><dt className="font-medium text-gray-800">align</dt><dd className="text-gray-600">Combined alignment time: dp + group + conf.</dd></div>
      </dl>
    </details>
  );
}
