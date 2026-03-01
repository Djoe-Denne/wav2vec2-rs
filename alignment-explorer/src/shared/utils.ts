/**
 * Normalize word text for comparison: trim, lowercase, collapse whitespace.
 */
export function normalizeWordText(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ');
}

/**
 * Generate a stable id for an audio entry from its relative path (e.g. for routing).
 */
export function entryIdFromPath(relativePath: string): string {
  return relativePath
    .replace(/\\/g, '/')
    .replace(/\.flac$/i, '')
    .replace(/\s+/g, '_');
}
