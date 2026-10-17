export function base64ToImageSrc(base64: string): string {
  if (!base64) {
    return "";
  }

  if (base64.startsWith("data:image")) {
    return base64;
  }

  return `data:image/jpeg;base64,${base64}`;
}
