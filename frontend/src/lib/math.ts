export function normalizeMath(content: string): string {
  // Models commonly use TeX delimiters; remark-math expects dollar delimiters.
  // Leave code spans and fenced examples untouched.
  return content.split(/(```[\s\S]*?```|`[^`\n]*`)/g).map((part, index) =>
    index % 2 ? part : part
      .replace(/\\\[([\s\S]*?)\\\]/g, (_, math: string) => `\n$$\n${math.trim()}\n$$\n`)
      .replace(/\\\(([\s\S]*?)\\\)/g, (_, math: string) => `$${math.trim()}$`)
  ).join("");
}

