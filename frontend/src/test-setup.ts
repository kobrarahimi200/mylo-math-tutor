import "@testing-library/jest-dom/vitest";

class TestResizeObserver implements ResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}

globalThis.ResizeObserver = TestResizeObserver;
Element.prototype.scrollIntoView = () => undefined;

const storage = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
	configurable: true,
	value: {
		get length() { return storage.size; },
		key: (index: number) => Array.from(storage.keys())[index] ?? null,
		getItem: (key: string) => storage.get(key) ?? null,
		setItem: (key: string, value: string) => { storage.set(key, value); },
		removeItem: (key: string) => { storage.delete(key); },
		clear: () => storage.clear(),
	} satisfies Storage,
});