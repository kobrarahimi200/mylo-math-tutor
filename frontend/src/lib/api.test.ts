import { afterEach, expect, it, vi } from "vitest";
import { sendMessage } from "./api";

afterEach(() => vi.unstubAllGlobals());

it("explains backend connection failures", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
  await expect(sendMessage("2+2", "beginner")).rejects.toThrow("backend is running on port 8000");
});

it("explains proxy failures", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ status: 502, ok: false }));
  await expect(sendMessage("2+2", "beginner")).rejects.toThrow("Start the backend on port 8000");
});

it("renders structured validation errors as text", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    status: 422, ok: false, json: async () => ({ detail: [{ msg: "Question too long" }, { msg: "Invalid level" }] }),
  }));
  await expect(sendMessage("2+2", "beginner")).rejects.toThrow("Question too long; Invalid level");
});


it("explains non-JSON backend failures without losing the formula", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    status: 500, ok: false, json: async () => { throw new SyntaxError("Not JSON"); },
  }));
  await expect(sendMessage("$$x^2$$", "beginner")).rejects.toThrow("Your formula is still available");
});
