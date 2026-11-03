import { useCallback, useEffect, useState } from "react";
import { getProviders, listProviderModels, removeProviderKey, saveProviderKey, saveProviderSettings, testProvider } from "../lib/api";
import type { ProviderOverview } from "../types";

export function ProviderSettings() {
  const [overview, setOverview] = useState<ProviderOverview | null>(null);
  const [selected, setSelected] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [apiKey, setApiKey] = useState("");
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const apply = useCallback((next: ProviderOverview) => { setOverview(next); setSelected((current) => current || next.provider); }, []);
  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    try { await action(); } catch (error) { setMessage({ ok: false, text: error instanceof Error ? error.message : "Request failed." }); } finally { setBusy(false); }
  };

  useEffect(() => { getProviders().then(apply).catch((error: Error) => setMessage({ ok: false, text: error.message })); }, [apply]);

  const info = overview?.providers.find((provider) => provider.id === selected);
  useEffect(() => {
    setModels([]);
    if (info?.configured) listProviderModels(info.id).then((result) => setModels(result.models)).catch(() => setModels([]));
  }, [info?.id, info?.configured]);

  if (!overview) return <div className="setting-row"><span><b>AI provider</b><small>{message?.text ?? "Loading…"}</small></span></div>;
  const activeLabel = overview.active.available ? `${overview.providers.find((p) => p.id === overview.active.provider)?.label ?? overview.active.provider} · ${overview.active.model}` : "Offline math only";
  const modelOptions = info ? Array.from(new Set([info.model, ...models])) : [];

  return (
    <div className="setting-row provider-settings" style={{ display: "block" }}>
      <span><b>AI provider</b><small>Active: <strong data-testid="active-provider">{activeLabel}</strong></small></span>
      {overview.active.notice && <p role="status">{overview.active.notice}</p>}
      <label>Provider{" "}
        <select value={selected} onChange={(event) => { setSelected(event.target.value); setMessage(null); }}>
          {overview.providers.map((provider) => <option key={provider.id} value={provider.id}>{provider.label}{provider.configured ? "" : " (not connected)"}</option>)}
        </select>
      </label>
      {info && <>
        <label>Model{" "}
          <select value={info.model} onChange={(event) => run(async () => apply(await saveProviderSettings({ models: { [info.id]: event.target.value } })))}>
            {modelOptions.map((model) => <option key={model} value={model}>{model}</option>)}
          </select>
        </label>
        {info.needsKey && <div>
          <input type="password" autoComplete="off" aria-label={`${info.label} API key`} placeholder={info.configured ? `Key saved (${info.credentialSource})` : "Paste API key"} value={apiKey} onChange={(event) => setApiKey(event.target.value)} />
          <button type="button" disabled={busy || apiKey.trim().length < 8} onClick={() => run(async () => { apply(await saveProviderKey(info.id, apiKey.trim())); setApiKey(""); setMessage({ ok: true, text: "Key saved in the system credential store." }); })}>Save key</button>
          {info.credentialSource === "keychain" && <button type="button" disabled={busy} onClick={() => run(async () => { apply(await removeProviderKey(info.id)); setMessage(null); })}>Remove key</button>}
        </div>}
        {info.paused && <p role="alert">{info.paused}</p>}
        <button type="button" disabled={busy} onClick={() => run(async () => { const result = await testProvider(info.id); setMessage({ ok: result.ok, text: result.message }); apply(await getProviders()); })}>{info.configured || !info.needsKey ? "Test connection" : "Connect"}</button>
        <button type="button" disabled={busy || overview.provider === info.id} onClick={() => run(async () => apply(await saveProviderSettings({ provider: info.id })))}>{overview.provider === info.id ? "Active provider" : "Use this provider"}</button>
      </>}
      <label>Fallback provider{" "}
        <select value={overview.fallback} onChange={(event) => run(async () => apply(await saveProviderSettings({ fallback: event.target.value })))}>
          <option value="">None (offline math only)</option>
          {overview.providers.filter((provider) => provider.id !== overview.provider).map((provider) => <option key={provider.id} value={provider.id}>{provider.label}</option>)}
        </select>
      </label>
      <small>Used only when the active provider fails with a billing, rate-limit, credential, model, or network error. You will be told when it happens.</small>
      {message && <p role={message.ok ? "status" : "alert"}>{message.text}</p>}
    </div>
  );
}
