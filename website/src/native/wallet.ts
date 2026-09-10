import { secp256k1 } from '@noble/curves/secp256k1.js';

export type Wallet = { format: 'neuroshard-key-v1'; seed: string; public_key: string };
const utf8 = new TextEncoder();
export const hex = (value: Uint8Array) => Array.from(value, b => b.toString(16).padStart(2, '0')).join('');
const bytes = (value: string) => Uint8Array.from(value.match(/../g) || [], c => parseInt(c, 16));
async function sha(value: string) { return new Uint8Array(await crypto.subtle.digest('SHA-256', utf8.encode(value))); }

// Match Python's canonical, ASCII-only JSON exactly, including surrogate pairs.
export function canonical(value: unknown): string {
  if (value === null || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value)) throw new Error('Transaction numbers must be safe integers');
    return String(value);
  }
  if (typeof value === 'string') return JSON.stringify(value).replace(/[\u007f-\uffff]/g, c => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0'));
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  if (typeof value === 'object') return '{' + Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => canonical(k) + ':' + canonical(v)).join(',') + '}';
  throw new Error('Unsupported transaction value');
}
export async function importWallet(raw: string): Promise<Wallet> {
  if (raw.length > 2048) throw new Error('Key file is too large');
  const value = JSON.parse(raw) as Wallet;
  if (value.format !== 'neuroshard-key-v1' || !/^[0-9a-f]{64}$/.test(value.seed || '')) throw new Error('Use a NeuroShard native key backup');
  const publicKey = hex(secp256k1.getPublicKey(await sha(value.seed), true));
  if (publicKey !== value.public_key) throw new Error('Key file public identity mismatch');
  return {format: 'neuroshard-key-v1', seed: value.seed, public_key: publicKey};
}
export async function createWallet(): Promise<Wallet> {
  const seed = hex(crypto.getRandomValues(new Uint8Array(32)));
  return {format: 'neuroshard-key-v1', seed, public_key: hex(secp256k1.getPublicKey(await sha(seed), true))};
}
export async function sign(wallet: Wallet, body: Record<string, unknown>) {
  return {body, public_key: wallet.public_key,
    signature: hex(secp256k1.sign(utf8.encode(canonical(body)), await sha(wallet.seed), {format: 'der', prehash: true}))};
}
export async function requestId(envelope: {body: Record<string, unknown>; public_key: string}) {
  return hex(await sha(canonical({body: envelope.body, public_key: envelope.public_key})));
}
export function encodedTransaction(value: unknown) {
  return btoa(canonical(value));
}
export function validatePublicKey(value: string) {
  secp256k1.Point.fromBytes(bytes(value));
}
