export type Network = {
  chain_id: string; height: number; round: number; model_root: string;
  validator_count: number; issued: string; burned: string; initial_supply: string;
  profile: string; ready: boolean; stalled: boolean; genesis_sha256: string; latest_block_hash: string; latest_block_height: number;
  latest_block_time: string; bootstrap_peers: string[];
};
export type Block = { height: number; hash: string; time: string; app_hash: string; app_state_height: number;
  transactions: { hash: string; kind: string; sender: string; code: number; log: string; body: Record<string, unknown> }[] };

export const asset = (path: string) => import.meta.env.BASE_URL + path.replace(/^\//, '');

export async function get<T>(path: string): Promise<T> {
  const response = await fetch(asset(path), { signal: AbortSignal.timeout(10000), cache: 'no-store' });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Node returned ${response.status}`);
  return value as T;
}
export function neuro(value: string): string {
  const atoms = BigInt(value);
  const fraction = (atoms % 1000000n).toString().padStart(6, '0').replace(/0+$/, '');
  return (atoms / 1000000n).toLocaleString() + (fraction ? `.${fraction}` : '');
}
export function short(value: string) { return value.length > 24 ? `${value.slice(0, 10)}…${value.slice(-10)}` : value; }
