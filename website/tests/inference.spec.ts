import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
const releaseNetwork = JSON.parse(readFileSync(new URL('../../src/neuroshard/client/networks/llm-testnet.json', import.meta.url), 'utf8'));

const root = 'a'.repeat(64);
const key = '0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798'; // gitleaks:allow -- public secp256k1 generator point
const quote = {chain_id: releaseNetwork.chain_id, genesis_sha256: releaseNetwork.genesis_sha256,
  model_root: root, model_name: 'SmolLM2-135M-Instruct + NeuroShard adapter', provider: key,
  provider_online: true, fee_atoms: '1000', price_per_max_token_atoms: '1000', max_tokens: 64,
  request_lifetime_blocks: 240, pending_requests: 0};

test('browser signatures and Unicode canonical encoding verify in the native Python protocol', async ({page}) => {
  await page.goto('/chat');
  const envelope = await page.evaluate(async () => {
    const cryptoModule = await import('/src/native/wallet.ts');
    const wallet = await cryptoModule.createWallet();
    const body = {kind: 'fixture', nonce: 0, text: 'café 😀 \u007f', nested: {b: 2, a: 1}};
    return {envelope: await cryptoModule.sign(wallet, body), canonical: cryptoModule.canonical(body)};
  });
  const python = process.env.NEUROSHARD_TEST_PYTHON || 'python3';
  const result = execFileSync(python, ['-c', 'import json,sys; from neuroshard.core.crypto.ecdsa import ecdsa_verify; v=json.load(sys.stdin); e=v["envelope"]; c=json.dumps(e["body"],sort_keys=True,separators=(",",":")); assert c==v["canonical"]; assert ecdsa_verify(c,e["signature"],bytes.fromhex(e["public_key"])); print("verified")'],
    {input: JSON.stringify(envelope), env: {...process.env, PYTHONPATH: '../src'}}).toString();
  expect(result.trim()).toBe('verified');
});

test('a new browser key cannot spend an invented balance or leak its seed', async ({page}) => {
  let sent = '';
  await page.route('**/api/**', route => {
    const url = new URL(route.request().url());
    return route.fulfill({json: url.pathname === '/api/inference' ? quote : url.pathname === '/api/account' ? {balance: '0',locked:'0',nonce:0} : {...releaseNetwork,height:10,ready:true}});
  });
  await page.route('**/rpc', route => {sent += route.request().postData(); return route.fulfill({json:{result:{code:0}}});});
  await page.goto('/chat'); await page.getByRole('button',{name:'Create a new local key'}).click();
  await expect(page.getByText('0 NEURO',{exact:true})).toBeVisible();
  await page.getByLabel('Your prompt').fill('Hello');
  await page.getByRole('checkbox').check();
  await expect(page.getByRole('button',{name:'Sign and pay'})).toBeDisabled();
  expect(sent).toBe('');
  const stored = await page.evaluate(() => ({local:{...localStorage},session:{...sessionStorage}}));
  expect(JSON.stringify(stored)).not.toContain('seed');
});

test('unknown transport outcome retains a single request ID without a duplicate payment', async ({page}) => {
  let broadcasts = 0;
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname;
    return route.fulfill({status:path === '/api/inference/request' ? 404:200,json:path === '/api/inference' ? quote : path === '/api/account' ? {balance:'1000000',locked:'0',nonce:0} : {...releaseNetwork,height:10,ready:true}});
  });
  await page.route('**/rpc', route => {broadcasts++; return route.abort('connectionreset');});
  await page.goto('/chat'); await page.getByRole('button',{name:'Create a new local key'}).click();
  const download = page.waitForEvent('download'); await page.getByRole('button',{name:'Download key backup'}).click(); await download;
  await page.getByLabel('Your prompt').fill('Hello'); await page.getByRole('checkbox').check();
  await page.getByRole('button',{name:'Sign and pay'}).click();
  await expect(page.getByRole('alert')).toContainText('check its status');
  await expect(page.getByRole('button',{name:'Waiting for settlement'})).toBeDisabled();
  expect(broadcasts).toBe(1);
  await page.reload();
  await expect(page.getByRole('heading',{name:'Request pending'})).toBeVisible();
  await expect(page.getByRole('button',{name:'Create a new local key'})).toBeVisible();
});
