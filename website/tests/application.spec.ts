import { expect, test, type Page } from '@playwright/test';

import { readFileSync } from 'node:fs';
const releaseNetwork = JSON.parse(readFileSync(new URL('../../src/neuroshard/client/networks/llm-testnet.json', import.meta.url), 'utf8'));

const key = '02' + '1'.repeat(64), root = 'a'.repeat(64);
const network = { chain_id: releaseNetwork.chain_id, height: 70, latest_block_height: 71, round: 2, model_root: root,
  validator_count: 4, total_voting_power: 40, issued: '2000000', burned: '4000', initial_supply: '90000000',
  profile: 'testnet', ready: true, stalled: false, genesis_sha256: releaseNetwork.genesis_sha256, latest_block_hash: 'B'.repeat(64),
  latest_block_time: '2026-09-10T00:00:00Z', bootstrap_peers: ['1'.repeat(40) + '@example.org:26656'] };
const blocks = [{ height: 70, hash: root, time: '2026-09-10T00:00:00Z', app_hash: root, app_state_height: 69,
  transactions: [{ hash: root, kind: 'transfer', sender: key, code: 0, log: '', body: { amount: '9007199254740993' } }] }];
async function fixtures(page: Page) {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    const responses: Record<string, unknown> = {
      '/api/network': network, '/api/blocks': { blocks }, '/api/block/70': blocks[0],
      '/api/account': { public_key: key, balance: '9007199254740993', nonce: 8 },
      '/api/validators': { validators: [{ public_key: root, owner: key, power: 10, bond: '2500000' }], total: 1, next_after: null },
      '/api/model': { ...network, parameter_count: 134519616, base_parameter_count: 134515008, trainable_parameter_count: 4608, serving_root: root, model_name: "SmolLM2-135M-Instruct + NeuroShard adapter", serving_validation_loss: 1.7773, last_training_loss: 4.5,
        execution: { model: {num_layers: 2, hidden_dim: 32, vocab_size: 256}, dataset_sha256: root,
          optimizer: 'SGD-no-momentum', learning_rate: 0.1, batch_size: 4, sequence_length: 32, numerics: 'CPU test fixture' } },
      '/api/training': { records: [{ round: 2, height: 70, loss: 4.5, time: blocks[0].time, model_root: root, transaction_hash: root }],
        indexed_height: 70, chain_height: 70, indexed_rounds: 2, next_before: null, error: null },
    };
    await route.fulfill({ status: responses[url.pathname] ? 200 : 404, json: responses[url.pathname] || {error: 'Unknown fixture'} });
  });
  await page.route('**/work/status', route => route.fulfill({ json: { remaining_tasks: 12, workers_by_stage: [1, 0], active_operations: 0, sponsor: key } }));
}

test('home, model history, and checkpoint are public', async ({page}) => {
  await fixtures(page); await page.goto('/');
  await expect(page.getByText('Connected to native ledger')).toBeVisible();
  await page.getByRole('link', {name: 'Model', exact: true}).click();
  await expect(page.getByText('134,519,616', {exact: true})).toBeVisible();
  await expect(page.getByRole('button', {name: 'Step 2'})).toBeVisible();
  await page.getByRole('button', {name: 'Step 2'}).click();
  await expect(page.getByText('Step 2 · loss 4.500000 · block 70')).toBeVisible();
  await expect(page.getByRole('link', {name: 'Download latest checkpoint'})).toHaveAttribute('href', '/api/model/checkpoint.json');
});

test('ledger links resolve, accepted execution is explicit, and balances retain precision', async ({page}) => {
  await fixtures(page); await page.goto('/ledger?height=70');
  await expect(page.getByText('transfer · Accepted')).toBeVisible();
  await expect(page.getByText('9007199254740993', {exact: false})).toBeVisible();
  await page.getByLabel('Account public key').fill(key);
  await page.getByRole('button', {name: 'Look up account'}).click();
  await expect(page.getByText('9,007,199,254.740993 NEURO', {exact: true})).toBeVisible();
  await expect(page.getByRole('heading', {name: 'Active validators · 1'})).toBeVisible();
});

test('join offers minimal published commands without website registration', async ({page}) => {
  await fixtures(page); await page.goto('/join');
  await expect(page.locator('pre').filter({hasText: 'pip install'})).toContainText('neuroshard-ai');
  await expect(page.locator('pre').filter({hasText: 'neuroshard join'})).toBeVisible();
  await expect(page.getByText('12 task attempts remaining', {exact: false})).toBeVisible();
  await expect(page.locator('input[type=email], input[type=password]')).toHaveCount(0);
});

test('unavailable gateway is shown without invented network values', async ({page}) => {
  await page.route('**/api/network', route => route.fulfill({status: 503, json: {error: 'Node unavailable'}}));
  await page.goto('/'); await expect(page.getByText('Node unavailable', {exact: true})).toBeVisible();
  await expect(page.getByRole('button', {name: 'Retry'})).toBeVisible();
  await expect(page.locator('.native-stats strong')).toHaveText(['—','—','—','—']);
});

test('mobile navigation and sponsor failure remain usable', async ({page}) => {
  await page.setViewportSize({width: 390, height: 844}); await fixtures(page);
  await page.route('**/work/status', route => route.fulfill({status: 503, json: {error: 'Sponsor offline'}}));
  await page.goto('/'); await page.getByRole('button', {name: 'Toggle navigation'}).click();
  await page.locator('nav').getByRole('link', {name: 'Run a node'}).click();
  await expect(page.getByText('Sponsor currently unavailable')).toBeVisible();
  await expect(page.getByRole('heading', {name: 'Your machine. Your keys.'})).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
