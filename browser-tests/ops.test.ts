import { test, expect } from '@playwright/test';

test('qkv operation', async ({ page }) => {
    await page.goto('/browser-tests/qkv.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});

test('rope operation', async ({ page }) => {
    await page.goto('/browser-tests/rope.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});

test('attentionMask operation', async ({ page }) => {
    await page.goto('/browser-tests/attentionMask.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});

test('gatherSub operation', async ({ page }) => {
    await page.goto('/browser-tests/gatherSub.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});

test('scatterSub operation', async ({ page }) => {
    await page.goto('/browser-tests/scatterSub.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});

test('fusedSoftmax operation', async ({ page }) => {
    await page.goto('/browser-tests/fusedSoftmax.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});

test('rmsNorm operation', async ({ page }) => {
    await page.goto('/browser-tests/normRMS.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});

test('rmsNorm Grad operation', async ({ page }) => {
    await page.goto('/browser-tests/normRMSGrad.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});

test('appendCache operation', async ({ page }) => {
    await page.goto('/browser-tests/appendCache.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});

test('gelu operation', async ({ page }) => {
    await page.goto('/browser-tests/gelu.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS');
});
