import { test, expect } from '@playwright/test';

test('training', async ({ page }) => {
    test.slow();
    await page.goto('/browser-tests/rope-train.html');
    const body = page.locator('body');
    await expect(body).toHaveText('PASS', { timeout: 120000 });
});
