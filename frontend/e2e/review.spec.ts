import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("desktop review links events, URL, evidence, keyboard and JSON export", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/review/demo-violation");
  await expect(page.getByRole("heading", { name: "Требует внимания" })).toBeVisible();
  const stage = await page.locator(".video-stage").boundingBox();
  const image = await page.locator(".hero-evidence img").boundingBox();
  expect(image!.height).toBeLessThanOrEqual(stage!.height + 1);
  await expect(page.locator(".hero-evidence img")).toHaveCSS("object-fit", "contain");
  await page.getByRole("button", { name: /^Отметка 2:/ }).click();
  await expect(page).toHaveURL(/event=f-1-/);
  await expect(page.getByRole("region", { name: "Рабочее место без сотрудника" })).toBeVisible();
  await page.getByRole("button", { name: "Следующее событие" }).click();
  await expect(page.getByRole("region", { name: "Событие для проверки" })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Отметка 3:/ })).toHaveAttribute("aria-pressed", "true");
  await page.reload();
  await expect(page.getByRole("region", { name: "Событие для проверки" })).toBeVisible();
  await page.getByRole("button", { name: /^Отметка 1:/ }).click();
  const open = page.getByRole("button", { name: "Увеличить кадр выбранного события" });
  await open.click();
  const dialog = page.getByRole("dialog", { name: "Кадры · Хранение наличных" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Закрыть" })).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await expect(dialog.getByText("2 / 3", {exact:true})).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(open).toBeFocused();
  await page.getByLabel("Поиск событий").fill("нет такого описания");
  await expect(page.getByText("По этому запросу ничего нет.")).toBeVisible();
  await page.getByRole("button", { name: "Сбросить фильтры" }).click();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", {name:"Скачать JSON"}).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("fokus-demo-violation.json");
  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const result = JSON.parse(Buffer.concat(chunks).toString());
  expect(result.job_id).toBe("demo-violation");
  expect(result.analysis_provider).toBe("synthetic-demo");
  expect(result.violations).toHaveLength(3);
  await page.evaluate(()=>window.scrollTo({top:0,behavior:"instant"}));
  await page.screenshot({path:"test-results/review-desktop.png",fullPage:true});
  expect(errors).toEqual([]);
});

test("processing transitions to done and three terminal states remain distinct", async ({ page }) => {
  await page.goto("/review/demo-processing");
  await expect(page.getByText("Точный прогресс анализа сервис не сообщает.")).toBeVisible();
  await expect(page.getByRole("progressbar")).toHaveCount(0);
  await expect(page.getByRole("heading",{name:"Требует внимания"})).toBeVisible({timeout:10_000});
  for (const [path, title] of [
    ["demo-no-violation", "Нарушения не обнаружены"],
    ["demo-insufficient", "Недостаточно данных"],
    ["demo-failed", "Не удалось разобрать запись."],
  ]) {
    await page.goto(`/review/${path}`);
    await expect(page.getByRole("heading",{name:title,exact:true})).toBeVisible();
    await expect(page.getByRole("button",{name:/^Отметка /})).toHaveCount(0);
  }
});

test("mobile and keyboard-only navigation preserve working controls", async ({ page }) => {
  await page.setViewportSize({width:390,height:844});
  await page.emulateMedia({reducedMotion:"reduce"});
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(page.locator(".skip-link")).toBeFocused();
  for(let i=0;i<3;i++) await page.keyboard.press("Tab");
  await expect(page.getByRole("button", {name:/Как это работает/})).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", {name:/Как это работает/})).toBeFocused();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({path:"test-results/home-mobile.png",fullPage:true});
  await page.goto("/review/demo-violation");
  await page.getByRole("button", {name:/^Отметка 3:/}).click();
  await expect(page.getByRole("region",{name:"Событие для проверки"})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.evaluate(()=>window.scrollTo({top:0,behavior:"instant"}));
  await page.screenshot({path:"test-results/review-mobile.png",fullPage:true});
});

test("main screens pass automated accessibility checks", async ({ page }) => {
  for (const path of ["/", "/review/demo-violation", "/review/demo-no-violation", "/review/demo-failed"]) {
    await page.goto(path);
    if (path !== "/") await page.locator(".demo-banner").waitFor();
    else await page.locator(".service-state:not([data-state=checking])").waitFor();
    const result = await new AxeBuilder({page}).withTags(["wcag2a","wcag2aa","wcag21aa"]).analyze();
    expect(result.violations.map(({id,nodes})=>({id,nodes:nodes.map(node=>({html:node.html,summary:node.failureSummary}))}))).toEqual([]);
  }
});

test("unknown job has a recoverable state without a fabricated failed verdict", async ({ page, request }) => {
  const service = await request.get("/api/review/service");
  test.skip((await service.json()).health !== "online", "This check requires the local FastAPI service.");
  await page.goto("/review/00000000-0000-0000-0000-000000000000");
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Задание не найдено");
  await expect(page.getByRole("button",{name:"Проверить снова"})).toBeVisible();
  await expect(page.getByRole("heading",{name:"Не удалось разобрать запись."})).toHaveCount(0);
});

test("real mock backend accepts a MOV and serves its report after reload", async ({ page }) => {
  test.skip(!process.env.E2E_MOV_PATH, "Set E2E_MOV_PATH to a safe synthetic MOV and explicitly enable backend mock.");
  await page.goto("/");
  await expect(page.getByText("Подключён демоанализатор",{exact:true})).toBeVisible();
  await page.getByLabel("Выбрать видео MOV",{exact:true}).setInputFiles(process.env.E2E_MOV_PATH!);
  await page.getByRole("button",{name:"Запустить демоанализ"}).click();
  await expect(page).toHaveURL(/\/review\/[a-f0-9-]{36}$/);
  await expect(page.getByRole("heading",{name:"Требует внимания"})).toBeVisible();
  await expect(page.locator(".demo-banner")).toContainText("mock provider");
  const data = await page.evaluate(()=>JSON.parse(localStorage.getItem("fokus.recent-jobs.v1") || "[]"));
  expect(Object.keys(data[0]).sort()).toEqual(["createdAt","filename","jobId","status"]);
  const video = page.getByLabel("Исходное видео",{exact:true});
  await expect(video).toBeVisible();
  await expect.poll(()=>video.evaluate((element)=>(element as HTMLVideoElement).readyState)).toBeGreaterThan(0);
  await page.getByRole("button",{name:/^Отметка 2:/}).click();
  await expect.poll(()=>video.evaluate((element)=>(element as HTMLVideoElement).currentTime)).toBeCloseTo(4,1);
  await page.reload();
  await expect(page.getByRole("heading",{name:"Требует внимания"})).toBeVisible();
  await expect(page.getByLabel("Исходное видео",{exact:true})).toHaveCount(0);
  await expect(page.getByRole("button",{name:"Увеличить кадр выбранного события"})).toBeVisible();
  await page.getByLabel("Выбрать исходное видео для просмотра").setInputFiles(process.env.E2E_MOV_PATH!);
  await expect(page.getByText("Файл выбран локально. Соответствие заданию не проверено.")).toBeVisible();
});
