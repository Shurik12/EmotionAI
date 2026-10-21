const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','preview','--host','127.0.0.1','--port','3005'],{stdio:['ignore','pipe','pipe']});
process.on('exit',()=>server.kill());
(async()=>{
 await new Promise((resolve,reject)=>{server.stdout.on('data',b=>{if(b.toString().includes('http://'))resolve();});server.on('exit',()=>reject(Error('Server stopped')));setTimeout(()=>reject(Error('Server startup timeout')),15000).unref();});
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,headless:true,args:['--no-sandbox','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});
 const context=await browser.newContext({permissions:['camera','microphone']});const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:3005/#/screens/P03');await page.locator('h1').waitFor();
 await page.locator('[data-action="role:doctor"]').click();await page.waitForURL('**/screens/D01');assert.match(await page.locator('h1').textContent(),/Задачи/);
 await page.locator('[data-action="role:patient"]').click();await page.locator('[data-action="catalog"]').first().click();assert.equal(await page.locator('.rz-screenlink').count(),68);
 await page.locator('[data-action="nav:P04"]').click();await page.locator('input[data-bind="form.sleep"]').fill('8');await page.locator('[data-action="nav:P05"]').click();
 await page.goBack();await page.locator('h1').waitFor();assert.equal(await page.locator('input[data-bind="form.sleep"]').inputValue(),'8');
 await page.goto('http://127.0.0.1:3005/#/screens/D04');await page.locator('h1').waitFor();
 await page.setViewportSize({width:390,height:844});await page.goto('http://127.0.0.1:3005/#/screens/P03');await page.locator('h1').waitFor();assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 await page.goto('http://127.0.0.1:3005/#/capture');await page.locator('input[type="checkbox"]').check();await page.getByRole('button',{name:'Начать запись',exact:true}).click();await page.getByRole('button',{name:'Остановить',exact:true}).waitFor();await page.waitForTimeout(1200);await page.getByRole('button',{name:'Остановить',exact:true}).click();await page.getByRole('link',{name:'Скачать запись'}).waitFor();
 assert.equal(errors.length,0,errors.join('\n'));console.log('PASS: role switch, 68 screen catalog, form state, browser Back, direct link, mobile width, real recorder with fake device.');await browser.close();server.kill();
})().catch(e=>{console.error(e);process.exit(1)});
