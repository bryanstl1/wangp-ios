import {chromium} from '@playwright/test';import {serve} from '../scripts/serve.mjs';import assert from 'node:assert/strict';
const s=await serve();let browser;
try{
 browser=await chromium.launch({args:['--no-sandbox','--enable-unsafe-webgpu','--use-angle=swiftshader','--enable-features=Vulkan']});
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(s.url+'/tests/gpu-harness.html');await page.waitForFunction(()=>window.checkGPU);
 const result=await page.evaluate(()=>window.checkGPU());assert.equal(result.liveBuffers,0);
 await page.goto(s.url+'/legacy-lab.html');
 for(const [button,out,expected] of [['run','out','GPU COMPUTE FUNCIONAL'],['mat','matout','PRUEBA SUPERADA'],['attention','attout','PRUEBA SUPERADA']]){
  await page.locator('#'+button).click();await page.waitForFunction(({out,expected})=>document.getElementById(out).textContent.includes(expected),{out,expected});
 }
 await page.locator('#quant').click();assert.match(await page.locator('#memout').innerText(),/INT8 CPU local/);assert.deepEqual(errors,[]);
 console.log(JSON.stringify({environment:'Chromium SwiftShader (not iPhone)',...result,legacyComputeMatricesAttentionINT8:'PASS'},null,2));
}finally{await browser?.close();await new Promise(r=>s.server.close(r));}
