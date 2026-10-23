import {readFile} from 'node:fs/promises';import {createHash} from 'node:crypto';import {execFile} from 'node:child_process';import {promisify} from 'node:util';
const run=promisify(execFile),base=process.env.PAGES_URL||'https://bryanstl1.github.io/wangp-ios/',attempts=Number(process.env.PAGES_ATTEMPTS||30);
const files=['index.html','src/app.js','src/checkpoint.js','src/math.js','src/gpu.js','src/wan-block.js','src/storage.js','src/integrity.js','src/style.css','tests/fixtures/official-block-reference.json','tests/fixtures/official-block-chunks.json','legacy-lab.html','checkpoint-inspector.html','build.json'];
const sha=buffer=>createHash('sha256').update(buffer).digest('hex');const expected=await Promise.all(files.map(async file=>({file,sha256:sha(await readFile(file))})));
let passed=false,errors=[];
for(let attempt=1;attempt<=attempts;attempt++){
 errors=[];for(const item of expected){try{const url=new URL(item.file,base);url.searchParams.set('verify',process.env.GITHUB_SHA||String(Date.now()));const {stdout}=await run('curl',['--fail','--silent','--show-error','--max-time','20','-H','Cache-Control: no-cache',url.href],{encoding:'buffer',maxBuffer:4*1024*1024});if(sha(stdout)!==item.sha256)errors.push(item.file+': hash differs');}catch(e){errors.push(item.file+': '+e.message.split('\n')[0]);}}
 if(errors.length===0){passed=true;break;}console.log(`Pages attempt ${attempt}/${attempts}: ${errors.slice(0,3).join('; ')}`);if(attempt<attempts)await new Promise(r=>setTimeout(r,10000));
}
if(!passed)throw Error('Published files not verified: '+errors.join('; '));
console.log(JSON.stringify({passed:true,url:base,files:expected.length,method:'HTTP200 and SHA256 exact comparison with checkout',timestamp:new Date().toISOString()},null,2));
