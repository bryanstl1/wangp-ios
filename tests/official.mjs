// Actual official block-zero weights + artificial activations; no diffusion/video inference.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile,open,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {chromium} from '@playwright/test';
import {serve} from '../scripts/serve.mjs';
import {runWanBlock,blockShapes} from '../src/wan-block.js';
import {compare} from '../src/math.js';
import {verifiedProvider} from '../src/integrity.js';

const weightsDir=process.env.WAN_WEIGHTS_DIR??'/tmp/wan-official';
const referencePath=process.env.WAN_REFERENCE??'/tmp/wan-real-reference.json';
const reportPath=process.env.WAN_REPORT??'test-results/official.json';
const report={scope:'Official Wan2.1 T2V single block, artificial activations; not video inference',passed:false};
let browser,server;
try {
  const manifest=JSON.parse(await readFile(path.join(weightsDir,'manifest.json'),'utf8'));
  const pinned=JSON.parse(await readFile(new URL('./fixtures/official-block-provenance.json',import.meta.url),'utf8'));
  const reference=JSON.parse(await readFile(referencePath,'utf8'));
  assert.equal(manifest.revision,pinned.revision);
  assert.equal(manifest.headerSha256,pinned.headerSha256);
  assert.equal(reference.provenance.revision,pinned.revision);
  assert.deepEqual(Object.keys(manifest.tensors).sort(),Object.keys(pinned.tensors).sort());
  const paths=new Map();let verifiedBytes=0;
  for(const [name,spec] of Object.entries(pinned.tensors)) {
    assert.deepEqual(manifest.tensors[name],spec,`Manifest differs from pinned provenance: ${name}`);
    assert.equal(spec.dtype,'F32');
    assert.equal(path.basename(spec.file),spec.file);
    const filename=path.join(weightsDir,spec.file);
    const file=await readFile(filename);
    assert.equal(file.byteLength,spec.byteLength,`Wrong tensor length: ${name}`);
    assert.equal(createHash('sha256').update(file).digest('hex'),spec.sha256,`SHA256 mismatch: ${name}`);
    verifiedBytes+=file.byteLength;
    paths.set(name.replace(/^blocks\.0\./,''),{filename,spec});
  }
  // Ensure the model config cannot request a missing or differently shaped tensor.
  for(const [name,shape] of Object.entries(blockShapes(reference.config)))
    assert.deepEqual(paths.get(name)?.spec.shape,shape,`Shape mismatch: ${name}`);
  async function readTensor(name,{elementOffset=0,count}={}) {
    const entry=paths.get(name);assert.ok(entry,`Unknown tensor: ${name}`);
    const elements=entry.spec.byteLength/4;count??=elements-elementOffset;
    assert.ok(Number.isInteger(elementOffset)&&Number.isInteger(count)&&elementOffset>=0&&count>=0&&elementOffset+count<=elements,'Invalid tensor range');
    const buffer=Buffer.alloc(count*4),file=await open(entry.filename,'r');
    try {let consumed=0;while(consumed<buffer.length){const {bytesRead}=await file.read(buffer,consumed,buffer.length-consumed,elementOffset*4+consumed);assert.ok(bytesRead>0,'Truncated tensor');consumed+=bytesRead;}}
    finally {await file.close();}
    return buffer;
  }
  const asFloat=buffer=>new Float32Array(buffer.buffer,buffer.byteOffset,buffer.byteLength/4);
  const chunks=JSON.parse(await readFile(new URL('./fixtures/official-block-chunks.json',import.meta.url),'utf8'));
  const provider=verifiedProvider({get:async name=>asFloat(await readTensor(name)),
    read:async(name,options)=>asFloat(await readTensor(name,options))},chunks);
  report.provenance={model:manifest.model,revision:manifest.revision,tensors:paths.size,verifiedBytes,allSha256Verified:true};
  const args={config:reference.config,grid:reference.config.grid,provider,
    ...Object.fromEntries(Object.entries(reference.inputs).map(([k,v])=>[k,new Float32Array(v.data)]))};
  let started=performance.now();
  const cpu=await runWanBlock(args);
  report.cpu={...compare(cpu,reference.output.data,2e-4,2e-4),milliseconds:performance.now()-started};
  console.log('Official CPU reference:',JSON.stringify(report.cpu));
  assert.ok(report.cpu.passed,'Official CPU output differs from PyTorch');

  server=await serve();
  browser=await chromium.launch({args:['--no-sandbox','--enable-unsafe-webgpu','--use-angle=swiftshader','--enable-features=Vulkan']});
  const page=await browser.newPage();
  await page.route('**/__official/**',async route=>{
    try {
      const url=new URL(route.request().url());
      if(url.pathname==='/__official/reference.json')return route.fulfill({json:reference});
      const name=decodeURIComponent(url.pathname.slice('/__official/'.length));
      const entry=paths.get(name);assert.ok(entry,'Unknown tensor');
      const range=route.request().headers().range;
      const match=/^bytes=(\d+)-(\d+)$/.exec(range??'');assert.ok(match,'Expected bounded Range');
      const lo=Number(match[1]),hi=Number(match[2]);assert.ok(lo%4===0&&(hi+1)%4===0&&hi>=lo&&hi-lo+1<=8*1024*1024,'Invalid range');
      const body=await readTensor(name,{elementOffset:lo/4,count:(hi-lo+1)/4});
      await route.fulfill({status:206,contentType:'application/octet-stream',headers:{'Content-Range':`bytes ${lo}-${hi}/${entry.spec.byteLength}`,'Content-Length':String(body.length)},body});
    }catch(error){await route.fulfill({status:400,body:String(error)});}
  });
  await page.goto(server.url+'/tests/gpu-harness.html');
  report.gpu=await page.evaluate(async()=>{
    const [{runWanBlock},{GpuRuntime},{compare},{verifiedProvider}]=await Promise.all([import('/src/wan-block.js'),import('/src/gpu.js'),import('/src/math.js'),import('/src/integrity.js')]);
    const chunks=await(await fetch('/tests/fixtures/official-block-chunks.json')).json();
    const f=await(await fetch('/__official/reference.json')).json();
    const shapes=(await import('/src/wan-block.js')).blockShapes(f.config);
    const read=async(name,{elementOffset=0,count}={})=>{
      count??=shapes[name].reduce((a,b)=>a*b,1)-elementOffset;
      const response=await fetch('/__official/'+encodeURIComponent(name),{headers:{Range:`bytes=${elementOffset*4}-${(elementOffset+count)*4-1}`}});
      if(response.status!==206)throw Error(await response.text());
      const body=await response.arrayBuffer();if(body.byteLength!==count*4)throw Error('Wrong tensor byte length');
      return new Float32Array(body);
    };
    const gpu=await GpuRuntime.create();
    const started=performance.now();
    try {
      const out=await runWanBlock({config:f.config,grid:f.config.grid,gpu,provider:verifiedProvider({get:name=>read(name),read},chunks),
        ...Object.fromEntries(Object.entries(f.inputs).map(([k,v])=>[k,new Float32Array(v.data)]))});
      return {...compare(out,f.output.data,2e-4,2e-4),milliseconds:performance.now()-started,
        environment:'Chromium SwiftShader; not an iPhone/Safari measurement',limits:gpu.limits,peakBufferBytes:gpu.peakBufferBytes,liveBuffers:gpu.liveBuffers};
    }finally{gpu.destroy();}
  });
  console.log('Official WebGPU reference:',JSON.stringify(report.gpu));
  assert.ok(report.gpu.passed,'Official GPU output differs from PyTorch');
  assert.equal(report.gpu.liveBuffers,0,'GPU buffer leak');
  // End-to-end UI/Checkpoint integration using a transport fixture of real official
  // bytes. This is test transport, not live HF networking or video generation.
  const headerObject=JSON.parse(await readFile(path.join(weightsDir,'header.json'),'utf8'));
  let raw=Buffer.from(JSON.stringify(headerObject));raw=Buffer.concat([raw,Buffer.alloc((8-raw.length%8)%8,32)]);
  assert.equal(createHash('sha256').update(raw).digest('hex'),pinned.headerSha256);
  const prefix=Buffer.alloc(8);prefix.writeBigUInt64LE(BigInt(raw.length));
  const headerBytes=Buffer.concat([prefix,raw]);let tamper=false;
  await page.route('https://huggingface.co/**',async route=>{
    try {
      const match=/^bytes=(\d+)-(\d+)$/.exec(route.request().headers().range??'');assert.ok(match);
      const lo=Number(match[1]),hi=Number(match[2]);let body;
      if(hi<headerBytes.length)body=headerBytes.subarray(lo,hi+1);
      else {
        const entry=[...paths.values()].find(({spec})=>lo>=headerBytes.length+spec.data_offsets[0]&&hi<headerBytes.length+spec.data_offsets[1]);
        assert.ok(entry,'Unexpected checkpoint data request');
        const name=entry.spec.file.replace(/\.bin$/,'').replace(/^blocks\.0\./,'');
        body=await readTensor(name,{elementOffset:(lo-headerBytes.length-entry.spec.data_offsets[0])/4,count:(hi-lo+1)/4});
        if(tamper&&name==='self_attn.q.weight')body[0]^=1;
      }
      await route.fulfill({status:206,headers:{'Content-Type':'application/octet-stream','Content-Range':`bytes ${lo}-${hi}/5676070424`,'Content-Length':String(body.length),'Access-Control-Allow-Origin':'*','Access-Control-Expose-Headers':'Content-Range, Content-Length'},body});
    }catch(error){await route.fulfill({status:400,body:String(error)});}
  });
  await page.goto(server.url);
  await page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  await page.locator('#official').click();await page.locator('#inspect').click();
  await page.waitForFunction(()=>!document.querySelector('#inspect').disabled);
  assert.equal(await page.locator('#block').isDisabled(),false,'Official header must enable block diagnostics');
  await page.locator('#block').click();
  await page.waitForFunction(()=>!document.querySelector('#inspect').disabled,{},{timeout:120000});
  const uiReport=JSON.parse(await page.locator('#report').innerText());
  assert.equal(uiReport.status,'official-block-verified',JSON.stringify(uiReport));
  assert.equal(uiReport.weights.allBlockChunksSha256Verified,true);
  assert.equal(uiReport.liveBuffers,0);assert.equal(uiReport.videoGeneration,false);
  tamper=true;await page.locator('#block').click();
  await page.waitForFunction(()=>!document.querySelector('#inspect').disabled,{},{timeout:120000});
  const tamperedReport=JSON.parse(await page.locator('#report').innerText());
  assert.equal(tamperedReport.status,'error');assert.match(tamperedReport.message,/checksum mismatch/);
  report.ui={passed:true,transport:'fixture of real official checkpoint byte ranges',allChunksVerified:true,tamperedWeightRejected:true,comparison:uiReport.comparison};
  report.passed=true;
}catch(error){report.error=String(error.stack??error);process.exitCode=1;console.error(report.error);}
finally {
  await browser?.close();
  if(server)await new Promise(resolve=>server.server.close(resolve));
  await mkdir(path.dirname(reportPath),{recursive:true});
  await writeFile(reportPath,JSON.stringify(report,null,2)+'\n');
  console.log(`Official block report: ${reportPath}`);
}
