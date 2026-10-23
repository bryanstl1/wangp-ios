import {Checkpoint,checkAbort} from './checkpoint.js';
import {ChunkCache} from './storage.js';
import {GpuRuntime} from './gpu.js';
import {createProvider,runWanBlock,WAN_CONFIG,validateBlock} from './wan-block.js';
import {compare,linearCPU} from './math.js';
import {verifiedProvider} from './integrity.js';
const $=id=>document.getElementById(id);
const REVISION='37ec512624d61f7aa208f7ea8140a131f93afc9a';
const OFFICIAL=`https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/${REVISION}/diffusion_pytorch_model.safetensors`;
const HEADER_HASH='2ec9c33e0081ab18b8faadfc85efb5d370b74778000ac1c2180c9b66613ef0f0';
let checkpoint=null,controller=null,cache=null,result=null,compatible=false;
const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
function show(value){result={application:'WanGP Safari 0.7.0',timestamp:new Date().toISOString(),videoGeneration:false,...value};$('report').textContent=JSON.stringify(result,null,2);$('save-report').disabled=false;}
function status(text){$('operation-status').textContent=text;}
function clearModel(){checkpoint=null;compatible=false;$('block').disabled=true;$('checkpoint-status').textContent='Ningún checkpoint inspeccionado.';}
function busy(active){for(const id of ['inspect','diagnose','official','model-url','model-file','clear-file','cache','clear-cache'])$(id).disabled=active;$('cancel').disabled=!active;$('block').disabled=active||!compatible;}
async function operation(fn){if(controller)return;controller=new AbortController();const signal=controller.signal;busy(true);$('progress').removeAttribute('value');try{await fn(signal);}catch(e){status(e.name==='AbortError'?'Operación cancelada. Puedes volver a intentarlo.':'Error: '+e.message);show({status:e.name==='AbortError'?'cancelled':'error',message:e.message});}finally{controller=null;busy(false);if(!$('progress').hasAttribute('value'))$('progress').value=0;}}
async function getCache(){cache??=await ChunkCache.open();return cache;}
$('cancel').onclick=()=>{controller?.abort();status('Cancelando; esperando que termine la operación GPU en curso…');};
$('official').onclick=()=>{$('model-url').value=OFFICIAL;$('model-file').value='';clearModel();$('checkpoint-status').textContent='Wan 2.1 T2V 1.3B · revisión oficial fijada. Pulsa Inspeccionar modelo.';};
$('model-url').addEventListener('input',clearModel);$('model-file').addEventListener('change',clearModel);
$('clear-file').onclick=()=>{$('model-file').value='';clearModel();};
$('inspect').onclick=()=>operation(async signal=>{
  clearModel();const source=$('model-file').files[0]||$('model-url').value.trim();if(!source)throw Error('Selecciona un archivo .safetensors o usa el checkpoint oficial.');
  let storage;const pinned=source===OFFICIAL;
  if($('cache').checked&&pinned){try{storage=await getCache();}catch(e){status('Caché no disponible: '+e.message);}}
  status('Leyendo y validando cabecera; no se descarga el checkpoint entero.');
  const cp=await Checkpoint.open(source,{signal,cache:storage,onCacheError:e=>status('Caché no disponible; lectura directa: '+e.message)});
  // Pin the original official header as well as all required block tensor shapes.
  const header=await (source instanceof Blob?source.slice(8,8+cp.headerBytes).arrayBuffer():null);
  let sha;
  if(header)sha=await hash(header);
  else {const {readRange}=await import('./checkpoint.js');sha=await hash((await readRange(source,8,7+cp.headerBytes,{signal,cache:storage})).bytes);}
  checkAbort(signal);checkpoint=cp;
  try{validateBlock(cp,WAN_CONFIG);compatible=sha===HEADER_HASH;}catch{compatible=false;}
  $('checkpoint-status').textContent=`${cp.tensors.size} tensores · ${(cp.total/1e9).toFixed(2)} GB en el archivo · ${compatible?'cabecera oficial compatible; pesos por verificar':'solo inspección; bloque oficial no reconocido'}`;
  show({status:'checkpoint-inspected',source:source instanceof Blob?'local-file':pinned?'official-pinned':'custom-url',...cp.summary(),headerSha256:sha,blockCompatible:compatible,firstTensors:[...cp.tensors.values()].slice(0,12).map(({name,dtype,shape})=>({name,dtype,shape}))});
  $('progress').value=1;status('Cabecera validada. '+(compatible?'Puedes verificar el bloque real de Wan.':'La generación y la prueba del bloque siguen desactivadas para este archivo.'));
});
$('diagnose').onclick=()=>operation(async signal=>{
  status('Compilando WGSL y comparando cálculo local con CPU…');const gpu=await GpuRuntime.create();
  try{checkAbort(signal);const x=new Float32Array([1,2,3,4]),w=new Float32Array([1,0,0,1,2,3]),b=new Float32Array([1,2,3]);const y=await gpu.linear(x,w,b,2,2,3,signal);const comparison=compare(y,linearCPU(x,w,b,2,2,3),1e-6,1e-6);
    $('capabilities').replaceChildren();const capabilities={'WebGPU':comparison.passed?'Cálculo verificado':'Diferencia numérica','Buffer máximo':`${(gpu.limits.maxBufferSize/1048576).toFixed(0)} MiB`,'Binding de almacenamiento':`${(gpu.limits.maxStorageBufferBindingSize/1048576).toFixed(0)} MiB`,'shader-f16':gpu.features.includes('shader-f16')?'Disponible (no usado)':'No disponible','Memoria libre de Safari':'No expuesta por API','Temperatura':'No expuesta por API'};
    for(const [key,value] of Object.entries(capabilities)){const div=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;div.append(dt,dd);$('capabilities').append(div);}
    show({status:comparison.passed?'gpu-verified':'numerical-failure',operation:'synthetic 2x2 linear diagnostic; not Wan inference',comparison,limits:gpu.limits,liveBuffers:gpu.liveBuffers});$('progress').value=1;status(comparison.passed?'WebGPU ejecutó el cálculo y coincidió con CPU.':'La verificación numérica falló.');
  }finally{gpu.destroy();}
});
$('block').onclick=()=>operation(async signal=>{
  if(!checkpoint||!compatible)throw Error('Inspecciona primero el checkpoint oficial compatible.');
  checkpoint.options.signal=signal;
  const fixture=await(await fetch('./tests/fixtures/official-block-reference.json',{signal})).json();
  const manifest=await(await fetch('./tests/fixtures/official-block-chunks.json',{signal})).json();
  if(manifest.revision!==REVISION||manifest.headerSha256!==HEADER_HASH)throw Error('Manifest de integridad incompatible');
  const provider=verifiedProvider(createProvider(checkpoint),manifest),gpu=await GpuRuntime.create();
  const stages=['self_attn.q','self_attn.k','self_attn.v','self_attn.o','cross_attn.q','cross_attn.k','cross_attn.v','cross_attn.o','ffn.0','ffn.2'];
  const start=performance.now(),before=checkpoint.downloaded;status('Ejecutando bloque 0 con pesos oficiales y entradas de diagnóstico…');
  try{const out=await runWanBlock({x:new Float32Array(fixture.inputs.x.data),context:new Float32Array(fixture.inputs.context.data),e:new Float32Array(fixture.inputs.e.data),grid:fixture.config.grid,config:WAN_CONFIG,provider,gpu,signal,onProgress:p=>{const i=stages.indexOf(p.stage);if(i>=0){$('progress').value=(i+p.completed/p.total)/stages.length;status(`${p.stage} · ${p.completed}/${p.total} filas · ${((checkpoint.downloaded-before)/1048576).toFixed(1)} MiB leídos (red o caché)`);}}});
    const comparison=compare(out,fixture.output.data,2e-4,2e-4);
    show({status:comparison.passed?'official-block-verified':'numerical-failure',operation:'Wan2.1 blocks.0 forward pass (not denoising, text-to-video or video generation)',weights:{model:'Wan-AI/Wan2.1-T2V-1.3B',revision:REVISION,headerSha256:HEADER_HASH,allBlockChunksSha256Verified:true},inputs:'known artificial activations from independent PyTorch fixture; prompt not used',backend:'WebGPU linear projections; JS norms, RoPE, attention, GELU and residuals',tokens:2,contextTokens:3,comparison,elapsedMs:Math.round(performance.now()-start),bytesRead:checkpoint.downloaded-before,peakGPUBufferBytes:gpu.peakBufferBytes,liveBuffers:gpu.liveBuffers,outputFirst:Array.from(out.slice(0,12)),outputSha256:await hash(out)});
    $('progress').value=1;status(comparison.passed?'Bloque oficial verificado frente a PyTorch. Esto todavía no es generación de video.':'El bloque se ejecutó, pero no coincidió con la referencia. No se considera verificado.');
  }finally{gpu.destroy();}
});
$('save-report').onclick=()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='wangp-safari-report.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);};
$('clear-cache').onclick=()=>operation(async()=>{const c=await getCache();await c.clear();status('Caché de pesos eliminada.');show({status:'cache-cleared',bytes:await c.usage()});});
try{const draft=JSON.parse(localStorage.getItem('wangp-draft')||'{}');for(const id of ['prompt','resolution','frames'])if(draft[id])$(id).value=draft[id];}catch{}
for(const id of ['prompt','resolution','frames'])$(id).addEventListener('input',()=>{try{localStorage.setItem('wangp-draft',JSON.stringify(Object.fromEntries(['prompt','resolution','frames'].map(id=>[id,$(id).value]))));}catch{}});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&controller){controller.abort();status('Operación cancelada al pasar la página a segundo plano.');}});
window.addEventListener('pagehide',()=>{controller?.abort();cache?.close();cache=null;if(checkpoint)checkpoint.options.cache=null;});
document.documentElement.dataset.ready='true';
