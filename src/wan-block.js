import {checkAbort} from './checkpoint.js';
import {linearCPU,normalize,attention,rope,gelu,finite} from './math.js';
export const WAN_CONFIG=Object.freeze({dim:1536,numHeads:12,ffnDim:8960,eps:1e-6});
export function blockShapes(config=WAN_CONFIG) {
  const {dim:d,ffnDim:f,numHeads:h,eps}=config;
  if(![d,f,h].every(v=>Number.isInteger(v)&&v>0)||d%h||d/h%2||!Number.isFinite(eps)||eps<=0)throw Error('Invalid Wan block config');
  const shapes={'modulation':[1,6,d],'norm3.weight':[d],'norm3.bias':[d]};
  for(const group of ['self_attn','cross_attn']) {
    for(const op of ['q','k','v','o']){shapes[`${group}.${op}.weight`]=[d,d];shapes[`${group}.${op}.bias`]=[d];}
    for(const op of ['norm_q','norm_k'])shapes[`${group}.${op}.weight`]=[d];
  }
  Object.assign(shapes,{'ffn.0.weight':[f,d],'ffn.0.bias':[f],'ffn.2.weight':[d,f],'ffn.2.bias':[d]});return shapes;
}
export function validateBlock(checkpoint,config=WAN_CONFIG,prefix='blocks.0.') {
  for(const [name,shape] of Object.entries(blockShapes(config))) {
    const t=checkpoint.tensors?.get(prefix+name);
    if(!t||JSON.stringify(t.shape)!==JSON.stringify(shape)||!['F32','F16','BF16'].includes(t.dtype))throw Error('Missing/incompatible tensor: '+prefix+name);
  }return true;
}
export function createProvider(checkpoint,config=WAN_CONFIG,prefix='blocks.0.') {
  validateBlock(checkpoint,config,prefix);
  return {get:name=>checkpoint.read(prefix+name),read:(name,options)=>checkpoint.read(prefix+name,options)};
}
// Weight row streaming. At most chunkRows*inDim weights are decoded at once.
export async function streamedLinear(provider,name,x,rows,k,n,{gpu,signal,chunkRows=128,onProgress}={}) {
  checkAbort(signal);finite(x);const bias=await provider.get(name+'.bias');
  if(bias.length!==n)throw Error('Linear bias shape mismatch');finite(bias);const out=new Float32Array(rows*n);
  for(let start=0;start<n;start+=chunkRows){checkAbort(signal);const count=Math.min(chunkRows,n-start);
    const w=await provider.read(name+'.weight',{elementOffset:start*k,count:count*k});finite(w);
    const b=bias.slice(start,start+count),y=gpu?await gpu.linear(x,w,b,rows,k,count,signal):linearCPU(x,w,b,rows,k,count);
    for(let r=0;r<rows;r++)out.set(y.subarray(r*count,(r+1)*count),r*n+start);
    onProgress?.({stage:name,completed:start+count,total:n});
    // Yield to cancellation/UI between finite GPU submissions or CPU chunks.
    await new Promise(resolve=>setTimeout(resolve,0));
  }finite(out);return out;
}
export async function runWanBlock({x,context,e,grid,config=WAN_CONFIG,provider,gpu,signal,onProgress,linear}) {
  blockShapes(config);const {dim:d,numHeads:h,ffnDim:f,eps}=config;
  const tokens=grid?.reduce((a,b)=>a*b,1),ct=context?.length/d;
  if(!grid||grid.length!==3||grid.some(v=>!Number.isInteger(v)||v<1)||tokens>64||!Number.isInteger(ct)||ct<1||ct>512||x.length!==tokens*d||e.length!==6*d)throw Error('Invalid block activations (diagnostic limited to 64 tokens)');
  [x,context,e].forEach(finite);checkAbort(signal);
  const get=async name=>{checkAbort(signal);const a=await provider.get(name);finite(a);return a;};
  const lin=linear??((name,v,r,k,n)=>streamedLinear(provider,name,v,r,k,n,{gpu,signal,onProgress}));
  const modulation=await get('modulation');if(modulation.length!==6*d)throw Error('Modulation shape mismatch');
  const m=Float32Array.from(e,(v,i)=>v+modulation[i]);
  const mod=(v,shift,scale)=>Float32Array.from(v,(a,i)=>a*(1+m[scale*d+i%d])+m[shift*d+i%d]);
  const residual=(a,b,gate=null)=>Float32Array.from(a,(v,i)=>v+b[i]*(gate===null?1:m[gate*d+i%d]));
  async function attn(name,a,c,rotary) {
    const q=normalize(await lin(name+'.q',a,tokens,d,d),d,eps,await get(name+'.norm_q.weight'),null,true);
    const k=normalize(await lin(name+'.k',c,c.length/d,d,d),d,eps,await get(name+'.norm_k.weight'),null,true);
    const v=await lin(name+'.v',c,c.length/d,d,d);
    const y=attention(rotary?rope(q,grid,h,d/h):q,rotary?rope(k,grid,h,d/h):k,v,tokens,c.length/d,h,d/h);
    return lin(name+'.o',y,tokens,d,d);
  }
  const normalized=mod(normalize(x,d,eps),0,1);
  let out=residual(x,await attn('self_attn',normalized,normalized,true),2);
  onProgress?.({stage:'self_attention',completed:1,total:1});checkAbort(signal);
  out=residual(out,await attn('cross_attn',normalize(out,d,eps,await get('norm3.weight'),await get('norm3.bias')),context,false));
  const ff=gelu(await lin('ffn.0',mod(normalize(out,d,eps),3,4),tokens,d,f));
  out=residual(out,await lin('ffn.2',ff,tokens,f,d),5);finite(out);checkAbort(signal);return out;
}
