import {checkAbort} from './checkpoint.js';
function dimensions(rows,k,n){if(![rows,k,n].every(v=>Number.isInteger(v)&&v>0&&v<=65536))throw Error('Invalid GPU dimensions');}
export function linearWGSL(rows,k,n) {
  dimensions(rows,k,n);
  return `@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=${rows*n}u){return;}let r=i/${n}u;let o=i%${n}u;
 var sum:f32=0.0;for(var j:u32=0u;j<${k}u;j++){sum+=x[r*${k}u+j]*w[o*${k}u+j];}
 y[i]=sum+b[o];}`;
}
export function quantizedWGSL(rows,k,n,bits,groupSize) {
  dimensions(rows,k,n);
  if(![4,8].includes(bits)||!Number.isInteger(groupSize)||groupSize<1||groupSize>65536)throw Error('Invalid packed quantization');
  const perWord=32/bits,mask=2**bits-1,sign=2**(bits-1),range=2**bits;
  return `@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> packed:array<u32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@group(0) @binding(3) var<storage,read> scales:array<f32>;
@group(0) @binding(4) var<storage,read_write> y:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=${rows*n}u){return;}let r=i/${n}u;let o=i%${n}u;
 var sum:f32=0.0;for(var j:u32=0u;j<${k}u;j++){
  let index=o*${k}u+j;let raw=(packed[index/${perWord}u]>>((index%${perWord}u)*${bits}u))&${mask}u;
  let signed=select(i32(raw),i32(raw)-${range},raw>=${sign}u);
  sum+=x[r*${k}u+j]*(f32(signed)*scales[index/${groupSize}u]);
 }y[i]=sum+b[o];}`;
}
export class GpuRuntime {
  static async create() {
    if(!globalThis.navigator?.gpu)throw Error('WebGPU no disponible. Requiere Safari con WebGPU y HTTPS.');
    const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('Sin adaptador WebGPU');
    const device=await adapter.requestDevice();return new GpuRuntime(device,adapter);
  }
  constructor(device,adapter) {
    this.device=device;this.liveBuffers=0;this.peakBufferBytes=0;this.activeBytes=0;this.pipelines=new Map();this.lost=null;this.destroyed=false;
    this.limits=Object.fromEntries(['maxBufferSize','maxStorageBufferBindingSize','maxComputeWorkgroupsPerDimension','maxComputeInvocationsPerWorkgroup'].map(k=>[k,device.limits[k]]));
    this.features=[...adapter.features];
    device.lost.then(info=>{this.lost=info.message||info.reason||'GPU device lost';});
  }
  async linearQuantized(x,q,b,rows,k,n,signal){return this.linear(x,null,b,rows,k,n,signal,q);}
  async linear(x,w,b,rows,k,n,signal,q=null) {
    checkAbort(signal);if(this.destroyed||this.lost)throw Error('GPU no disponible: '+(this.lost||'destruida'));
    if(x.length!==rows*k||(!q&&w.length!==n*k)||b.length!==n)throw Error('GPU linear shape mismatch');
    let values=w;
    if(q){
      quantizedWGSL(rows,k,n,q.bits,q.groupSize);
      if(q.length!==n*k||!(q.data instanceof Uint8Array)||q.data.byteLength!==Math.ceil(q.length*q.bits/8)||!(q.scales instanceof Float32Array)||q.scales.length!==Math.ceil(q.length/q.groupSize)||q.scales.some(v=>!Number.isFinite(v)||v<=0))throw Error('Invalid packed tensor shape/scales');
      values=new Uint32Array(Math.ceil(q.data.byteLength/4));new Uint8Array(values.buffer).set(q.data);
    }
    const code=q?quantizedWGSL(rows,k,n,q.bits,q.groupSize):linearWGSL(rows,k,n),device=this.device,buffers=[];
    if(Math.ceil(rows*n/64)>device.limits.maxComputeWorkgroupsPerDimension)throw Error('GPU dispatch exceeds device limit');
    const make=(size,usage)=>{size=Math.max(4,Math.ceil(size/4)*4);if(size>device.limits.maxBufferSize||usage&GPUBufferUsage.STORAGE&&size>device.limits.maxStorageBufferBindingSize)throw Error('Buffer exceeds GPU limit');const v=device.createBuffer({size,usage});buffers.push(v);this.liveBuffers++;this.activeBytes+=size;this.peakBufferBytes=Math.max(this.peakBufferBytes,this.activeBytes);return v;};
    device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');
    let result,error;
    try {
      const key=`${rows},${k},${n},${q?.bits??32},${q?.groupSize??0}`;let pipeline=this.pipelines.get(key);
      if(!pipeline){const module=device.createShaderModule({code});const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(e=>e.message).join('\n'));pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});if(this.pipelines.size>=32)this.pipelines.clear();this.pipelines.set(key,pipeline);}
      const input=make(x.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),weight=make(values.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),bias=make(b.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),out=make(rows*n*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=make(rows*n*4,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      const uploads=[[input,x],[weight,values],[bias,b]],bindings=[input,weight,bias];
      if(q){const scales=make(q.scales.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);uploads.push([scales,q.scales]);bindings.push(scales);}
      for(const [buffer,a] of uploads)device.queue.writeBuffer(buffer,0,a);
      bindings.push(out);
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:bindings.map((buffer,binding)=>({binding,resource:{buffer}}))});
      const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(rows*n/64));pass.end();enc.copyBufferToBuffer(out,0,read,0,rows*n*4);device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);result=new Float32Array(read.getMappedRange().slice(0));read.unmap();checkAbort(signal);
    } catch(e){error=e;} finally {for(const buffer of buffers){this.activeBytes-=buffer.size;buffer.destroy();this.liveBuffers--;}}
    const validation=await device.popErrorScope(),oom=await device.popErrorScope();
    if(error)throw error;if(validation||oom)throw Error((validation||oom).message);if(this.lost)throw Error(this.lost);return result;
  }
  destroy(){this.destroyed=true;this.pipelines.clear();this.device.destroy();}
}
