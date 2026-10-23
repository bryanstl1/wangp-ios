import {checkAbort} from './checkpoint.js';
export function linearWGSL(rows,k,n) {
  if(![rows,k,n].every(v=>Number.isInteger(v)&&v>0&&v<=65536))throw Error('Invalid GPU dimensions');
  return `@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=${rows*n}u){return;}let r=i/${n}u;let o=i%${n}u;
 var sum:f32=0.0;for(var j:u32=0u;j<${k}u;j++){sum+=x[r*${k}u+j]*w[o*${k}u+j];}
 y[i]=sum+b[o];}`;
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
  async linear(x,w,b,rows,k,n,signal) {
    checkAbort(signal);if(this.destroyed||this.lost)throw Error('GPU no disponible: '+(this.lost||'destruida'));
    if(x.length!==rows*k||w.length!==n*k||b.length!==n)throw Error('GPU linear shape mismatch');
    const code=linearWGSL(rows,k,n),device=this.device,buffers=[];
    if(Math.ceil(rows*n/64)>device.limits.maxComputeWorkgroupsPerDimension)throw Error('GPU dispatch exceeds device limit');
    const make=(size,usage)=>{size=Math.max(4,Math.ceil(size/4)*4);if(size>device.limits.maxBufferSize||usage&GPUBufferUsage.STORAGE&&size>device.limits.maxStorageBufferBindingSize)throw Error('Buffer exceeds GPU limit');const v=device.createBuffer({size,usage});buffers.push(v);this.liveBuffers++;this.activeBytes+=size;this.peakBufferBytes=Math.max(this.peakBufferBytes,this.activeBytes);return v;};
    device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');
    let result,error;
    try {
      const key=`${rows},${k},${n}`;let pipeline=this.pipelines.get(key);
      if(!pipeline){const module=device.createShaderModule({code});const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(e=>e.message).join('\n'));pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});if(this.pipelines.size>=32)this.pipelines.clear();this.pipelines.set(key,pipeline);}
      const input=make(x.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),weight=make(w.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),bias=make(b.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),out=make(rows*n*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),read=make(rows*n*4,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      for(const [buffer,a] of [[input,x],[weight,w],[bias,b]])device.queue.writeBuffer(buffer,0,a);
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,weight,bias,out].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(rows*n/64));pass.end();enc.copyBufferToBuffer(out,0,read,0,rows*n*4);device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);result=new Float32Array(read.getMappedRange().slice(0));read.unmap();checkAbort(signal);
    } catch(e){error=e;} finally {for(const buffer of buffers){this.activeBytes-=buffer.size;buffer.destroy();this.liveBuffers--;}}
    const validation=await device.popErrorScope(),oom=await device.popErrorScope();
    if(error)throw error;if(validation||oom)throw Error((validation||oom).message);if(this.lost)throw Error(this.lost);return result;
  }
  destroy(){this.destroyed=true;this.pipelines.clear();this.device.destroy();}
}
