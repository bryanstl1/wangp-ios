// Bounded random access only: never fall back to whole-file HTTP downloads.
export const MAX_HEADER = 16 * 1024 * 1024;
export const MAX_READ = 4 * 1024 * 1024;
const WIDTH = {F64:8,F32:4,F16:2,BF16:2,I64:8,U64:8,I32:4,U32:4,I16:2,U16:2,I8:1,U8:1,BOOL:1,F8_E4M3:1,F8_E5M2:1};
export function checkAbort(signal) { signal?.throwIfAborted(); }
function integer(x) { return Number.isSafeInteger(x) && x >= 0; }
export async function readRange(source, start, end, {signal,fetcher=fetch,cache,onCacheError}={}) {
  checkAbort(signal);
  const length=end-start+1;
  if(!integer(start)||!integer(end)||length<1||length>MAX_HEADER) throw Error('Read range exceeds bound');
  if(source instanceof Blob) {
    if(end>=source.size) throw Error('Truncated local file');
    const bytes=new Uint8Array(await source.slice(start,end+1).arrayBuffer());checkAbort(signal);
    return {bytes,total:source.size};
  }
  const url=new URL(source);
  if(!['https:','http:'].includes(url.protocol)||url.username||url.password) throw Error('Use an HTTP(S) checkpoint URL without credentials');
  const key=`${url.href}#bytes=${start}-${end}`;
  if(cache){try{const hit=await cache.get(key);checkAbort(signal);if(hit&&hit.bytes.length===length&&integer(hit.total)&&hit.total>end)return hit;}catch(e){checkAbort(signal);onCacheError?.(e);}}
  // HF resolve endpoints tolerate a query key; leave signed/custom URLs unchanged.
  if(url.hostname==='huggingface.co'&&url.pathname.includes('/resolve/'))url.searchParams.set('wangp_range',`${start}-${end}`);
  const r=await fetcher(url.href,{headers:{Range:`bytes=${start}-${end}`},signal,cache:'no-store',credentials:'omit',referrerPolicy:'no-referrer'});
  const fail=async msg=>{try{await r.body?.cancel();}catch{} throw Error(msg);};
  if(r.status!==206) return fail(`Expected HTTP 206; received ${r.status}. Full download refused.`);
  const match=/^bytes (\d+)-(\d+)\/(\d+)$/.exec(r.headers.get('Content-Range')||'');
  if(!match||Number(match[1])!==start||Number(match[2])!==end||!integer(Number(match[3]))||Number(match[3])<=end) return fail('Invalid Content-Range (CORS must expose this header)');
  const announced=r.headers.get('Content-Length');
  if(announced!==null&&Number(announced)!==length) return fail('Unexpected Content-Length');
  if(!r.body) throw Error('Missing response body');
  const reader=r.body.getReader(),bytes=new Uint8Array(length);let offset=0;
  try {while(true){checkAbort(signal);const {value,done}=await reader.read();if(done)break;
    if(offset+value.byteLength>length)throw Error('Response length exceeds bound');bytes.set(value,offset);offset+=value.byteLength;}
    if(offset!==length)throw Error('Truncated response length');
  } catch(e) {try{await reader.cancel();}catch{} throw e;} finally{reader.releaseLock();}
  const result={bytes,total:Number(match[3])};
  if(cache){try{await cache.put(key,result);}catch(e){onCacheError?.(e);}}checkAbort(signal);return result;
}
export function validateHeader(header,dataBytes) {
  if(!header||typeof header!=='object'||Array.isArray(header)||!integer(dataBytes))throw Error('Invalid safetensors header');
  const tensors=new Map();
  for(const [name,t] of Object.entries(header)) {
    if(name==='__metadata__') {if(!t||Array.isArray(t)||typeof t!=='object'||Object.values(t).some(v=>typeof v!=='string'))throw Error('Invalid metadata');continue;}
    if(!t||!Object.hasOwn(WIDTH,t.dtype)||!Array.isArray(t.shape)||!t.shape.every(integer)||!Array.isArray(t.data_offsets)||t.data_offsets.length!==2) throw Error(`Invalid tensor descriptor: ${name}`);
    let elements=1;for(const d of t.shape){elements*=d;if(!integer(elements))throw Error(`Unsafe shape: ${name}`);}
    const [a,b]=t.data_offsets,size=elements*WIDTH[t.dtype];
    if(!integer(a)||!integer(b)||!integer(size)||b<a||b>dataBytes||b-a!==size)throw Error(`Invalid tensor offsets/size: ${name}`);
    tensors.set(name,{name,dtype:t.dtype,shape:t.shape.slice(),start:a,end:b,elements,width:WIDTH[t.dtype]});
  }
  let cursor=0;
  for(const t of [...tensors.values()].sort((a,b)=>a.start-b.start||a.end-b.end)) {
    if(t.start!==cursor)throw Error('Tensor overlap or hole in checkpoint');cursor=t.end;
  }
  if(cursor!==dataBytes)throw Error('Unindexed data in checkpoint');
  return tensors;
}
export function decodeTensor(bytes,dtype) {
  if(!['F32','F16','BF16'].includes(dtype))throw Error(`Unsupported compute dtype ${dtype}; quantized files require their own schema/kernels`);
  const width=WIDTH[dtype];if(bytes.byteLength%width)throw Error('Misaligned tensor data');
  const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),out=new Float32Array(bytes.byteLength/width);
  const scratch=new DataView(new ArrayBuffer(4));
  for(let i=0;i<out.length;i++) {
    if(dtype==='F32'){out[i]=v.getFloat32(i*4,true);continue;}
    const h=v.getUint16(i*2,true);
    if(dtype==='BF16'){scratch.setUint32(0,h*65536,true);out[i]=scratch.getFloat32(0,true);continue;}
    const sign=h&32768?-1:1,exp=(h>>>10)&31,m=h&1023;
    out[i]=exp===0?sign*2**-24*m:exp===31?(m?NaN:sign*Infinity):sign*2**(exp-15)*(1+m/1024);
  }return out;
}
// JSON.parse silently accepts duplicate keys; checkpoint descriptors must not.
export function parseHeader(text) {
  let i=0;
  const ws=()=>{while(/\s/.test(text[i]||'')&&i<text.length)i++;};
  const string=()=>{const start=i++;while(i<text.length){if(text[i]==='\\'){i+=2;continue;}if(text[i++]==='"')return JSON.parse(text.slice(start,i));}throw Error('Unterminated header string');};
  const value=()=>{ws();if(text[i]==='"'){string();return;}if(text[i]==='{'){i++;ws();const keys=new Set();if(text[i]==='}'){i++;return;}while(i<text.length){ws();if(text[i]!=='"')throw Error('Invalid JSON key');const key=string();if(keys.has(key))throw Error('Duplicate header key: '+key);keys.add(key);ws();if(text[i++]!==':')throw Error('Invalid JSON colon');value();ws();const c=text[i++];if(c==='}')return;if(c!==',')throw Error('Invalid JSON object');}throw Error('Unclosed header');}
    if(text[i]==='['){i++;ws();if(text[i]===']'){i++;return;}while(i<text.length){value();ws();const c=text[i++];if(c===']')return;if(c!==',')throw Error('Invalid JSON array');}throw Error('Unclosed header');}
    const start=i;while(i<text.length&&!/[\s,}\]]/.test(text[i]))i++;if(i===start)throw Error('Invalid JSON value');JSON.parse(text.slice(start,i));};
  if(text[0]!=='{')throw Error('Header must begin with object');value();ws();if(i!==text.length)throw Error('Trailing header data');return JSON.parse(text);
}
export class Checkpoint {
  static async open(source,options={}) {
    const first=await readRange(source,0,7,options),n=new DataView(first.bytes.buffer).getBigUint64(0,true);
    if(n<2n||n>BigInt(MAX_HEADER))throw Error('Invalid/oversized safetensors header');
    const headerBytes=Number(n),second=await readRange(source,8,7+headerBytes,options);
    if(first.total!==second.total)throw Error('Checkpoint changed during read');
    const obj=parseHeader(new TextDecoder('utf-8',{fatal:true}).decode(second.bytes));
    const cp=new Checkpoint();Object.assign(cp,{source,options,total:first.total,headerBytes,tensors:validateHeader(obj,first.total-8-headerBytes),downloaded:8+headerBytes});return cp;
  }
  async read(name,{elementOffset=0,count}={}) {
    const t=this.tensors.get(name);if(!t)throw Error('Tensor not found: '+name);count??=t.elements-elementOffset;
    if(!integer(elementOffset)||!integer(count)||elementOffset+count>t.elements)throw Error('Invalid tensor element range');
    if(count*4>64*1024*1024)throw Error('Decoded tensor exceeds 64 MiB safety budget; use row chunks');
    const out=new Float32Array(count);let offset=0;
    while(offset<count){checkAbort(this.options.signal);const n=Math.min(count-offset,Math.floor(MAX_READ/t.width));
      const start=8+this.headerBytes+t.start+(elementOffset+offset)*t.width;
      const result=await readRange(this.source,start,start+n*t.width-1,this.options);
      if(result.total!==this.total)throw Error('Checkpoint changed during tensor read');
      out.set(decodeTensor(result.bytes,t.dtype),offset);offset+=n;this.downloaded+=n*t.width;
      this.options.onProgress?.({downloaded:this.downloaded,tensor:name,elements:offset,totalElements:count});
    }return out;
  }
  summary() {const dtypes={};for(const t of this.tensors.values())dtypes[t.dtype]=(dtypes[t.dtype]||0)+1;return {fileBytes:this.total,headerBytes:this.headerBytes,tensorCount:this.tensors.size,weightBytes:this.total-8-this.headerBytes,dtypes,downloaded:this.downloaded};}
}
