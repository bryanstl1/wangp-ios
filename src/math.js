// Float32 outputs with scalar double accumulation: an explicit CPU oracle, not a fast video runtime.
export function finite(x) {if(!x)throw Error('Nonfinite or missing tensor');for(const v of x)if(!Number.isFinite(v))throw Error('Nonfinite or missing tensor');}
export function linearCPU(x,w,b,rows,k,n) {
  if(x.length!==rows*k||w.length!==n*k||b.length!==n)throw Error('Linear shape mismatch');
  const y=new Float32Array(rows*n);
  for(let r=0;r<rows;r++)for(let o=0;o<n;o++){let s=0;for(let j=0;j<k;j++)s+=x[r*k+j]*w[o*k+j];y[r*n+o]=s+b[o];}return y;
}
export function normalize(x,dim,eps=1e-6,weight=null,bias=null,rms=false) {
  if(!Number.isInteger(dim)||dim<1||x.length%dim||eps<0||weight&&weight.length!==dim||bias&&bias.length!==dim)throw Error('Norm shape mismatch');
  const y=new Float32Array(x.length);
  for(let r=0;r<x.length/dim;r++) {let mean=0,s=0;
    if(!rms){for(let j=0;j<dim;j++)mean+=x[r*dim+j];mean/=dim;}
    for(let j=0;j<dim;j++)s+=(x[r*dim+j]-mean)**2;const inv=1/Math.sqrt(s/dim+eps);
    for(let j=0;j<dim;j++)y[r*dim+j]=(x[r*dim+j]-mean)*inv*(weight?.[j]??1)+(bias?.[j]??0);
  }return y;
}
export function gelu(x) {return Float32Array.from(x,v=>.5*v*(1+Math.tanh(Math.sqrt(2/Math.PI)*(v+.044715*v**3))));}
export function attention(q,k,v,t,s,heads,dim) {
  if(q.length!==t*heads*dim||k.length!==s*heads*dim||v.length!==k.length||s<1)throw Error('Attention shape mismatch');finite(q);finite(k);finite(v);
  const out=new Float32Array(q.length),scores=new Float64Array(s);
  for(let i=0;i<t;i++)for(let h=0;h<heads;h++) {
    let mx=-Infinity;
    for(let j=0;j<s;j++){let a=0;for(let d=0;d<dim;d++)a+=q[(i*heads+h)*dim+d]*k[(j*heads+h)*dim+d];scores[j]=a/Math.sqrt(dim);mx=Math.max(mx,scores[j]);}
    let total=0;for(let j=0;j<s;j++){scores[j]=Math.exp(scores[j]-mx);total+=scores[j];}
    for(let d=0;d<dim;d++){let a=0;for(let j=0;j<s;j++)a+=scores[j]/total*v[(j*heads+h)*dim+d];out[(i*heads+h)*dim+d]=a;}
  }return out;
}
export function rope(x,grid,heads,dim) {
  const [f,h,w]=grid;if(dim%2||grid.length!==3||grid.some(v=>!Number.isInteger(v)||v<1)||x.length!==f*h*w*heads*dim)throw Error('RoPE shape mismatch');
  const pairs=dim/2,side=Math.floor(pairs/3),sizes=[pairs-2*side,side,side],out=new Float32Array(x);
  for(let t=0;t<f*h*w;t++) {const positions=[Math.floor(t/(h*w)),Math.floor(t/w)%h,t%w];
    for(let head=0;head<heads;head++){let pair=0;for(let axis=0;axis<3;axis++)for(let j=0;j<sizes[axis];j++,pair++) {
      const a=positions[axis]*10000**(-j/sizes[axis]),c=Math.cos(a),s=Math.sin(a),i=(t*heads+head)*dim+2*pair;
      out[i]=x[i]*c-x[i+1]*s;out[i+1]=x[i]*s+x[i+1]*c;
    }}
  }return out;
}
export function compare(got,expected,atol=1e-4,rtol=1e-4) {
  let maxAbsolute=0,maxRelative=0,failures=0;
  if(got.length!==expected.length)return {passed:false,failures:Infinity,maxAbsolute:Infinity,maxRelative:Infinity};
  for(let i=0;i<got.length;i++){const a=Math.abs(got[i]-expected[i]),r=a/Math.max(1e-8,Math.abs(expected[i]));maxAbsolute=Math.max(maxAbsolute,a);maxRelative=Math.max(maxRelative,r);if(!Number.isFinite(a)||a>atol+rtol*Math.abs(expected[i]))failures++;}
  return {passed:failures===0,failures,maxAbsolute,maxRelative,atol,rtol};
}
export function quantize(x,bits=8,groupSize=64) {
  finite(x);if(![4,8].includes(bits)||!Number.isInteger(groupSize)||groupSize<1)throw Error('Unsupported quantization');
  const scales=new Float32Array(Math.ceil(x.length/groupSize)),data=new Uint8Array(Math.ceil(x.length*bits/8)),limit=bits===4?7:127;
  for(let g=0;g<scales.length;g++){let mx=0;for(let i=g*groupSize;i<Math.min(x.length,(g+1)*groupSize);i++)mx=Math.max(mx,Math.abs(x[i]));scales[g]=mx===0?1:mx/limit;
    for(let i=g*groupSize;i<Math.min(x.length,(g+1)*groupSize);i++){const q=Math.max(-limit,Math.min(limit,Math.round(x[i]/scales[g])));if(bits===8)data[i]=q&255;else data[i>>1]|=(q&15)<<((i%2)*4);}}
  return {bits,groupSize,length:x.length,scales,data};
}
export function dequantize({bits,groupSize,length,scales,data}) {const y=new Float32Array(length);for(let i=0;i<length;i++){let q=bits===8?data[i]:(data[i>>1]>>((i%2)*4))&15;const bound=bits===8?128:8;if(q>=bound)q-=bound*2;y[i]=q*scales[Math.floor(i/groupSize)];}return y;}
