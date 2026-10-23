// Authenticate each official row chunk before it reaches an operator.
const sha256=async values=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',values)),b=>b.toString(16).padStart(2,'0')).join('');
export function verifiedProvider(provider,manifest) {
  async function verify(name,values,expected){if(!expected||await sha256(values)!==expected)throw Error('Official tensor checksum mismatch: '+name);return values;}
  return {
    get:async name=>{const spec=manifest.tensors[name];if(!spec)throw Error('Unknown official tensor '+name);return verify(name,await provider.get(name),spec.sha256);},
    read:async(name,{elementOffset=0,count}={})=>{const spec=manifest.tensors[name],chunk=spec?.chunks.find(c=>c.elementOffset===elementOffset&&c.count===count);if(!chunk)throw Error('Unknown official tensor chunk '+name);return verify(name,await provider.read(name,{elementOffset,count}),chunk.sha256);}
  };
}
