// Optional, quota bounded IndexedDB cache; weights are never uploaded.
const request=r=>new Promise((resolve,reject)=>{r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
const completed=tx=>new Promise((resolve,reject)=>{tx.oncomplete=()=>resolve();tx.onerror=tx.onabort=()=>reject(tx.error||Error('Storage transaction aborted'));});
const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
export class ChunkCache {
  static async open({maxBytes=256*1024*1024,name='wangp-weights-v1'}={}) {
    if(!globalThis.indexedDB||!globalThis.crypto?.subtle)throw Error('Almacenamiento local no disponible');
    const r=indexedDB.open(name,1);r.onupgradeneeded=()=>{r.result.createObjectStore('chunks',{keyPath:'key'}).createIndex('created','created');r.result.createObjectStore('meta');};
    const db=await request(r);db.onversionchange=()=>db.close();const cache=new ChunkCache();Object.assign(cache,{db,maxBytes});return cache;
  }
  async get(key) {
    const tx=this.db.transaction('chunks','readonly'),done=completed(tx),row=await request(tx.objectStore('chunks').get(key));await done;
    if(!row)return null;const bytes=new Uint8Array(row.bytes);
    if(await digest(bytes)!==row.sha256){await this.delete(key);throw Error('Cache checksum mismatch; chunk removed');}
    return {bytes,total:row.total};
  }
  async delete(key) {
    const tx=this.db.transaction(['chunks','meta'],'readwrite'),done=completed(tx),store=tx.objectStore('chunks');
    const row=await request(store.get(key)),used=await request(tx.objectStore('meta').get('used'))||0;
    store.delete(key);tx.objectStore('meta').put(Math.max(0,used-(row?.size||0)),'used');await done;
  }
  async put(key,{bytes,total}) {
    if(bytes.byteLength>this.maxBytes)throw Error('Chunk exceeds cache budget');
    const sha256=await digest(bytes),copy=bytes.slice().buffer;
    const tx=this.db.transaction(['chunks','meta'],'readwrite'),done=completed(tx),store=tx.objectStore('chunks'),meta=tx.objectStore('meta');
    let used=await request(meta.get('used'))||0;const old=await request(store.get(key));if(old){used-=old.size;store.delete(key);}
    if(used+bytes.byteLength>this.maxBytes){await new Promise((resolve,reject)=>{const c=store.index('created').openCursor();c.onerror=()=>reject(c.error);c.onsuccess=()=>{const cursor=c.result;if(!cursor||used+bytes.byteLength<=this.maxBytes){resolve();return;}used-=cursor.value.size;cursor.delete();cursor.continue();};});}
    store.put({key,bytes:copy,total,size:bytes.byteLength,sha256,created:Date.now()});meta.put(used+bytes.byteLength,'used');await done;
  }
  async usage(){const tx=this.db.transaction('meta','readonly'),done=completed(tx),n=await request(tx.objectStore('meta').get('used'));await done;return n||0;}
  async clear(){const tx=this.db.transaction(['chunks','meta'],'readwrite'),done=completed(tx);tx.objectStore('chunks').clear();tx.objectStore('meta').clear();await done;}
  close(){this.db.close();}
}
