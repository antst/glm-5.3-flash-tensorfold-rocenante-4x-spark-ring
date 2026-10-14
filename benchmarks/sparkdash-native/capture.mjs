import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

export const sha256 = value => crypto.createHash("sha256").update(value).digest("hex");
export const ROOT = path.dirname(new URL(import.meta.url).pathname);
export function journal(action, detail = {}) {
  fs.appendFileSync(path.join(ROOT, "changes.jsonl"), JSON.stringify({utc:new Date().toISOString(),action,...detail})+"\n");
}
export function saveFile(file, value) {
  const body = Buffer.isBuffer(value) ? value : Buffer.from(typeof value==="string"?value:JSON.stringify(value,null,2)+"\n");
  if (!path.resolve(file).startsWith(ROOT+"/")) throw Error("Write outside benchmark subtree");
  const before = fs.existsSync(file) ? sha256(fs.readFileSync(file)) : null;
  fs.mkdirSync(path.dirname(file),{recursive:true,mode:0o700});
  fs.writeFileSync(file,body,{mode:0o600});
  journal("write-file",{path:path.relative(ROOT,file),before_sha256:before,after_sha256:sha256(body),bytes:body.length});
}
export function parseSse(body) {
  const events=[], invalid=[];
  for(const block of body.replace(/\r\n/g,"\n").split("\n\n")) {
    for(const line of block.split("\n")) {
      if(!line.trim().startsWith("data:")) continue;
      const data=line.trim().slice(5).trim();
      if(!data||data==="[DONE]") continue;
      try{events.push(JSON.parse(data));}catch{invalid.push(data.slice(0,120));}
    }
  }
  let usage=null, tensorfold=null, reasoningPieces=0;
  for(const event of events) {
    if(event.usage) usage=event.usage;
    if(event.tensorfold) tensorfold=event.tensorfold;
    for(const choice of event.choices||[]) {
      const d=choice.delta||{};
      if(d.reasoning_content||d.reasoning||d.thinking) reasoningPieces++;
    }
  }
  return {events,invalid,usage,tensorfold,reasoningPieces,done:/(?:^|\n)data:\s*\[DONE\](?:\s|$)/m.test(body),
    errors:events.filter(v=>v.error).map(v=>v.error)};
}
let active=null;
export function setRecorder(value){active=value;}
export class Recorder {
  constructor(){this.records=[];this.context={};this.flushed=0;}
  async capture(url,options,next) {
    const body=String(options.body??"");
    if(Buffer.byteLength(body)>4*2**20) throw Error("Benchmark request exceeds capture bound");
    const record={index:this.records.length+1,url:String(url),method:options.method||"GET",
      context:{...this.context,progress:this.context.progress?.()||null},
      startedPerformanceMs:performance.now(),body,bodySha256:sha256(body),chunks:[],bytes:0,complete:false};
    delete record.context.progressFn;
    this.records.push(record);
    try {
      const response=await next(url,options);
      record.status=response.status;
      record.headers=Object.fromEntries([...response.headers].filter(([name])=>/^(content-type|server|date|x-request-id)$/.test(name)));
      const stream=response.body;
      if(stream) {
        const getReader=stream.getReader.bind(stream);
        stream.getReader=(...args)=>{
          const reader=getReader(...args);
          const read=reader.read.bind(reader);
          reader.read=async()=>{
            try {
              const chunk=await read();
              if(chunk.done){record.complete=true;record.endedPerformanceMs=performance.now();}
              else {
                record.bytes+=chunk.value.byteLength;
                if(record.bytes>8*2**20) throw Error("Benchmark response exceeds capture bound");
                record.chunks.push({atPerformanceMs:performance.now(),value:Buffer.from(chunk.value)});
              }
              return chunk;
            } catch(error){record.readError=String(error.message||error);throw error;}
          };
          return reader;
        };
      } else {record.complete=true;record.endedPerformanceMs=performance.now();}
      return response;
    } catch(error){record.fetchError=String(error.message||error);record.endedPerformanceMs=performance.now();throw error;}
  }
  flush(directory) {
    const manifests=[];
    for(const record of this.records.slice(this.flushed)) {
      const wire=Buffer.concat(record.chunks.map(v=>v.value));
      const parsed=parseSse(wire.toString());
      const stem=String(record.index).padStart(5,"0");
      const json={...record,chunks:record.chunks.map(v=>({atPerformanceMs:v.atPerformanceMs,bytes:v.value.length})),
        responseSha256:sha256(wire),responseBytes:wire.length,parsed};
      saveFile(path.join(directory,stem+".response.sse"),wire);
      saveFile(path.join(directory,stem+".request-response.json"),json);
      manifests.push({index:record.index,request_sha256:record.bodySha256,response_sha256:sha256(wire),
        json:path.basename(stem+".request-response.json"),sse:path.basename(stem+".response.sse")});
    }
    this.flushed=this.records.length;
    return manifests;
  }
}
export async function capturedFetch(url,options,next) {
  return active ? active.capture(url,options,next) : next(url,options);
}
