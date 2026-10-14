#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {ROOT,Recorder,setRecorder,saveFile,journal,parseSse,sha256} from "./capture.mjs";
import {parseArgs,requestPlan,sourceIdentity,loadCollectors,validateHealth,validateJob,auditRequests,PIN,UPSTREAM} from "./protocol.mjs";

const HELP=[
"Use Node26+. No production inference: port8888 is rejected.",
"  node run-sparkdash.mjs --url http://PRIVATE-IP:18898 --suite decode",
"    --decode-types prose,code --concurrency 1,2,4,8 --max-tokens 400",
"    --repeats 1 --output ./runs/NEW --metadata PRIVATE.json",
"  node run-sparkdash.mjs --url http://PRIVATE-IP:18898 --suite prefill",
"    --prefill-sizes 8192,16384,32768,65536,131072,262144",
"    --output ./runs/NEW",
"  --plan: offline source/settings/prompt identity; no HTTP and no writes.",
"  --replay-prefill: separate supplemental exact-body replays after cold sweep;",
"    NOT part of upstream PrefillBench. --suite all runs decode then prefill.",
"  Source UI default is structured and4K/8K/16K/32K prefill. This CLI explicitly",
"  selects prose/code and Mia-table lengths. Historical UI flags/repeats unknown.",
"  Each decode repeat is a complete upstream job including its32-token warmup.",
"  Inference runs require external owner authorization/idle matched workers."
].join("\n");

async function health(args,control){
  const response=await fetch(args.url+"/health",{signal:AbortSignal.any([control.signal,AbortSignal.timeout(10000)])});
  if(!response.ok)throw Error("Health HTTP"+response.status);
  const raw=await response.text();
  if(Buffer.byteLength(raw)>256*1024)throw Error("Health response exceeds bound");
  const value=JSON.parse(raw);
  validateHealth(value,args);
  return value;
}
async function models(args,control){
  const r=await fetch(args.url+"/v1/models",{signal:AbortSignal.any([control.signal,AbortSignal.timeout(10000)])});
  if(!r.ok)throw Error("Models HTTP"+r.status);
  const value=await r.json();
  if(!value.data?.some(v=>v.id===args.model))throw Error("Requested model not advertised");
  return value;
}
function recordsParsed(recorder){
  return recorder.records.map(r=>({...r,parsed:parseSse(Buffer.concat(r.chunks.map(c=>c.value)).toString())}));
}
async function runJob(manager,kind,opts,recorder,args,output,control,group){
  if(control.signal.aborted)throw Error("Cancelled before next job");
  const before=await health(args,control);
  const first=recorder.records.length;
  let id;
  const sparkId="private-matched-glm";
  const timeout=kind==="decode"?(opts.concurrencies.length+1)*360000+60000:
    5*90000+args.sizes.reduce((a,n)=>a+(Math.min(2700000,Math.max(90000,60000+n*8))*(n<=32768?3:n<=131072?2:1)),0)+60000;
  let resolve;
  const finished=new Promise(r=>resolve=r);
  manager.setEventSink((event,job)=>resolve(manager.getJob(job.benchId)||job));
  const cancel=()=>{if(id)manager.cancel(sparkId,id);};
  control.signal.addEventListener("abort",cancel,{once:true});
  const timer=setTimeout(cancel,timeout);
  const heartbeat=setInterval(()=>{
    const live=id?manager.getJob(id):null;
    console.log(JSON.stringify({phase:"benchmark-running",group,status:live?.status,progress:live?.progress}));
  },15000);
  recorder.context={group,kind,workload:opts.promptType||null,
    progress:()=>manager.getActive(sparkId)?.progress||null};
  try {
    const job=manager.start({...opts,sparkId,lanIp:args.host,host:args.host,port:args.port,tls:args.tls,modelId:args.model});
    id=job.benchId;
    const result=await finished;
    const traces=recorder.flush(path.join(output,"requests"));
    saveFile(path.join(output,group+".job.json"),result);
    saveFile(path.join(output,group+".traces.json"),traces);
    validateJob(result,kind,args);
    const after=await health(args,control);
    const expected=kind==="decode"?1+args.concurrency.reduce((a,b)=>a+b,0):
      5+args.sizes.reduce((a,n)=>a+(n<=32768?3:n<=131072?2:1),0);
    const audit=auditRequests(before,after,recordsParsed(recorder).slice(first),expected);
    saveFile(path.join(output,group+".audit.json"),audit);
    saveFile(path.join(output,group+".health-before.json"),before);
    saveFile(path.join(output,group+".health-after.json"),after);
    console.log(JSON.stringify({phase:"benchmark-job-complete",group,request_delta:audit.requests_delta,
      rows:result.results.map(row=>kind==="decode"?{concurrency:row.concurrency,aggregateDecodeTps:row.aggregateDecodeTps,meanTtftMs:row.meanTtftMs}:
        {targetTokens:row.targetTokens,promptTokens:row.promptTokens,cachedTokens:row.cachedTokens,prefillTps:row.prefillTps,ttftMs:row.ttftMs,method:row.method})}));
    return {group,kind,job:result,audit};
  }finally{
    clearTimeout(timer);clearInterval(heartbeat);
    control.signal.removeEventListener("abort",cancel);
  }
}
function summary(jobs,args){
  const rows=[];
  for(const type of args.types)for(const concurrency of args.concurrency){
    const waves=jobs.filter(j=>j.kind==="decode"&&j.job.config.promptType===type).map(j=>j.job.results.find(r=>r.concurrency===concurrency)).filter(Boolean);
    if(!waves.length)continue;
    const rates=waves.map(v=>v.aggregateDecodeTps).sort((a,b)=>a-b),mid=rates.length>>1;
    rows.push({type,concurrency,job_repeats:waves.length,aggregate_decode_tps_median:rates.length%2?rates[mid]:(rates[mid-1]+rates[mid])/2,
      aggregate_decode_tps_range:[rates[0],rates.at(-1)],actual_collector_waves:waves});
  }
  return {decode:rows,prefill:jobs.filter(j=>j.kind==="prefill").flatMap(j=>j.job.results)};
}
async function supplementalReplay(collector,recorder,args,output,control){
  const results=[];
  for(const size of args.sizes){
    const previous=recorder.records.filter(v=>v.context.kind==="prefill"&&v.context.progress?.currentContext===size).at(-1);
    if(!previous)throw Error("No cold sample body available for supplemental replay");
    recorder.context={group:"supplemental-replay-"+size,kind:"supplemental-replay",targetTokens:size,originalRequestIndex:previous.index};
    const body=JSON.parse(previous.body);
    const response=await collector.streaming.runStreamingRequest(args.url+"/v1/chat/completions",body,
      AbortSignal.any([control.signal,AbortSignal.timeout(collector.prefill.timeoutMsForSize(size))]),
      {thinking:false,retryOnThinking400:true,collectContent:true});
    if(response.error)throw Error("Supplemental replay failed: "+response.error);
    results.push({targetTokens:size,sourceRequestIndex:previous.index,scope:"Exact last cold-sample body replay; not SparkDash PrefillBench",
      result:response,computed_prompt_tokens:Math.max(0,response.prefillTokens-response.cachedPromptTokens)});
    recorder.flush(path.join(output,"requests"));
  }
  saveFile(path.join(output,"supplemental-replays.json"),results);
  return results;
}
export async function main(argv=process.argv.slice(2)){
  const args=parseArgs([...argv]);
  if(args.help){console.log(HELP);return 0;}
  const identity=sourceIdentity(),plan=requestPlan(args);
  if(args.plan){
    const prompts=await import(pathToFileURL(path.join(UPSTREAM,"src/shared/llmPrompts.js")).href);
    console.log(JSON.stringify({offline_plan:true,http_requests:0,file_writes:0,settings:args,plan,source:identity,
      exact_decode_prompts:args.types.flatMap(type=>args.concurrency.map(concurrency=>({type,concurrency,prompts:prompts.pickDecodeBenchPrompts(concurrency,type)})))},null,2));
    return 0;
  }
  fs.mkdirSync(path.dirname(args.output),{recursive:true,mode:0o700});
  fs.mkdirSync(args.output,{recursive:false,mode:0o700});
  journal("benchmark-run-created",{path:path.relative(ROOT,args.output),endpoint:args.url,planned_inference_requests:plan.expected_successful_inference_requests});
  const started=new Date().toISOString(),control=new AbortController();
  const interrupted=()=>control.abort();
  process.once("SIGINT",interrupted);process.once("SIGTERM",interrupted);
  const recorder=new Recorder(),jobs=[];
  setRecorder(recorder);
  let collector,before,after;
  try {
    saveFile(path.join(args.output,"settings.json"),{args,plan,source:identity,started});
    const ownSources=["run-sparkdash.mjs","capture.mjs","protocol.mjs","recorded-undici.mjs"];
    saveFile(path.join(args.output,"runner-shas.json"),Object.fromEntries(ownSources.map(name=>[name,sha256(fs.readFileSync(path.join(ROOT,name)))])));
    if(args.metadata){
      const raw=fs.readFileSync(args.metadata);
      if(raw.length>2**20)throw Error("Provided metadata exceeds bound");
      const data=JSON.parse(raw);
      if(!data||Array.isArray(data)||typeof data!=="object")throw Error("Provided metadata must be a JSON object");
      saveFile(path.join(args.output,"provided-metadata.json"),data);
      saveFile(path.join(args.output,"provided-metadata-identity.json"),{path:args.metadata,sha256:sha256(raw),
        scope:"Supplied external runtime provenance; this HTTP-only CLI does not independently verify host/container/weight identity."});
    }
    before=await health(args,control);
    saveFile(path.join(args.output,"health-before.json"),before);
    saveFile(path.join(args.output,"models.json"),await models(args,control));
    collector=await loadCollectors(path.join(args.output,"manager-state"));
    if(args.suite!=="prefill"){
      for(const type of args.types)for(let repeat=1;repeat<=args.repeats;repeat++){
        const manager=new collector.decode.DecodeBenchManager(
          path.join(args.output,"manager-state",type+"-"+repeat+"-history.json"),
          path.join(args.output,"manager-state",type+"-"+repeat+"-active.json"));
        jobs.push(await runJob(manager,"decode",{concurrencies:args.concurrency,maxTokens:args.maxTokens,promptType:type,debug:args.debug},
          recorder,args,args.output,control,"decode-"+type+"-"+repeat));
      }
    }
    if(args.suite!=="decode"){
      const manager=new collector.prefill.PrefillBenchManager(
        path.join(args.output,"manager-state","prefill-history.json"),path.join(args.output,"manager-state","prefill-active.json"));
      jobs.push(await runJob(manager,"prefill",{contextSizes:args.sizes,contextLength:args.context},
        recorder,args,args.output,control,"prefill"));
    }
    const supplemental=args.replay?await supplementalReplay(collector,recorder,args,args.output,control):[];
    after=await health(args,control);
    const audit=auditRequests(before,after,recordsParsed(recorder),plan.expected_successful_inference_requests);
    saveFile(path.join(args.output,"health-after.json"),after);
    const report={schema:"sparkdash-actual-collector-private-benchmark/v1",passed:true,started,finished:new Date().toISOString(),
      source_pin:PIN,source_pin_sha256:identity.source_pin_sha256,settings:args,plan,audit,jobs,
      summary:summary(jobs,args),supplemental,scope:[
        "Actual upstream collector code and metrics; separate request-byte capture adds small client work equally to both runs.",
        "Default wrapper repeat count1 is one UI-equivalent job per type; historical Mia flags/repeats are unknown.",
        "Decode repeats reuse exact prompts/prefixes; prefill samples all use fresh UUID salts. Target prefill labels are estimates; usage reports actual token count.",
        "Supplemental replay, when enabled, is separate from upstream PrefillBench and does not alter its cold-sample rows.",
        "No host/model/network configuration is changed by this client. Use externally reserved, source-bound private workers.",
        "Authoritative usage, DONE/EOF and zero reasoning are additional receipt gates; upstream may fall back to estimated counts or accept partial rows."]};
    saveFile(path.join(args.output,"result.json"),report);
    console.log(JSON.stringify({passed:true,output:args.output,request_delta:audit.requests_delta,summary:report.summary},null,2));
    return 0;
  }catch(error){
    recorder.flush(path.join(args.output,"requests"));
    saveFile(path.join(args.output,"failure.json"),{passed:false,started,finished:new Date().toISOString(),error:String(error.message||error),
      complete_jobs:jobs.length,http_attempts:recorder.records.length,before,after});
    throw error;
  }finally{
    setRecorder(null);
    if(collector)await collector.streaming.closeLlmStreamAgent();
    process.removeListener("SIGINT",interrupted);process.removeListener("SIGTERM",interrupted);
  }
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){
  main().then(code=>{process.exitCode=code;}).catch(error=>{console.error(JSON.stringify({passed:false,error:String(error.message||error)}));process.exitCode=1;});
}
