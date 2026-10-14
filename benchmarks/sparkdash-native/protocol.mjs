import fs from "node:fs";
import path from "node:path";
import {registerHooks} from "node:module";
import {pathToFileURL} from "node:url";
import {ROOT,sha256} from "./capture.mjs";

export const PIN="fab8dadff951d1c8cc1450a657805d8e3ea5c9f6";
export const UPSTREAM=path.join(ROOT,"upstream",PIN);
export const DECODE_CONCURRENCY=[1,2,4,8];
export const MIA_PREFILL_LENGTHS=[8192,16384,32768,65536,131072,262144];
export function repeatsForSize(n){return n<=32768?3:n<=131072?2:1;}
export function parseArgs(argv){
  const args={suite:"decode",types:["prose","code"],concurrency:[...DECODE_CONCURRENCY],maxTokens:400,repeats:1,
    sizes:[...MIA_PREFILL_LENGTHS],lanes:8,context:1048576,model:"GLM-5.3-Flash-EXL3",plan:false,debug:false};
  const value=()=>{if(!argv.length)throw Error("Missing option value");return argv.shift();};
  while(argv.length){
    const key=argv.shift();
    if(key==="--plan"){args.plan=true;continue;}
    if(key==="--debug"){args.debug=true;continue;}
    if(key==="--help"){args.help=true;continue;}
    if(key==="--replay-prefill"){args.replay=true;continue;}
    if(key==="--url")args.url=value();
    else if(key==="--output")args.output=path.resolve(value());
    else if(key==="--suite")args.suite=value();
    else if(key==="--decode-types")args.types=value().split(",");
    else if(key==="--concurrency")args.concurrency=value().split(",").map(Number);
    else if(key==="--prefill-sizes")args.sizes=value().split(",").map(Number);
    else if(key==="--max-tokens")args.maxTokens=Number(value());
    else if(key==="--repeats")args.repeats=Number(value());
    else if(key==="--serving-lanes")args.lanes=Number(value());
    else if(key==="--context-window")args.context=Number(value());
    else if(key==="--model")args.model=value();
    else if(key==="--metadata")args.metadata=path.resolve(value());
    else throw Error("Unknown option: "+key);
  }
  if(args.help)return args;
  if(!["decode","prefill","all"].includes(args.suite))throw Error("Invalid suite");
  if(!Number.isInteger(args.repeats)||args.repeats<1||args.repeats>20)throw Error("Repeats must be1..20");
  if(!Number.isInteger(args.maxTokens)||args.maxTokens<64||args.maxTokens>2048)throw Error("max_tokens must be64..2048");
  if(args.lanes!==8)throw Error("This matched study requires configured P8");
  if(!Number.isInteger(args.context)||args.context<=Math.max(...args.sizes)+512)throw Error("Insufficient configured context");
  if(!args.types.length||new Set(args.types).size!==args.types.length||args.types.some(v=>!["prose","code","structured","json"].includes(v)))throw Error("Invalid decode types");
  if(!args.concurrency.length||new Set(args.concurrency).size!==args.concurrency.length||args.concurrency.some(v=>!DECODE_CONCURRENCY.includes(v)))throw Error("Use C1/C2/C4/C8");
  args.concurrency.sort((a,b)=>a-b);
  if(!args.sizes.length||new Set(args.sizes).size!==args.sizes.length||args.sizes.some(v=>!Number.isInteger(v)||v<256||v>300000))throw Error("Invalid prefill size");
  args.sizes.sort((a,b)=>a-b);
  if(args.replay&&args.suite==="decode")throw Error("Supplemental replay requires prefill suite");
  if(!args.url)throw Error("An explicit private endpoint is required");
  const url=new URL(args.url), octets=url.hostname.split(".").map(Number);
  const ipv4=octets.length===4&&octets.every(v=>Number.isInteger(v)&&v>=0&&v<=255);
  const privateHost=url.hostname==="localhost"||(ipv4&&(octets[0]===127||octets[0]===10||
    (octets[0]===192&&octets[1]===168)||(octets[0]===172&&octets[1]>=16&&octets[1]<=31)));
  if(!["http:","https:"].includes(url.protocol)||!privateHost||!url.port||Number(url.port)<1024||
     Number(url.port)===8888||url.username||url.password||url.search||url.hash||!["/","/v1",""].includes(url.pathname))
    throw Error("Use an explicit private HTTP origin/port; production8888 is prohibited");
  args.url=url.origin;
  args.host=url.hostname;args.port=Number(url.port);args.tls=url.protocol==="https:";
  if(!args.output&&!args.plan)throw Error("A new --output directory is required");
  if(args.output&&(!args.output.startsWith(path.join(ROOT,"runs")+path.sep)||fs.existsSync(args.output)))
    throw Error("Output must be NEW beneath benchmark/runs/");
  return args;
}
export function requestPlan(args){
  const decode=args.suite!=="prefill"?args.types.length*args.repeats*(1+args.concurrency.reduce((a,b)=>a+b,0)):0;
  const prefill=args.suite!=="decode"?5+args.sizes.reduce((a,n)=>a+repeatsForSize(n),0):0;
  const replay=args.replay?args.sizes.length:0;
  return {decode_requests:decode,prefill_requests:prefill,supplemental_replay_requests:replay,
    expected_successful_inference_requests:decode+prefill+replay,
    decode_warmup_max_tokens:32,decode_measured_max_tokens:args.maxTokens,
    prefill_warmup_targets:[512,4096],calibration_samples:3,calibration_target_tokens:16,
    prefill_completion_cap:8,prefill_samples_by_target:args.sizes.map(target=>({target,repeats:repeatsForSize(target)})),
    source_default_decode_type:"structured",selected_types:args.types,
    source_default_prefill_sizes:[4096,8192,16384,32768],selected_sizes:args.sizes,
    repeat_scope:"Each requested decode repeat invokes a new actual SparkDash job including its native warmup.",
    replay_scope:args.replay?"Supplemental replay after the cold sweep; not a SparkDash PrefillBench feature.":"No exact-prompt replay: upstream PrefillBench salts every sample.",
    historical_mia_repeat_count_and_flags_known:false};
}
export function sourceIdentity(){
  const pin=JSON.parse(fs.readFileSync(path.join(ROOT,"source-pin.json")));
  for(const [name,expected]of Object.entries(pin.files)){
    if(sha256(fs.readFileSync(path.join(UPSTREAM,name)))!==expected.sha256)throw Error("Upstream source hash changed: "+name);
  }
  const localLock=JSON.parse(fs.readFileSync(path.join(ROOT,"package-lock.json")));
  const dependency=localLock.packages["node_modules/undici"];
  if(dependency.version!==pin.undici_lock.version||dependency.integrity!==pin.undici_lock.integrity)
    throw Error("Undici dependency differs from upstream lock");
  return {pin,local_package_lock_sha256:sha256(fs.readFileSync(path.join(ROOT,"package-lock.json"))),
    source_pin_sha256:sha256(fs.readFileSync(path.join(ROOT,"source-pin.json"))),node_version:process.version};
}
let loaded=null;
export async function loadCollectors(stateDirectory){
  if(loaded)return loaded;
  process.env.BENCH_HISTORY_PATH=path.join(stateDirectory,"decode-history.json");
  process.env.BENCH_ACTIVE_PATH=path.join(stateDirectory,"decode-active.json");
  process.env.PREFILL_BENCH_HISTORY_PATH=path.join(stateDirectory,"prefill-history.json");
  process.env.PREFILL_BENCH_ACTIVE_PATH=path.join(stateDirectory,"prefill-active.json");
  const parent=pathToFileURL(path.join(UPSTREAM,"server/collectors/LlmStreaming.js")).href;
  const shim=pathToFileURL(path.join(ROOT,"recorded-undici.mjs")).href;
  registerHooks({resolve(specifier,context,next){
    if(specifier==="undici"&&context.parentURL===parent)return {url:shim,shortCircuit:true};
    return next(specifier,context);
  }});
  loaded={
    decode:await import(pathToFileURL(path.join(UPSTREAM,"server/collectors/DecodeBench.js")).href),
    prefill:await import(pathToFileURL(path.join(UPSTREAM,"server/collectors/PrefillBench.js")).href),
    streaming:await import(pathToFileURL(path.join(UPSTREAM,"server/collectors/LlmStreaming.js")).href),
    prompts:await import(pathToFileURL(path.join(UPSTREAM,"src/shared/llmPrompts.js")).href),
  };
  return loaded;
}
export function validateHealth(health,args){
  if(health?.ok!==true||health?.streams?.max!==args.lanes||health.context_length!==args.context||
     health.busy!==false||health.requests_running!==0||!Number.isInteger(health.requests_total)||health.requests_total<0)
    throw Error("Health does not prove idle selected P8/context service");
  if(health.requests_waiting!=null&&health.requests_waiting!==0)throw Error("Queued requests at benchmark boundary");
  for(const key of ["decoding","prefilling","filling","paused"])
    if(health.streams[key]!==0)throw Error("Active engine lanes at benchmark boundary");
}
export function validateJob(job,kind,args){
  if(job.status!=="completed"||job.error)throw Error("SparkDash job did not complete successfully");
  if(kind==="decode"){
    if(job.results.length!==args.concurrency.length)throw Error("Decode wave count mismatch");
    for(const row of job.results){
      if(row.error||row.streamsFailed!==0||row.streamsOk!==row.concurrency||row.totalCompletionTokens!==row.concurrency*args.maxTokens||
        row.totalDecodeTokens!==row.concurrency*(args.maxTokens-1)||!(row.aggregateDecodeTps>0))
        throw Error("Decode wave lacks exact successful fixed-cap output");
      if(row.streams.some(v=>v.reasoningChunks!==0||v.completionTokens!==args.maxTokens))throw Error("Decode stream mismatch");
    }
  }else{
    if(job.results.length!==args.sizes.length)throw Error("Prefill row count mismatch");
    for(const row of job.results){
      if(row.error||row.notice||row.samples!==row.samplesRequested||row.samples!==repeatsForSize(row.targetTokens)||
        !(row.prefillTps>0)||!(row.promptTokens>0))throw Error("Incomplete prefill row");
    }
  }
}
export function auditRequests(before,after,records,expected){
  const delta=after.requests_total-before.requests_total;
  if(records.length!==expected||delta!==expected)throw Error("Request counter/attempt delta differs: foreign traffic, retry, or reset");
  for(const r of records){
    const parsed=r.parsed;
    if(r.status!==200||!r.complete||r.fetchError||r.readError||!parsed.done||parsed.invalid.length||parsed.errors.length||
      !Number.isInteger(parsed.usage?.prompt_tokens)||!(parsed.usage.prompt_tokens>0)||
      !Number.isInteger(parsed.usage?.completion_tokens)||!(parsed.usage.completion_tokens>0)||
      parsed.reasoningPieces||(parsed.usage?.completion_tokens_details?.reasoning_tokens||0))
      throw Error("Incomplete/error/reasoning stream or missing authoritative usage");
  }
  return {requests_before:before.requests_total,requests_after:after.requests_total,requests_delta:delta,
    expected_inference_requests:expected,actual_http_attempts:records.length,exclusive_counter_check_passed:true,
    scope:"Matching counters cannot exclude unrelated GPU/network activity; runtime bindings are supplied separately."};
}
