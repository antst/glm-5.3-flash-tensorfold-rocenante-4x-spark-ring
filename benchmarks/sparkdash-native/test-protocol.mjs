import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {ROOT,Recorder,setRecorder,parseSse,sha256,journal} from "./capture.mjs";
import {parseArgs,requestPlan,sourceIdentity,loadCollectors,validateHealth,validateJob,auditRequests} from "./protocol.mjs";
import {setTransportForCpuTests} from "./recorded-undici.mjs";
import {main} from "./run-sparkdash.mjs";
import {compareResults,compareDecodeOutputs} from "./compare-sparkdash.mjs";

const state=fs.mkdtempSync(path.join(ROOT,"cpu-checks-"));
journal("cpu-checks-created",{path:path.relative(ROOT,state),network_disabled:true});
const originalFetch=globalThis.fetch;
globalThis.fetch=async()=>{throw Error("CPU checks prohibit all real HTTP");};
let fixtureRequests=[],fixtureActive=0,fixturePeak=0;
function fixtureSse(body,{missingUsage=false,reasoning=false}={}){
  const prompt=body.messages[0].content;
  const output=body.ignore_eos?body.max_tokens:2;
  const promptTokens=Math.max(32,prompt.split(/\s+/).length);
  const events=[
    {id:"cpu-fixture",model:"GLM-5.3-Flash-EXL3",choices:[{delta:{role:"assistant"}}]},
    {choices:[{delta:reasoning?{reasoning_content:"hidden"}:{content:" alpha€"}}]},
    {choices:[{delta:{content:" beta"},finish_reason:"length"}]},
  ];
  if(!missingUsage)events.push({choices:[],usage:{prompt_tokens:promptTokens,completion_tokens:output,total_tokens:promptTokens+output,
    prompt_tokens_details:{cached_tokens:0},completion_tokens_details:{reasoning_tokens:reasoning?1:0}}});
  return events.map(v=>"data: "+JSON.stringify(v)+"\r\n\r\n").join("")+"data: [DONE]\r\n\r\n";
}
async function fakeTransport(url,options){
  const body=JSON.parse(options.body);
  fixtureRequests.push({url:String(url),body});
  fixtureActive++;fixturePeak=Math.max(fixturePeak,fixtureActive);
  const wire=Buffer.from(fixtureSse(body));
  const euro=wire.indexOf(Buffer.from("€"));
  const pieces=[wire.subarray(0,euro+1),wire.subarray(euro+1,euro+2),wire.subarray(euro+2)];
  let i=0,closed=false;
  const stream=new ReadableStream({
    async pull(controller){
      await new Promise(r=>setTimeout(r,3));
      if(closed)return;
      if(options.signal?.aborted){closed=true;fixtureActive--;controller.error(new DOMException("Aborted","AbortError"));return;}
      if(i<pieces.length){controller.enqueue(pieces[i++]);return;}
      closed=true;fixtureActive--;controller.close();
    },
    cancel(){if(!closed){closed=true;fixtureActive--;}}
  });
  return new Response(stream,{status:200,headers:{"content-type":"text/event-stream"}});
}
setTransportForCpuTests(fakeTransport);
const collector=await loadCollectors(state);
function fixtureHealth(count=0){return {ok:true,busy:false,requests_running:0,requests_total:count,context_length:1048576,
  streams:{max:8,decoding:0,prefilling:0,filling:0,paused:0}};}
const commonArgs=()=>parseArgs(["--url","http://127.0.0.1:18898","--plan"]);
async function nativeJob(manager,opts){
  let resolve;
  const finished=new Promise(r=>resolve=r);
  manager.setEventSink((kind,job)=>resolve(manager.getJob(job.benchId)||job));
  const started=manager.start({sparkId:"cpu",lanIp:"127.0.0.1",port:18898,modelId:"GLM-5.3-Flash-EXL3",...opts});
  return {started,finished};
}
const noNetworkNames=[
  ["reject production8888",["--url","http://127.0.0.1:8888","--plan"]],
  ["reject public endpoint",["--url","https://example.com:18898","--plan"]],
  ["reject credentials",["--url","http://u:p@127.0.0.1:18898","--plan"]],
  ["reject path injection",["--url","http://127.0.0.1:18898/other","--plan"]],
  ["reject configured P6",["--url","http://127.0.0.1:18898","--serving-lanes","6","--plan"]],
  ["reject non-study concurrency",["--url","http://127.0.0.1:18898","--concurrency","3","--plan"]],
  ["reject duplicate C1",["--url","http://127.0.0.1:18898","--concurrency","1,1","--plan"]],
  ["reject output outside benchmark",["--url","http://127.0.0.1:18898","--output","/tmp/unowned"]],
];
for(const [name,argv]of noNetworkNames)test(name,()=>assert.throws(()=>parseArgs(argv)));

test("exact source and dependency pin integrity",()=>{
  const identity=sourceIdentity();
  assert.equal(identity.pin.historical_and_cutoff_source_are_same,true);
  assert.equal(identity.pin.undici_lock.version,"8.9.0");
  assert.ok(Object.values(identity.pin.historical_local_byte_matches).every(Boolean));
});
test("one actual job per type is32 decode requests",()=>{
  assert.equal(requestPlan(commonArgs()).expected_successful_inference_requests,32);
});
test("three full native repeats are96 decode requests",()=>{
  const args=commonArgs();args.repeats=3;
  assert.equal(requestPlan(args).decode_requests,96);
});
test("prefill native warmups/calibration/repeats total19",()=>{
  const args=commonArgs();args.suite="prefill";
  assert.equal(requestPlan(args).prefill_requests,19);
  assert.deepEqual(requestPlan(args).prefill_samples_by_target.map(v=>v.repeats),[3,3,3,2,2,1]);
});
test("supplemental replays add6 and are distinctly labeled",()=>{
  const args=commonArgs();args.suite="all";args.replay=true;
  const plan=requestPlan(args);
  assert.equal(plan.expected_successful_inference_requests,57);
  assert.match(plan.replay_scope,/not a SparkDash/);
});
test("busy or wrong-lane health fails before any inference",()=>{
  const args=commonArgs();
  assert.throws(()=>validateHealth({...fixtureHealth(),busy:true},args));
  const h=fixtureHealth();h.streams.max=6;
  assert.throws(()=>validateHealth(h,args));
});
test("foreign request delta or counter reset rejects receipt",()=>{
  assert.throws(()=>auditRequests({requests_total:0},{requests_total:3},[],2));
  assert.throws(()=>auditRequests({requests_total:2},{requests_total:1},[],1));
});
test("wire capture preserves split UTF8 and CRLF byte-exact",async()=>{
  const recorder=new Recorder();setRecorder(recorder);
  const body={model:"GLM-5.3-Flash-EXL3",messages:[{role:"user",content:"fixture"}],max_tokens:400,ignore_eos:true};
  const result=await collector.streaming.runStreamingRequest("http://127.0.0.1:18898/v1/chat/completions",body,
    AbortSignal.timeout(5000),{collectContent:true});
  setRecorder(null);
  assert.equal(result.completionTokens,400);
  assert.equal(result.decodeTokens,399);
  assert.match(result.content,/alpha€/);
  const raw=Buffer.concat(recorder.records[0].chunks.map(v=>v.value));
  assert.equal(sha256(raw),sha256(Buffer.from(fixtureSse(body))));
  assert.equal(recorder.records[0].complete,true);
});
test("authoritative usage beats content/chunk estimates in native metric",async()=>{
  const result=await collector.streaming.runStreamingRequest("http://127.0.0.1:18898/v1/chat/completions",
    {messages:[{role:"user",content:"fixture"}],max_tokens:400,ignore_eos:true},AbortSignal.timeout(5000),{debug:true});
  assert.equal(result.completionTokens,400);assert.equal(result.decodeTokens,399);
  assert.ok(result.decodeMs>0);assert.ok(result.decodeTps>0);
});
test("receipt rejects missing DONE, missing usage, reasoning or invalid SSE",()=>{
  const good=parseSse(fixtureSse({messages:[{content:"fixture"}],max_tokens:400,ignore_eos:true}));
  const before={requests_total:0},after={requests_total:1};
  const record={status:200,complete:true,parsed:good};
  assert.doesNotThrow(()=>auditRequests(before,after,[record],1));
  for(const parsed of [{...good,done:false},{...good,usage:null},{...good,reasoningPieces:1},{...good,invalid:["broken"]}])
    assert.throws(()=>auditRequests(before,after,[{...record,parsed}],1));
});
test("native decode manager retains original warmup and all C1/C2/C4/C8 prompts",async()=>{
  fixtureRequests=[];fixturePeak=0;fixtureActive=0;
  const recorder=new Recorder();setRecorder(recorder);
  const manager=new collector.decode.DecodeBenchManager(path.join(state,"decode-history.json"),path.join(state,"decode-active.json"));
  const {finished}=await nativeJob(manager,{promptType:"code",concurrencies:[1,2,4,8],maxTokens:400});
  const job=await finished;setRecorder(null);
  validateJob(job,"decode",commonArgs());
  assert.equal(fixtureRequests.length,16);assert.equal(fixturePeak,8);
  const warm=fixtureRequests[0].body;
  assert.equal(warm.messages[0].content,collector.prompts.DECODE_CODE_WARMUP_PROMPT);
  assert.equal(warm.max_tokens,32);assert.ok(!("min_tokens" in warm));assert.ok(!("ignore_eos" in warm));assert.ok(!("stop" in warm));
  assert.deepEqual(warm.chat_template_kwargs,{enable_thinking:false,thinking:false,thinking_mode:"disabled"});
  assert.ok(!("reasoning_effort" in warm));
  const measured=fixtureRequests.slice(1).map(v=>v.body);
  assert.deepEqual(measured.map(v=>v.messages[0].content),[1,2,4,8].flatMap(n=>collector.prompts.pickDecodeBenchPrompts(n,"code")));
  assert.ok(measured.every(v=>v.min_tokens===400&&v.ignore_eos===true&&Array.isArray(v.stop)&&v.stop.length===0));
  assert.deepEqual(job.results.map(v=>v.totalDecodeTokens),[399,798,1596,3192]);
  assert.ok(recorder.records.every(v=>v.complete));
});
test("native prefill manager preserves two warmups three calibrations fresh repeats",async()=>{
  fixtureRequests=[];fixturePeak=0;fixtureActive=0;
  const recorder=new Recorder();setRecorder(recorder);
  const manager=new collector.prefill.PrefillBenchManager(path.join(state,"prefill-history.json"),path.join(state,"prefill-active.json"));
  const args=commonArgs();args.sizes=[8192,65536];
  const {finished}=await nativeJob(manager,{contextSizes:args.sizes,contextLength:args.context});
  const job=await finished;setRecorder(null);
  validateJob(job,"prefill",args);
  assert.equal(fixtureRequests.length,10);assert.equal(fixturePeak,1);
  assert.ok(fixtureRequests.every(v=>v.body.max_tokens===8&&!v.body.ignore_eos&&!v.body.min_tokens));
  const salts=fixtureRequests.map(v=>v.body.messages[0].content.match(/^\[prefill-bench ([^\]]+)\]/)[1]);
  assert.equal(new Set(salts).size,10);
  assert.deepEqual(job.results.map(v=>v.samples),[3,2]);
});
test("prefill corrected rate excludes cached tokens and caps correction",()=>{
  const value=collector.prefill.rateFromSample({promptTokens:1000,cachedTokens:500,ttftMs:100,serverPromptMs:null,serverPromptN:null},100);
  assert.equal(value.tps,10000);assert.equal(value.lowConfidence,true);
  assert.equal(collector.prefill.rateFromSample({promptTokens:1000,cachedTokens:1000,ttftMs:100},0).tps,0);
});
test("prefill even repeats select lower median sample rather than arithmetic median",async()=>{
  let index=0;
  const result=await collector.prefill.runPrefillMeasured({targetTokens:65536},0,null,async()=>({
    targetTokens:65536,promptTokens:1000,promptChars:4000,cachedTokens:0,ttftMs:++index===1?1000:500,
    ttftContentMs:1,completionTokens:2,durationMs:1,error:null}));
  assert.equal(result.samples,2);assert.equal(result.prefillTps,1000);assert.equal(result.ttftMs,1000);
});
test("native size-dependent timeout formula remains intact",()=>{
  assert.equal(collector.prefill.timeoutMsForSize(8192),125536);
  assert.equal(collector.prefill.timeoutMsForSize(262144),2157152);
  assert.equal(collector.prefill.timeoutMsForSize(999999),2700000);
});
test("partial prefill and truncated fixed-cap decode fail receipt gates",()=>{
  const args=commonArgs();
  assert.throws(()=>validateJob({status:"completed",results:[{notice:"partial"}]},"prefill",args));
  assert.throws(()=>validateJob({status:"completed",results:[{totalCompletionTokens:10}]},"decode",args));
});
test("offline plan performs zero HTTP and writes no output",async()=>{
  let calls=0;const original=globalThis.fetch,log=console.log;
  globalThis.fetch=async()=>{calls++;throw Error("No plan HTTP");};
  console.log=()=>{};
  try{assert.equal(await main(["--url","http://127.0.0.1:18898","--suite","all","--plan"]),0);}
  finally{globalThis.fetch=original;console.log=log;}
  assert.equal(calls,0);
});
test("complete CLI uses actual collectors and writes auditable raw requests without HTTP",async()=>{
  fixtureRequests=[];fixtureActive=0;fixturePeak=0;
  const output=path.join(ROOT,"runs","cpu-"+Date.now()+"-"+process.pid),log=console.log;
  globalThis.fetch=async url=>{
    if(String(url).endsWith("/health"))return new Response(JSON.stringify(fixtureHealth(fixtureRequests.length)),{status:200});
    if(String(url).endsWith("/v1/models"))return new Response(JSON.stringify({data:[{id:"GLM-5.3-Flash-EXL3"}]}),{status:200});
    throw Error("CPU fixtures prohibit real network");
  };
  console.log=()=>{};
  try{
    assert.equal(await main(["--url","http://127.0.0.1:18898","--suite","all","--output",output]),0);
  }finally{globalThis.fetch=async()=>{throw Error("No CPU HTTP");};console.log=log;}
  const result=JSON.parse(fs.readFileSync(path.join(output,"result.json")));
  assert.equal(result.passed,true);assert.equal(result.audit.requests_delta,51);
  assert.equal(result.summary.decode.length,8);assert.equal(result.summary.prefill.length,6);
  const raws=fs.readdirSync(path.join(output,"requests")).filter(v=>v.endsWith(".request-response.json"));
  assert.equal(raws.length,51);
  assert.ok(raws.every(name=>JSON.parse(fs.readFileSync(path.join(output,"requests",name))).parsed.done));
  journal("cpu-full-cli-verified",{output:path.relative(ROOT,output),mocked_requests:51,real_http_requests:0});
});

test("matched comparison refuses setting/source differences",()=>{
  const settings={suite:"decode",types:["prose"],concurrency:[1],maxTokens:400,repeats:1,sizes:[8192],lanes:8,context:1048576,model:"GLM-5.3-Flash-EXL3",debug:false};
  const baseline={passed:true,audit:{exclusive_counter_check_passed:true},source_pin:"same",source_pin_sha256:"same",
    settings,summary:{decode:[{type:"prose",concurrency:1,aggregate_decode_tps_median:100,aggregate_decode_tps_range:[100,100]}],prefill:[]}};
  const candidate=structuredClone(baseline);candidate.summary.decode[0].aggregate_decode_tps_median=110;
  assert.ok(Math.abs(compareResults(baseline,candidate).decode[0].change_percent-10)<1e-8);
  candidate.settings.maxTokens=500;
  assert.throws(()=>compareResults(baseline,candidate));
});
test("decode comparison preserves mismatch evidence and rejects prompt differences",()=>{
  assert.equal(compareDecodeOutputs({key:{content_sha256:"same",completion_tokens:400}},{key:{content_sha256:"same",completion_tokens:400}}).all_decoded_text_and_usage_match,true);
  assert.equal(compareDecodeOutputs({key:{content_sha256:"a"}},{key:{content_sha256:"b"}}).all_decoded_text_and_usage_match,false);
  assert.throws(()=>compareDecodeOutputs({a:{}},{b:{}}));
});

test.after(async()=>{
  setRecorder(null);globalThis.fetch=originalFetch;
  await collector.streaming.closeLlmStreamAgent();
});
