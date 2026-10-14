#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {ROOT,saveFile,sha256} from "./capture.mjs";
export function compareResults(baseline,candidate){
  if(baseline.passed!==true||candidate.passed!==true||!baseline.audit?.exclusive_counter_check_passed||!candidate.audit?.exclusive_counter_check_passed)
    throw Error("Both benchmark receipts must pass exclusive request accounting");
  if(baseline.source_pin!==candidate.source_pin||baseline.source_pin_sha256!==candidate.source_pin_sha256)
    throw Error("Collector/source pin differs");
  for(const key of ["suite","types","concurrency","maxTokens","repeats","sizes","lanes","context","model","debug","replay"])
    if(JSON.stringify(baseline.settings[key])!==JSON.stringify(candidate.settings[key]))throw Error("Protocol differs: "+key);
  const decode=baseline.summary.decode.map(b=>{
    const c=candidate.summary.decode.find(v=>v.type===b.type&&v.concurrency===b.concurrency);
    if(!c)throw Error("Candidate decode row absent");
    return {type:b.type,concurrency:b.concurrency,baseline_aggregate_decode_tps_median:b.aggregate_decode_tps_median,
      candidate_aggregate_decode_tps_median:c.aggregate_decode_tps_median,
      change_percent:100*(c.aggregate_decode_tps_median/b.aggregate_decode_tps_median-1),
      baseline_range:b.aggregate_decode_tps_range,candidate_range:c.aggregate_decode_tps_range};
  });
  const prefill=baseline.summary.prefill.map(b=>{
    const c=candidate.summary.prefill.find(v=>v.targetTokens===b.targetTokens);
    if(!c||b.method!==c.method||b.samples!==c.samples)throw Error("Prefill sample/method differs");
    return {targetTokens:b.targetTokens,method:b.method,baseline_prefill_tps:b.prefillTps,candidate_prefill_tps:c.prefillTps,
      change_percent:100*(c.prefillTps/b.prefillTps-1),
      baseline_actual_prompt_tokens:b.promptTokens,candidate_actual_prompt_tokens:c.promptTokens,
      baseline_cached_prompt_tokens:b.cachedTokens,candidate_cached_prompt_tokens:c.cachedTokens,
      baseline_ttft_ms:b.ttftMs,candidate_ttft_ms:c.ttftMs,
      baseline_low_confidence:b.lowConfidence,candidate_low_confidence:c.lowConfidence};
  });
  return {schema:"sparkdash-actual-collector-matched-comparison/v1",passed:true,protocol_matched:true,decode,prefill,
    scope:["Protocol/source/settings matched; supplied runtime receipts and operator isolation establish hardware/software comparability separately.",
      "UUID-salted prefill samples preserve upstream cold behavior but actual prompt token counts can differ.",
      "These are new local measurements, not reconstructed Mia historical runs or a claim of historically identical UI flags/repeats."]};
}
function decodeOutputs(directory,settings){
  const result={};
  for(const name of fs.readdirSync(path.join(directory,"requests"))){
    if(!name.endsWith(".request-response.json"))continue;
    const record=JSON.parse(fs.readFileSync(path.join(directory,"requests",name)));
    const body=JSON.parse(record.body);
    if(record.context.kind!=="decode"||body.max_tokens!==settings.maxTokens)continue;
    const key=[record.context.group,record.context.progress?.currentConcurrency,record.bodySha256].join("/");
    if(key in result)throw Error("Duplicate decode request key");
    const text=record.parsed.events.flatMap(v=>v.choices||[]).map(v=>v.delta?.content||"").join("");
    result[key]={content_sha256:sha256(Buffer.from(text)),completion_tokens:record.parsed.usage.completion_tokens};
  }
  return result;
}
export function compareDecodeOutputs(baseline,candidate){
  if(JSON.stringify(Object.keys(baseline).sort())!==JSON.stringify(Object.keys(candidate).sort()))throw Error("Decode prompt/group identities differ");
  const mismatches=Object.keys(baseline).filter(key=>JSON.stringify(baseline[key])!==JSON.stringify(candidate[key]));
  return {requests_compared:Object.keys(baseline).length,all_decoded_text_and_usage_match:mismatches.length===0,mismatches};
}
async function main(){
  const argv=process.argv.slice(2),options={};
  while(argv.length){const key=argv.shift();if(!["--baseline","--candidate","--output"].includes(key)||!argv.length)throw Error("Use --baseline DIR --candidate DIR --output NEW.json");options[key.slice(2)]=path.resolve(argv.shift());}
  if(!options.baseline||!options.candidate||!options.output||!options.output.startsWith(ROOT+"/runs/")||fs.existsSync(options.output))
    throw Error("Three paths required; output must be NEW beneath benchmark/runs/");
  const load=directory=>JSON.parse(fs.readFileSync(path.join(directory,"result.json")));
  const baseline=load(options.baseline),candidate=load(options.candidate),result=compareResults(baseline,candidate);
  result.decode_output_consistency=compareDecodeOutputs(decodeOutputs(options.baseline,baseline.settings),decodeOutputs(options.candidate,candidate.settings));
  result.baseline_result_sha256=sha256(fs.readFileSync(path.join(options.baseline,"result.json")));
  result.candidate_result_sha256=sha256(fs.readFileSync(path.join(options.candidate,"result.json")));
  saveFile(options.output,result);
  console.log(JSON.stringify(result,null,2));
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href)
  main().catch(error=>{console.error(JSON.stringify({passed:false,error:error.message}));process.exitCode=1;});
