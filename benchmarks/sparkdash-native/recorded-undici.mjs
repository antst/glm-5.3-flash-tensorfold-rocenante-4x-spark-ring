// Separate instrumentation; byte-exact upstream source is never rewritten.
import {Agent,fetch as upstreamFetch} from "undici";
import {capturedFetch} from "./capture.mjs";
export {Agent};
let transport=upstreamFetch;
export function setTransportForCpuTests(fn){transport=fn;}
export function fetch(url,options){return capturedFetch(url,options,transport);}
