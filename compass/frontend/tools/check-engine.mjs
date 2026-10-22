import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createEngine } from '../src/prototype/engine.js';
const screens=JSON.parse(fs.readFileSync(new URL('../src/data/screens.json',import.meta.url)));
let html='',page='';
const e=createEngine((h,p)=>{html=h;page=p;},'P03');
for(const s of screens){e.navigate(s.id);assert.equal(page,s.id);assert.ok(html.includes(s.id));assert.ok(html.includes('<h1>'));}
e.action('role:patient');assert.equal(page,'P03');e.action('role:doctor');assert.equal(page,'D01');
e.navigate('U03');e.bind('form.otp','000000');e.action('verifyotp');assert.equal(page,'U03');e.bind('form.otp','123456');e.action('verifyotp');assert.equal(page,'D01');
e.navigate('P06');e.bind('form.diaryText','<script>alert(1)</script>');e.navigate('D04');assert.ok(!html.includes('<script>'));

// P08 is no longer the demo video-note recorder: it mounts the real depression
// analyzer, which the React side replaces by the data-react attribute.
e.navigate('P08');
assert.ok(html.includes('data-react="depression-analyzer"'),'P08 must mount the depression analyzer');
assert.ok(html.includes('Анализ признаков депрессии'),'P08 must be titled as the depression screen');
assert.ok(!html.includes('Пример видеокружочка'),'P08 must not render the old video-note recorder');

// Every icon the engine emits must be mapped explicitly in src/components/icons.ts.
// A namespace import from lucide-react would let a typo degrade silently into an
// invisible <span/>, and would also pull every icon into the bundle.
const iconSource=fs.readFileSync(new URL('../src/components/icons.ts',import.meta.url),'utf8');
const mapped=new Set([...iconSource.matchAll(/^\s*'([a-z0-9-]+)':/gm)].map(m=>m[1]));
assert.ok(mapped.size>=20,`icon map looks truncated: ${mapped.size} entries`);
const emitted=new Set();
for(const role of ['patient','doctor']){
  e.action(`role:${role}`);
  for(const s of screens){e.navigate(s.id);for(const m of html.matchAll(/data-lucide="([^"]+)"/g))emitted.add(m[1]);}
}
const unmapped=[...emitted].filter(n=>!mapped.has(n)).sort();
assert.deepEqual(unmapped,[],`icons missing from src/components/icons.ts: ${unmapped.join(', ')}`);

e.dispose();console.log(`PASS: 68 screens, roles, demo OTP, escaped text, ${emitted.size} icons mapped`);
