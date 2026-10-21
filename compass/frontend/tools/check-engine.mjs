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
e.dispose();console.log('PASS: 68 screens, roles, demo OTP, escaped text');
