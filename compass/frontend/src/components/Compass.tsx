import { useEffect, useRef, useState } from 'react';
import parse, { Element } from 'html-react-parser';
import { icons } from 'lucide-react';
import { createEngine } from '@/prototype/engine';
import { Chart } from './Chart';
type Engine = ReturnType<typeof createEngine>;
export function Compass({ initialScreen }: { initialScreen: string }) {
 const [view,setView]=useState({html:'',page:initialScreen,revision:0}); const engine=useRef<Engine|null>(null);
 useEffect(()=>{const e=createEngine((html:string,page:string)=>{setView(v=>({html,page,revision:v.revision+1})); if(location.hash!==`#/screens/${page}`)history.pushState({},'',`#/screens/${page}`);},initialScreen);engine.current=e;
 const back=()=>{const id=location.hash.split('/').pop();if(id)e.navigate(id);};window.addEventListener('popstate',back);window.addEventListener('hashchange',back);
 return()=>{e.dispose();engine.current=null;window.removeEventListener('popstate',back);window.removeEventListener('hashchange',back);};},[]);
 return <><div className="export-banner"><span>Демонстрация · данные учебные · изменения хранятся только в памяти вкладки</span><a href="#/capture">Запись с камеры и микрофона</a></div><div id="razuma-ux-root" onClick={event=>{const el=(event.target as HTMLElement).closest<HTMLButtonElement>('[data-action]');if(el&&!el.disabled){event.preventDefault();engine.current?.action(el.dataset.action!);}}} onChange={event=>{const el=event.target as HTMLInputElement;const value=el.type==='checkbox'?el.checked:el.value;if(el.dataset.bind)engine.current?.bind(el.dataset.bind,value);if(el.dataset.control)engine.current?.control(el.dataset.control,el.value);}}>
 <div key={view.revision}>{view.html?parse(view.html,{replace(node){if(!(node instanceof Element))return; if(node.attribs['data-chart'])return <Chart {...JSON.parse(node.attribs['data-chart'])}/>;if(node.attribs['data-lucide']){const name=node.attribs['data-lucide'].replace(/(^|-)(\w)/g,(_,a,b)=>b.toUpperCase()) as keyof typeof icons;const Icon=icons[name];return Icon?<Icon size={16} aria-hidden/>:<span/>;}}}):<p role="status">Загружаем интерфейс…</p>}</div></div></>;
}
