import { useEffect, useState } from 'react';
import { Compass } from './components/Compass';
import { Recorder } from './components/Recorder';
import screens from './data/screens.json';
// History routing: path is relative to the /compass/ base the SPA is served from.
// location.pathname = "/compass/P03" -> route "/P03".
function currentRoute(){
 const path=location.pathname;
 const i=path.indexOf('/compass');
 if(i>=0){
  const rest=path.slice(i+8).replace(/^\/+/, '').replace(/\/+$/, '');
  return rest?'/'+rest:'/P03';
 }
 return path===''?'/P03':path;
}
// history.replaceState/popstate don't fire a React state update by themselves;
// Compass.tsx's popstate listener is the only thing that re-reads the path.
// When Recorder.analyze() does replaceState and App is already mounted, we need
// to force a route re-read. Keep the registered callbacks in a shared set and
// let Recorder poke them via bumpRoute().
const routeListeners=new Set<()=>void>();
export function bumpRoute(){routeListeners.forEach(l=>l());}
export function App(){
 const [route,setRoute]=useState(currentRoute);
 useEffect(()=>{
  const update=()=>setRoute(currentRoute());
  routeListeners.add(update);
  window.addEventListener('popstate',update);
  return()=>{window.removeEventListener('popstate',update);routeListeners.delete(update);};
 },[]);
 if(route==='/capture')return <Recorder/>;
 const id=route.match(/^\/([PDU]\d{2})$/)?.[1];
 if(!id||!screens.some(s=>s.id===id))return <main className="capture"><h1>Экран не найден</h1><a href="/compass/P03">На главный экран</a></main>;
 return <Compass initialScreen={id}/>;
}
