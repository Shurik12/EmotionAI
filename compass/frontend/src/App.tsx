import { useEffect, useState } from 'react';
import { Compass } from './components/Compass';
import { Recorder } from './components/Recorder';
import screens from './data/screens.json';
export function App(){
 const [route,setRoute]=useState(()=>location.hash.slice(1)||'/screens/P03');
 useEffect(()=>{const update=()=>setRoute(location.hash.slice(1)||'/screens/P03');window.addEventListener('hashchange',update);window.addEventListener('popstate',update);return()=>{window.removeEventListener('hashchange',update);window.removeEventListener('popstate',update);};},[]);
 if(route==='/capture')return <Recorder/>;
 const id=route.match(/^\/screens\/([PDU]\d{2})$/)?.[1];
 if(!id||!screens.some(s=>s.id===id))return <main className="capture"><h1>Экран не найден</h1><a href="#/screens/P03">На главный экран</a></main>;
 return <Compass initialScreen={id}/>;
}
