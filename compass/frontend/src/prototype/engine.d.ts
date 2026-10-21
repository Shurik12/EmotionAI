export interface DemoEngine { action(value:string):void; bind(path:string,value:string|boolean):void; control(name:string,value:string):void; navigate(id:string):void; dispose():void; }
export function createEngine(publish:(html:string,page:string)=>void,initialScreen:string):DemoEngine;
