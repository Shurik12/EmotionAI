/** Contract for the future backend. Demo screens do not call this adapter yet. */
export type AnalysisStatus = 'queued' | 'processing' | 'completed' | 'failed';
export interface DiaryEntry { id:string; patientId:string; createdAt:string; text:string; mood:number|null; anxiety:number|null; sleepHours:number|null; mediaId?:string; }
export interface AnalysisJob { id:string; status:AnalysisStatus; modelVersion?:string; quality?:'sufficient'|'insufficient'; result?:{label:string; value:number}[]; error?:string; }
export interface ClinicalApi { saveEntry(entry:Omit<DiaryEntry,'id'|'createdAt'>):Promise<DiaryEntry>; uploadMedia(file:Blob,patientId:string):Promise<{mediaId:string}>; getAnalysis(jobId:string):Promise<AnalysisJob>; }
export function createHttpApi(base=import.meta.env.VITE_API_BASE_URL || ''):ClinicalApi {
 async function request<T>(path:string,init?:RequestInit):Promise<T>{if(!base)throw new Error('Сервер API не подключён. Укажите VITE_API_BASE_URL.');const response=await fetch(base.replace(/\/$/,'')+path,{...init,credentials:'same-origin',cache:'no-store'});if(!response.ok)throw new Error(`API: ${response.status}`);return response.json() as Promise<T>;}
 return {saveEntry:entry=>request('/entries',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(entry)}),uploadMedia(file,patientId){const form=new FormData();form.set('file',file);form.set('patientId',patientId);return request('/media',{method:'POST',body:form});},getAnalysis:id=>request('/analysis/'+encodeURIComponent(id))};
}
