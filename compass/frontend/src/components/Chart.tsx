export function Chart({ values, labels, label, max = 10 }: { values: number[]; labels: string[]; label: string; max?: number }) {
 const x=(i:number)=>58+i*410/Math.max(1,values.length-1), y=(v:number)=>158-v/max*120;
 return <svg className="rz-chart" viewBox="0 0 510 205" role="img" aria-label={label}><title>{label} · демонстрационные данные</title>{[0,max/2,max].map(v=><g key={v}><line x1="58" x2="468" y1={y(v)} y2={y(v)}/><text x="42" y={y(v)+4} textAnchor="end">{v}</text></g>)}<polyline className="rz-chart-line" points={values.map((v,i)=>`${x(i)},${y(v)}`).join(' ')}/>{values.map((v,i)=><g key={i}><circle cx={x(i)} cy={y(v)} r="4"/><text x={x(i)} y="182" textAnchor="middle">{labels[i]}</text></g>)}<text x="58" y="20">{max===10?'Баллы, 0–10':'Слов/мин'}</text></svg>;
}
